# jax_fsdm/diffusion_transformer.py
from typing import Any, Callable, Optional, Tuple, Type, Sequence, Union
from jax import lax
from typing import Any, Tuple, Optional
import math
from typing import Any, Tuple, Optional, Callable
import flax.linen as nn
from flax.linen.initializers import xavier_uniform
import jax
import jax.numpy as jnp
from einops import rearrange

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

# Port of DiT with Few-Shot Diffusion Model (FSDM) support
# Extended with few-shot conditioning support (context_channels, mode_conditioning)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = jnp.einsum("m,d->md", pos, omega)
    emb_sin = jnp.sin(out)
    emb_cos = jnp.cos(out)
    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    return jnp.expand_dims(
        get_1d_sincos_pos_embed_from_grid(
            embed_dim, jnp.arange(length, dtype=jnp.float32)
        ),
        0,
    )


def get_2d_sincos_pos_embed(rng, embed_dim, length):
    """
    rng included for Flax param signature; not used (deterministic).
    """
    grid_size = int(length**0.5)
    assert grid_size * grid_size == length

    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        emb = jnp.concatenate([emb_h, emb_w], axis=1)
        return emb

    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)
    grid = jnp.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return jnp.expand_dims(pos_embed, 0)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    hidden_size: int
    frequency_embedding_size: int = 256

    @nn.compact
    def __call__(self, t):
        x = self.timestep_embedding(t)
        x = nn.Dense(self.hidden_size,
                     kernel_init=nn.initializers.normal(0.02))(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_size,
                     kernel_init=nn.initializers.normal(0.02))(x)
        return x

    def timestep_embedding(self, t, max_period=10000):
        t = jax.lax.convert_element_type(t, jnp.float32)
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = jnp.exp(
            -math.log(max_period)
            * jnp.arange(start=0, stop=half, dtype=jnp.float32)
            / half
        )
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        return embedding


class LabelEmbedder(nn.Module):
    """Embeds class labels; supports classifier-free guidance dropout."""

    dropout_prob: float
    num_classes: int
    hidden_size: int

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            rng = self.make_rng("label_dropout")
            drop_ids = jax.random.bernoulli(
                rng, self.dropout_prob, (labels.shape[0],)
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = jnp.where(drop_ids, self.num_classes, labels)
        return labels

    @nn.compact
    def __call__(self, labels, train, force_drop_ids=None):
        embedding_table = nn.Embed(
            self.num_classes + 1,
            self.hidden_size,
            embedding_init=nn.initializers.normal(0.02),
        )
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = embedding_table(labels)
        return embeddings


class MlpBlock(nn.Module):
    """Transformer MLP block."""

    mlp_dim: int
    dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = 0.0
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
        nn.initializers.xavier_uniform()
    )
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(
        stddev=1e-6
    )

    @nn.compact
    def __call__(self, inputs, train: bool = True):
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
            features=self.mlp_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(inputs)
        x = nn.gelu(x)
        # Apply dropout after GELU activation
        if self.dropout_rate > 0:
            x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)
        output = nn.Dense(
            features=actual_out_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        return output


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""

    patch_size: int
    embed_dim: int
    bias: bool = True

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        patch_tuple = (self.patch_size, self.patch_size)
        num_patches = H // self.patch_size
        x = nn.Conv(
            self.embed_dim,
            patch_tuple,
            patch_tuple,
            use_bias=self.bias,
            padding="VALID",
            kernel_init=nn.initializers.xavier_uniform(),
        )(x)
        x = rearrange(x, "b h w c -> b (h w) c", h=num_patches, w=num_patches)
        return x


def modulate(x, shift, scale):
    """FiLM modulation."""
    return x * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Supports film and lag (cross-attn) conditioning.
    """

    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    context_channels: int = 0
    mode_conditioning: str = "film"  # "film" or "lag"
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, c, context=None, train: bool = False):
        # adaLN modulation params
        c_mod = nn.silu(c)
        c_mod = nn.Dense(
            6 * self.hidden_size, kernel_init=nn.initializers.constant(0.0)
        )(c_mod)
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = jnp.split(c_mod, 6, axis=-1)

        # Self-attention
        x_norm = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x_modulated = modulate(x_norm, shift_msa, scale_msa)
        attn_x = nn.MultiHeadDotProductAttention(
            kernel_init=nn.initializers.xavier_uniform(), num_heads=self.num_heads
        )(x_modulated, x_modulated)
        attn_x = gate_msa[:, None] * attn_x
        # Apply dropout to self-attention output
        if self.dropout_rate > 0:
            attn_x = nn.Dropout(self.dropout_rate)(attn_x, deterministic=not train)
        x = x + attn_x

        # Cross-attention for lag
        if self.mode_conditioning == "lag" and context is not None:
            x_norm_cross = nn.LayerNorm(use_bias=False, use_scale=False)(x)
            context_proj = nn.Dense(
                self.hidden_size, kernel_init=nn.initializers.xavier_uniform()
            )(context)
            cross_attn_x = nn.MultiHeadDotProductAttention(
                kernel_init=nn.initializers.xavier_uniform(), num_heads=self.num_heads
            )(x_norm_cross, context_proj, context_proj)
            # Apply dropout to cross-attention output
            if self.dropout_rate > 0:
                cross_attn_x = nn.Dropout(self.dropout_rate)(cross_attn_x, deterministic=not train)
            x = x + cross_attn_x

        # MLP
        x_norm2 = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_x = MlpBlock(
            mlp_dim=int(self.hidden_size * self.mlp_ratio),
            dropout_rate=self.dropout_rate
        )(x_modulated2, train=train)
        x = x + (gate_mlp[:, None] * mlp_x)
        return x


class FinalLayer(nn.Module):
    """Final projection back to image patches."""

    patch_size: int
    out_channels: int
    hidden_size: int
    context_channels: int = 0
    mode_conditioning: str = "film"

    @nn.compact
    def __call__(self, x, c):
        c = nn.silu(c)
        c = nn.Dense(2 * self.hidden_size, kernel_init=nn.initializers.constant(0))(
            c
        )
        shift, scale = jnp.split(c, 2, axis=-1)
        x = modulate(nn.LayerNorm(use_bias=False,
                     use_scale=False)(x), shift, scale)
        x = nn.Dense(
            self.patch_size * self.patch_size * self.out_channels,
            kernel_init=nn.initializers.constant(0),
        )(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with Transformer backbone and few-shot conditioning.
    """

    patch_size: int
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float
    class_dropout_prob: float
    num_classes: int
    learn_sigma: bool = False
    context_channels: int = 0
    mode_conditioning: str = "film"  # "film" or "lag"
    dropout_rate: float = 0.0
    cross_attn_layers: str = "all"  # "all" or comma-separated layer indices (e.g., "2,3,4,5")
    use_remat: bool = False  # Use gradient checkpointing (remat) to trade compute for memory

    @nn.compact
    def __call__(self, x, t, c=None, y=None, train=False, force_drop_ids=None):
        if len(x.shape) == 4 and x.shape[1] != x.shape[-1]:
            x = jnp.transpose(x, (0, 2, 3, 1))  # (B,C,H,W)->(B,H,W,C)

        batch_size = x.shape[0]
        input_size = x.shape[1]
        in_channels = x.shape[-1]
        out_channels = in_channels if not self.learn_sigma else in_channels * 2

        pad_applied = False
        if input_size == 28:
            x = jnp.pad(x, ((0, 0), (2, 2), (2, 2), (0, 0)), mode="constant")
            input_size = 32
            pad_applied = True

        num_patches = (input_size // self.patch_size) ** 2
        num_patches_side = input_size // self.patch_size

        pos_embed = self.param(
            "pos_embed", get_2d_sincos_pos_embed, self.hidden_size, num_patches
        )
        pos_embed = jax.lax.stop_gradient(pos_embed)

        x = PatchEmbed(self.patch_size, self.hidden_size)(x)
        x = x + pos_embed

        t_emb = TimestepEmbedder(self.hidden_size)(t)

        if y is not None:
            y_emb = LabelEmbedder(
                self.class_dropout_prob, self.num_classes, self.hidden_size
            )(y, train=train, force_drop_ids=force_drop_ids)
            t_emb = t_emb + y_emb

        if self.mode_conditioning == "film":
            # Create Dense layer unconditionally (required for Flax @nn.compact)
            context_proj_layer = nn.Dense(
                self.hidden_size, kernel_init=nn.initializers.xavier_uniform()
            )
            if c is not None:
                context_proj = context_proj_layer(c)
                conditioning = t_emb + context_proj
            else:
                # MUST call layer with dummy input to initialize parameters
                # AND add the zero projection to maintain consistent computation graph
                dummy_c = jnp.zeros(
                    (x.shape[0], self.context_channels), dtype=x.dtype)
                zero_context_proj = context_proj_layer(dummy_c)
                conditioning = t_emb + zero_context_proj  # FIX: consistent with c!=None case
        else:
            conditioning = t_emb

        # Parse cross_attn_layers: "all" or comma-separated indices like "2,3,4,5"
        if self.cross_attn_layers == "all":
            cross_attn_layer_set = set(range(self.depth))
        else:
            cross_attn_layer_set = set(int(x.strip()) for x in self.cross_attn_layers.split(",") if x.strip())
        
        # Wrap DiTBlock with remat if enabled
        BlockClass = DiTBlock
        if self.use_remat:
            BlockClass = nn.remat(DiTBlock)

        for layer_idx in range(self.depth):
            # Only use cross-attention at specified layers
            use_cross_attn = (self.mode_conditioning == "lag" and c is not None and layer_idx in cross_attn_layer_set)
            
            if use_cross_attn:
                x = BlockClass(
                    self.hidden_size,
                    self.num_heads,
                    self.mlp_ratio,
                    self.context_channels,
                    self.mode_conditioning,
                    dropout_rate=self.dropout_rate,
                )(x, conditioning, context=c, train=train)
            else:
                x = BlockClass(
                    self.hidden_size,
                    self.num_heads,
                    self.mlp_ratio,
                    self.context_channels,
                    self.mode_conditioning,
                    dropout_rate=self.dropout_rate,
                )(x, conditioning, train=train)

        x = FinalLayer(
            self.patch_size,
            out_channels,
            self.hidden_size,
            self.context_channels,
            self.mode_conditioning,
        )(x, conditioning)

        x = jnp.reshape(
            x,
            (
                batch_size,
                num_patches_side,
                num_patches_side,
                self.patch_size,
                self.patch_size,
                out_channels,
            ),
        )
        x = jnp.einsum("bhwpqc->bhpwqc", x)
        x = rearrange(
            x,
            "B H P W Q C -> B (H P) (W Q) C",
            H=int(num_patches_side),
            W=int(num_patches_side),
        )

        if pad_applied:
            x = x[:, 2:-2, 2:-2, :]

        x = jnp.transpose(x, (0, 3, 1, 2))  # (B,C,H,W)
        return x


__all__ = ["DiT", "DiTBlock", "TimestepEmbedder",
           "LabelEmbedder", "PatchEmbed"]
