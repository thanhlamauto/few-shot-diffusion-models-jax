# jax_fsdm/diffusion_transformer.py
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
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02))(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02))(x)
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
    dropout_rate: float = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
        nn.initializers.xavier_uniform()
    )
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(
        stddev=1e-6
    )

    @nn.compact
    def __call__(self, inputs):
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
            features=self.mlp_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(inputs)
        x = nn.gelu(x)
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

    @nn.compact
    def __call__(self, x, c, context=None):
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
        x = x + (gate_msa[:, None] * attn_x)

        # Cross-attention for lag
        if self.mode_conditioning == "lag" and context is not None:
            x_norm_cross = nn.LayerNorm(use_bias=False, use_scale=False)(x)
            context_proj = nn.Dense(
                self.hidden_size, kernel_init=nn.initializers.xavier_uniform()
            )(context)
            cross_attn_x = nn.MultiHeadDotProductAttention(
                kernel_init=nn.initializers.xavier_uniform(), num_heads=self.num_heads
            )(x_norm_cross, context_proj, context_proj)
            x = x + cross_attn_x

        # MLP
        x_norm2 = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_x = MlpBlock(mlp_dim=int(self.hidden_size * self.mlp_ratio))(x_modulated2)
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
        x = modulate(nn.LayerNorm(use_bias=False, use_scale=False)(x), shift, scale)
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
                dummy_c = jnp.zeros((x.shape[0], self.context_channels), dtype=x.dtype)
                zero_context_proj = context_proj_layer(dummy_c)
                conditioning = t_emb + zero_context_proj  # FIX: consistent with c!=None case
        else:
            conditioning = t_emb

        for _ in range(self.depth):
            if self.mode_conditioning == "lag" and c is not None:
                x = DiTBlock(
                    self.hidden_size,
                    self.num_heads,
                    self.mlp_ratio,
                    self.context_channels,
                    self.mode_conditioning,
                )(x, conditioning, context=c)
            else:
                x = DiTBlock(
                    self.hidden_size,
                    self.num_heads,
                    self.mlp_ratio,
                    self.context_channels,
                    self.mode_conditioning,
                )(x, conditioning)

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


__all__ = ["DiT", "DiTBlock", "TimestepEmbedder", "LabelEmbedder", "PatchEmbed"]
# jax_fsdm/diffusion_transformer.py
import math
from typing import Any, Tuple, Optional
import flax.linen as nn
from flax.linen.initializers import xavier_uniform
import jax
from jax import lax
import jax.numpy as jnp
from einops import rearrange

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

from typing import Any, Callable, Optional, Tuple, Type, Sequence, Union

# Port of DiT with Few-Shot Diffusion Model (FSDM) support
# Based on https://github.com/facebookresearch/DiT/blob/main/models.py
# Extended with few-shot conditioning support (context_channels, mode_conditioning)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    KHÔNG THAY ĐỔI - Giữ nguyên từ code mẫu
    """
    hidden_size: int
    frequency_embedding_size: int = 256

    @nn.compact
    def __call__(self, t):
        x = self.timestep_embedding(t)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02))(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.normal(0.02))(x)
        return x

    def timestep_embedding(self, t, max_period=10000):
        t = jax.lax.convert_element_type(t, jnp.float32)
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = jnp.exp(-math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        return embedding

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    KHÔNG THAY ĐỔI - Giữ nguyên từ code mẫu
    """
    dropout_prob: float
    num_classes: int
    hidden_size: int

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            rng = self.make_rng('label_dropout')
            drop_ids = jax.random.bernoulli(rng, self.dropout_prob, (labels.shape[0],))
        else:
            drop_ids = force_drop_ids == 1
        labels = jnp.where(drop_ids, self.num_classes, labels)
        return labels
    
    @nn.compact
    def __call__(self, labels, train, force_drop_ids=None):
        embedding_table = nn.Embed(self.num_classes + 1, self.hidden_size, embedding_init=nn.initializers.normal(0.02))
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = embedding_table(labels)
        return embeddings

class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block.
    KHÔNG THAY ĐỔI - Giữ nguyên từ code mẫu
    """
    mlp_dim: int
    dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-6)

    @nn.compact
    def __call__(self, inputs):
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
                features=self.mlp_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(inputs)
        x = nn.gelu(x)
        output = nn.Dense(
                features=actual_out_dim,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init)(x)
        return output
    
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding 
    KHÔNG THAY ĐỔI - Giữ nguyên từ code mẫu
    """
    patch_size: int
    embed_dim: int
    bias: bool = True

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        patch_tuple = (self.patch_size, self.patch_size)
        num_patches = (H // self.patch_size)
        x = nn.Conv(self.embed_dim, patch_tuple, patch_tuple, use_bias=self.bias, padding="VALID", kernel_init=nn.initializers.xavier_uniform())(x)
        x = rearrange(x, 'b h w c -> b (h w) c', h=num_patches, w=num_patches)
        return x
    
def modulate(x, shift, scale):
    """FiLM modulation - KHÔNG THAY ĐỔI"""
    return x * (1 + scale[:, None]) + shift[:, None]

# Positional embedding functions - KHÔNG THAY ĐỔI
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    pos = pos.reshape(-1)
    out = jnp.einsum('m,d->md', pos, omega)
    emb_sin = jnp.sin(out)
    emb_cos = jnp.cos(out)
    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)
    return emb

def get_1d_sincos_pos_embed(embed_dim, length):
    return jnp.expand_dims(
        get_1d_sincos_pos_embed_from_grid(embed_dim, jnp.arange(length, dtype=jnp.float32)),
        0
    )

def get_2d_sincos_pos_embed(rng, embed_dim, length):
    """
    THAY ĐỔI: Thêm parameter rng để tương thích với self.param() trong Flax
    Nhưng rng không được sử dụng vì đây là deterministic function
    """
    grid_size = int(length ** 0.5)
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

################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    
    THAY ĐỔI CHÍNH:
    - Thêm context_channels và mode_conditioning parameters để hỗ trợ few-shot conditioning
    - Thêm cross-attention layer cho lag mode conditioning
    - Điều chỉnh adaLN modulation để xử lý context embedding
    """
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    context_channels: int = 0  # THÊM: số chiều của context embedding
    mode_conditioning: str = "film"  # THÊM: "film" hoặc "lag"

    @nn.compact
    def __call__(self, x, c, context=None):
        """
        THAY ĐỔI: Thêm parameter context cho lag mode
        x: (B, N, hidden_size) - patch tokens
        c: (B, hidden_size) - time embedding (và context nếu film mode)
        context: (B, N_c, context_channels) hoặc None - few-shot context tokens (cho lag mode)
        """
        # Tính toán adaLN modulation parameters
        # THAY ĐỔI: Xử lý cả film và lag mode
        if self.mode_conditioning == "film":
            # Film mode: c đã chứa cả time và context embedding
            c_mod = nn.silu(c)
            c_mod = nn.Dense(6 * self.hidden_size, kernel_init=nn.initializers.constant(0.))(c_mod)
        else:
            # Lag mode: chỉ dùng time embedding cho adaLN
            c_mod = nn.silu(c)
            c_mod = nn.Dense(6 * self.hidden_size, kernel_init=nn.initializers.constant(0.))(c_mod)
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(c_mod, 6, axis=-1)
        
        # Self-attention với adaLN-Zero
        x_norm = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x_modulated = modulate(x_norm, shift_msa, scale_msa)
        attn_x = nn.MultiHeadDotProductAttention(
            kernel_init=nn.initializers.xavier_uniform(),
            num_heads=self.num_heads)(x_modulated, x_modulated)
        x = x + (gate_msa[:, None] * attn_x)

        # THÊM: Cross-attention cho lag mode conditioning
        if self.mode_conditioning == "lag" and context is not None:
            x_norm_cross = nn.LayerNorm(use_bias=False, use_scale=False)(x)
            # Project context to hidden_size
            context_proj = nn.Dense(self.hidden_size, kernel_init=nn.initializers.xavier_uniform())(context)
            # Cross-attention: x attends to context
            cross_attn_x = nn.MultiHeadDotProductAttention(
                kernel_init=nn.initializers.xavier_uniform(),
                num_heads=self.num_heads)(x_norm_cross, context_proj, context_proj)
            x = x + cross_attn_x

        # MLP với adaLN-Zero
        x_norm2 = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        x_modulated2 = modulate(x_norm2, shift_mlp, scale_mlp)
        mlp_x = MlpBlock(mlp_dim=int(self.hidden_size * self.mlp_ratio))(x_modulated2)
        x = x + (gate_mlp[:, None] * mlp_x)
        
        return x
    
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    
    THAY ĐỔI: Thêm hỗ trợ context_channels và mode_conditioning
    """
    patch_size: int
    out_channels: int
    hidden_size: int
    context_channels: int = 0  # THÊM
    mode_conditioning: str = "film"  # THÊM

    @nn.compact
    def __call__(self, x, c):
        """
        THAY ĐỔI: Xử lý cả film và lag mode
        c: (B, hidden_size) hoặc (B, hidden_size + context_channels) tùy mode
        """
        c = nn.silu(c)
        c = nn.Dense(2 * self.hidden_size, kernel_init=nn.initializers.constant(0))(c)
        shift, scale = jnp.split(c, 2, axis=-1)
        x = modulate(nn.LayerNorm(use_bias=False, use_scale=False)(x), shift, scale)
        x = nn.Dense(self.patch_size * self.patch_size * self.out_channels, 
                     kernel_init=nn.initializers.constant(0))(x)
        return x

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    
    THAY ĐỔI CHÍNH:
    - Thêm context_channels và mode_conditioning để hỗ trợ few-shot conditioning
    - Thêm context projection layer cho film mode
    - Thay đổi forward signature để nhận c parameter (few-shot conditioning)
    - Xử lý padding cho 28x28 images (MNIST)
    - Thay đổi input format từ (B, H, W, C) sang (B, C, H, W) để tương thích với PyTorch
    """
    patch_size: int
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float
    class_dropout_prob: float
    num_classes: int
    learn_sigma: bool = False
    # THÊM: Few-shot conditioning parameters
    context_channels: int = 0
    mode_conditioning: str = "film"  # "film" hoặc "lag"

    @nn.compact
    def __call__(self, x, t, c=None, y=None, train=False, force_drop_ids=None):
        """
        THAY ĐỔI: Thêm parameter c cho few-shot conditioning
        
        Args:
            x: (B, C, H, W) image tensor - THAY ĐỔI từ (B, H, W, C) trong code mẫu
            t: (B,) timesteps
            c: conditioning tensor
                - If mode_conditioning == "film": (B, context_channels) or None
                - If mode_conditioning == "lag": (B, N_c, context_channels) or None
            y: (B,) class labels (optional)
            train: training mode flag
            force_drop_ids: for classifier-free guidance
        
        Returns:
            x: (B, C, H, W) output tensor - THAY ĐỔI từ (B, H, W, C)
        """
        # THAY ĐỔI: Convert từ (B, C, H, W) sang (B, H, W, C) cho JAX
        if len(x.shape) == 4 and x.shape[1] != x.shape[-1]:  # (B, C, H, W)
            x = jnp.transpose(x, (0, 2, 3, 1))  # -> (B, H, W, C)
        
        batch_size = x.shape[0]
        input_size = x.shape[1]
        in_channels = x.shape[-1]
        out_channels = in_channels if not self.learn_sigma else in_channels * 2
        
        # THÊM: Xử lý padding cho 28x28 images
        pad_applied = False
        if input_size == 28:
            # Pad to 32x32
            x = jnp.pad(x, ((0, 0), (2, 2), (2, 2), (0, 0)), mode='constant')
            input_size = 32
            pad_applied = True
        
        num_patches = (input_size // self.patch_size) ** 2
        num_patches_side = input_size // self.patch_size
        
        # Positional embedding
        pos_embed = self.param("pos_embed", get_2d_sincos_pos_embed, self.hidden_size, num_patches)
        pos_embed = jax.lax.stop_gradient(pos_embed)
        
        # Patch embedding
        x = PatchEmbed(self.patch_size, self.hidden_size)(x)  # (B, num_patches, hidden_size)
        x = x + pos_embed
        
        # Timestep embedding
        t_emb = TimestepEmbedder(self.hidden_size)(t)  # (B, hidden_size)
        
        # Class label embedding (optional)
        if y is not None:
            y_emb = LabelEmbedder(self.class_dropout_prob, self.num_classes, self.hidden_size)(
                y, train=train, force_drop_ids=force_drop_ids)  # (B, hidden_size)
            t_emb = t_emb + y_emb
        
        # THÊM: Xử lý few-shot conditioning
        if self.mode_conditioning == "film":
            # Film mode: combine time embedding với context embedding
            # Create Dense layer unconditionally (required for Flax @nn.compact)
            context_proj_layer = nn.Dense(self.hidden_size, kernel_init=nn.initializers.xavier_uniform())
            if c is not None:
                # c: (B, context_channels)
                # Project context to hidden_size và combine với time embedding
                context_proj = context_proj_layer(c)
                conditioning = t_emb + context_proj  # (B, hidden_size)
            else:
                # MUST call layer with dummy input to initialize parameters
                # AND add the zero projection to maintain consistent computation graph
                dummy_c = jnp.zeros((x.shape[0], self.context_channels), dtype=x.dtype)
                zero_context_proj = context_proj_layer(dummy_c)
                conditioning = t_emb + zero_context_proj  # FIX: consistent with c!=None case
        else:
            # Lag mode: giữ time embedding riêng, context sẽ dùng cho cross-attention
            conditioning = t_emb
        
        # Apply DiT blocks
        for _ in range(self.depth):
            if self.mode_conditioning == "lag" and c is not None:
                # Pass context to cross-attention
                x = DiTBlock(
                    self.hidden_size, 
                    self.num_heads, 
                    self.mlp_ratio,
                    self.context_channels,
                    self.mode_conditioning
                )(x, conditioning, context=c)
            else:
                x = DiTBlock(
                    self.hidden_size, 
                    self.num_heads, 
                    self.mlp_ratio,
                    self.context_channels,
                    self.mode_conditioning
                )(x, conditioning)
        
        # Final layer
        x = FinalLayer(
            self.patch_size, 
            out_channels, 
            self.hidden_size,
            self.context_channels,
            self.mode_conditioning
        )(x, conditioning)  # (B, num_patches, p*p*c)
        
        # Reshape to image
        x = jnp.reshape(x, (batch_size, num_patches_side, num_patches_side, 
                            self.patch_size, self.patch_size, out_channels))
        x = jnp.einsum('bhwpqc->bhpwqc', x)
        x = rearrange(x, 'B H P W Q C -> B (H P) (W Q) C', H=int(num_patches_side), W=int(num_patches_side))
        
        # THÊM: Remove padding nếu đã apply
        if pad_applied:
            x = x[:, 2:-2, 2:-2, :]  # Remove padding
        
        # THAY ĐỔI: Convert lại về (B, C, H, W) để tương thích với PyTorch interface
        x = jnp.transpose(x, (0, 3, 1, 2))  # (B, H, W, C) -> (B, C, H, W)
        
        return x