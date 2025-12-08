import jax.numpy as jnp
from flax import linen as nn
from einops import rearrange, repeat
from typing import Tuple, Literal, Optional


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class FeedForward(nn.Module):
    dim: int
    hidden_dim: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout)(x, deterministic=not train)
        x = nn.Dense(self.dim)(x)
        x = nn.Dropout(self.dropout)(x, deterministic=not train)
        return x


class LSA(nn.Module):
    dim: int
    heads: int = 8
    dim_head: int = 64
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool = True):
        """
        x: (b, n, dim)
        """
        inner_dim = self.heads * self.dim_head
        b, n, _ = x.shape

        # learnable log temperature (khởi tạo giống dim_head ** -0.5)
        temperature = self.param(
            "temperature",
            lambda key: jnp.log(
                jnp.array(self.dim_head ** -0.5, dtype=jnp.float32))
        )
        scale = jnp.exp(temperature)

        # qkv projection
        # (b, n, 3 * inner_dim)
        qkv = nn.Dense(inner_dim * 3, use_bias=False)(x)

        # tách q, k, v và reshape thành (b, heads, n, dim_head)
        q, k, v = jnp.split(qkv, 3, axis=-1)  # each: (b, n, inner_dim)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        # attention scores
        dots = jnp.einsum('bhid,bhjd->bhij', q, k) * scale  # (b, h, n, n)

        # mask self-attention trên đường chéo
        mask = jnp.eye(n, dtype=bool)[None, None, :, :]     # (1,1,n,n)
        mask_value = jnp.finfo(dots.dtype).min
        dots = jnp.where(mask, mask_value, dots)

        attn = nn.softmax(dots, axis=-1)
        attn = nn.Dropout(self.dropout)(attn, deterministic=not train)

        # apply attention
        out = jnp.einsum('bhij,bhjd->bhid', attn, v)        # (b, h, n, d)
        out = rearrange(out, 'b h n d -> b n (h d)')        # (b, n, inner_dim)

        out = nn.Dense(self.dim)(out)
        out = nn.Dropout(self.dropout)(out, deterministic=not train)
        return out


class TransformerLayer(nn.Module):
    dim: int
    heads: int
    dim_head: int
    mlp_dim: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool = True):
        # PreNorm + LSA
        y = nn.LayerNorm()(x)
        y = LSA(dim=self.dim, heads=self.heads, dim_head=self.dim_head, dropout=self.dropout)(
            y, train=train
        )
        x = x + y

        # PreNorm + FeedForward
        y = nn.LayerNorm()(x)
        y = FeedForward(dim=self.dim, hidden_dim=self.mlp_dim, dropout=self.dropout)(
            y, train=train
        )
        x = x + y
        return x


class Transformer(nn.Module):
    dim: int
    depth: int
    heads: int
    dim_head: int
    mlp_dim: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool = True):
        for i in range(self.depth):
            x = TransformerLayer(
                dim=self.dim,
                heads=self.heads,
                dim_head=self.dim_head,
                mlp_dim=self.mlp_dim,
                dropout=self.dropout,
                name=f"layer_{i}"
            )(x, train=train)
        return x


class SPT(nn.Module):
    """
    Shifted Patch Tokenization cho small sets:
    x_set: (b, ns, c, w, h) -> gom ns vào channel, rồi patchify.
    """
    dim: int
    patch_size: int
    channels: int = 3
    sample_size: int = 5   # ns tối đa (giống sample_size PyTorch)

    def setup(self):
        patch_dim = self.patch_size * self.patch_size * self.sample_size * self.channels
        self.norm = nn.LayerNorm()
        self.proj = nn.Dense(self.dim)

    def __call__(self, x_set):
        """
        x_set: (bs, ns, ch, w, h)
        """
        bs, ns, ch, w, h = x_set.shape

        # (b, ns, c, w, h) -> (b, c, ns, w, h)
        x = jnp.transpose(x_set, (0, 2, 1, 3, 4))
        # (b, c*ns, w, h)
        x = x.reshape(bs, ch * ns, w, h)

        p = self.patch_size
        # b c (h p1) (w p2) -> b (h w) (p1 p2 c)
        x = rearrange(
            x,
            "b c (hh p1) (ww p2) -> b (hh ww) (p1 p2 c)",
            p1=p,
            p2=p,
        )

        x = self.norm(x)
        x = self.proj(x)
        return x   # (b, num_patches, dim)


class sViT(nn.Module):
    """
    Generalization of ViT cho small sets (set of images)
    Dùng SPT + LSA-based Transformer.
    """
    image_size: Tuple[int, int]
    patch_size: int
    num_classes: int
    dim: int
    depth: int
    heads: int
    mlp_dim: int

    pool: Literal['cls', 'mean', 'none'] = 'cls'
    channels: int = 3
    dim_head: int = 64
    dropout: float = 0.0
    emb_dropout: float = 0.0
    ns: int = 5
    t_dim: int = 256

    def setup(self):
        image_height, image_width = pair(self.image_size)
        patch_height, patch_width = pair(self.patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            "Image dimensions must be divisible by the patch size."

        self.np = (image_height // patch_height) * (image_width // patch_width)
        self.num_patches = self.np

        assert self.pool in {'cls', 'mean', 'none'}, \
            "pool type must be either 'cls', 'mean' or 'none'"

        # SPT patch embedding
        self.to_patch_embedding = SPT(
            dim=self.dim,
            patch_size=patch_height,  # assume square patch
            channels=self.channels,
            sample_size=self.ns,
        )

        # +2 cho cls + time token
        self.pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=1.0),
            (1, self.num_patches + 2, self.dim),
        )
        self.cls_token = self.param(
            "cls_token",
            nn.initializers.normal(stddev=1.0),
            (1, 1, self.dim),
        )

        self.emb_dropout_layer = nn.Dropout(self.emb_dropout)

        self.transformer = Transformer(
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            dim_head=self.dim_head,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
        )

        self.mlp_norm = nn.LayerNorm()
        self.mlp_dense = nn.Dense(self.num_classes)

        self.to_time_embedding = nn.Dense(self.dim)

    # tương đương forward(self, img) trong PyTorch
    def __call__(self, img, train: bool = True):
        """
        img: (b, ns, c, w, h)
        """
        x = self.to_patch_embedding(img)          # (b, n_patches, dim)
        b, n, _ = x.shape

        cls_tokens = repeat(
            self.cls_token, "1 n d -> b n d", b=b)  # (b, 1, dim)
        # (b, n+1, dim)
        x = jnp.concatenate([cls_tokens, x], axis=1)

        x = x + self.pos_embedding[:, : (n + 1), :]
        x = self.emb_dropout_layer(x, deterministic=not train)

        x = self.transformer(x, train=train)

        if self.pool == "mean":
            x = x.mean(axis=1)
        else:  # 'cls' (mặc định)
            x = x[:, 0]

        x = self.mlp_norm(x)
        x = self.mlp_dense(x)
        return x

    # tương đương forward_set(self, img, t_emb=None, c_old=None)
    def forward_set(
        self,
        img,
        t_emb: Optional[jnp.ndarray] = None,
        c_old: Optional[jnp.ndarray] = None,
        train: bool = True,
    ):
        """
        img: (b, ns, c, w, h)
        t_emb: (b * ns, t_dim) (giống logic view trong code PyTorch)
        """
        patches = self.to_patch_embedding(img)    # (b, n, dim)
        b, n, dim = patches.shape
        ns = self.ns

        cls_tokens = repeat(self.cls_token, "1 n d -> b n d", b=b)  # (b,1,dim)

        if t_emb is None:
            t_tok = jnp.zeros((b, 1, dim), dtype=patches.dtype)
        else:
            # PyTorch: t_emb -> Linear(t_dim->dim) -> view(b, ns, -1) -> lấy phần tử đầu
            t_tok = self.to_time_embedding(t_emb)   # (b*ns, dim) giả định
            t_tok = t_tok.reshape(b, ns, -1)
            t_tok = t_tok[:, 0:1, :]                # (b,1,dim)

        # concat cls + t + patches
        x = jnp.concatenate([cls_tokens, t_tok, patches],
                            axis=1)  # (b, n+2, dim)

        x = x + self.pos_embedding[:, : (n + 2), :]
        x = self.emb_dropout_layer(x, deterministic=not train)

        x_set = self.transformer(x, train=train)

        # pooling giống PyTorch
        if self.pool == "mean":
            x_out = x_set.mean(axis=1)
        elif self.pool == "sum":
            x_out = x_set.sum(axis=1)
        elif self.pool == "cls":
            x_out = x_set[:, 0]
        else:  # 'none'
            x_out = x_set

        x_out = self.mlp_norm(x_out)
        if c_old is not None:
            x_out = x_out + c_old
        hc = self.mlp_dense(x_out)

        # Return tuple instead of dict for JAX tracing compatibility
        return hc, x_set, x_set[:, 0]
