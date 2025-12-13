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


class SPTStack(nn.Module):
    """Giống paper code: stack ns ảnh vào channel rồi patchify."""
    dim: int
    patch_size: int
    channels: int = 3
    sample_size: int = 5  # FIX B: cố định như paper (5) hoặc tuỳ bạn đặt

    def setup(self):
        patch_dim = self.patch_size * self.patch_size * self.sample_size * self.channels
        self.norm = nn.LayerNorm()
        self.proj = nn.Dense(self.dim)

    def __call__(self, x_set):
        """
        x_set: (b, ns, c, h, w)
        """
        b, ns, c, h, w = x_set.shape

        # FIX B: pad/truncate ns -> sample_size để patch_dim luôn khớp
        if ns < self.sample_size:
            pad = jnp.zeros((b, self.sample_size - ns, c, h, w), dtype=x_set.dtype)
            x_set = jnp.concatenate([x_set, pad], axis=1)
        elif ns > self.sample_size:
            x_set = x_set[:, : self.sample_size]

        # (b, ns, c, h, w) -> (b, c, ns, h, w) -> (b, c*ns, h, w)
        x = jnp.transpose(x_set, (0, 2, 1, 3, 4))
        x = x.reshape(b, c * self.sample_size, h, w)

        p = self.patch_size
        x = rearrange(
            x,
            "b c (hh p1) (ww p2) -> b (hh ww) (p1 p2 c)",
            p1=p,
            p2=p,
        )

        x = self.norm(x)
        x = self.proj(x)
        return x  # (b, n_patches, dim)


class PatchEmbedPerSampleMean(nn.Module):
    """
    FIX D (đúng nghĩa Tokens(T) + per-patch aggregation):
    patchify từng ảnh riêng -> embed -> mean theo ns để ra (b, n_patches, dim).
    """
    dim: int
    patch_size: int
    channels: int = 3
    sample_size: int = 5

    def setup(self):
        patch_dim = self.patch_size * self.patch_size * self.channels
        self.norm = nn.LayerNorm()
        self.proj = nn.Dense(self.dim)

    def __call__(self, x_set):
        """
        x_set: (b, ns, c, h, w)
        """
        b, ns, c, h, w = x_set.shape

        # pad/truncate ns -> sample_size (để consistent)
        if ns < self.sample_size:
            pad = jnp.zeros((b, self.sample_size - ns, c, h, w), dtype=x_set.dtype)
            x_set = jnp.concatenate([x_set, pad], axis=1)
        elif ns > self.sample_size:
            x_set = x_set[:, : self.sample_size]

        ns_eff = self.sample_size
        p = self.patch_size

        # (b, ns, c, h, w) -> (b*ns, c, h, w)
        x = x_set.reshape(b * ns_eff, c, h, w)

        # patchify từng ảnh: (b*ns, np, patch_dim)
        x = rearrange(
            x,
            "bn c (hh p1) (ww p2) -> bn (hh ww) (p1 p2 c)",
            p1=p,
            p2=p,
        )

        x = self.norm(x)
        x = self.proj(x)  # (b*ns, np, dim)

        # reshape lại và mean theo ns: (b, ns, np, dim) -> (b, np, dim)
        x = x.reshape(b, ns_eff, x.shape[1], self.dim)
        x = x.mean(axis=1)
        return x  # (b, n_patches, dim)


class sViT(nn.Module):
    image_size: Tuple[int, int]
    patch_size: int
    num_classes: int
    dim: int
    depth: int
    heads: int
    mlp_dim: int

    # FIX C: cho phép 'sum'
    pool: Literal['cls', 'mean', 'sum', 'none'] = 'cls'

    channels: int = 3
    dim_head: int = 64
    dropout: float = 0.0
    emb_dropout: float = 0.0

    # ns là số ảnh thực đưa vào; sample_size là “khuôn” như paper
    ns: int = 5
    sample_size: int = 5  # FIX B: mặc định 5 như paper

    t_dim: int = 256

    # FIX D: chọn tokenization mode
    tokenize_mode: Literal["stack", "per_sample_mean"] = "stack"

    def setup(self):
        image_height, image_width = pair(self.image_size)
        patch_height, patch_width = pair(self.patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0

        self.np = (image_height // patch_height) * (image_width // patch_width)
        assert self.pool in {'cls', 'mean', 'sum', 'none'}

        # Patch embedding
        if self.tokenize_mode == "stack":
            self.to_patch_embedding = SPTStack(
                dim=self.dim,
                patch_size=patch_height,
                channels=self.channels,
                sample_size=self.sample_size,   # FIX B
            )
        else:
            self.to_patch_embedding = PatchEmbedPerSampleMean(
                dim=self.dim,
                patch_size=patch_height,
                channels=self.channels,
                sample_size=self.sample_size,   # FIX B + FIX D
            )

        # +2 cho CLS + TIME token (giống paper)
        self.pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=1.0),
            (1, self.np + 2, self.dim),
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

        # giống PyTorch: mlp_head = LayerNorm + Linear
        self.mlp_norm = nn.LayerNorm()
        self.mlp_dense = nn.Dense(self.num_classes)

        self.to_time_embedding = nn.Dense(self.dim)

    def __call__(self, img, train: bool = True):
        """
        forward thường (không time token), giống paper forward().
        img: (b, ns, c, h, w)
        """
        x = self.to_patch_embedding(img)    # (b, np, dim)
        b, n, _ = x.shape

        cls_tokens = jnp.repeat(self.cls_token, b, axis=0)  # (b,1,dim)
        x = jnp.concatenate([cls_tokens, x], axis=1)        # (b, np+1, dim)

        x = x + self.pos_embedding[:, : (n + 1), :]
        x = self.emb_dropout_layer(x, deterministic=not train)
        x = self.transformer(x, train=train)

        if self.pool == "mean":
            x = x.mean(axis=1)
        elif self.pool == "sum":
            x = x.sum(axis=1)
        else:  # 'cls' hoặc 'none' -> cls cho forward thường
            x = x[:, 0]

        x = self.mlp_norm(x)
        x = self.mlp_dense(x)
        return x

    def forward_set(
        self,
        img,
        t_emb: Optional[jnp.ndarray] = None,
        c_old: Optional[jnp.ndarray] = None,
        train: bool = True,
    ):
        """
        img: (b, ns, c, h, w)
        t_emb: (b*ns, t_dim)  (expanded) — giống paper
        """
        patches = self.to_patch_embedding(img)  # (b, np, dim)
        b, n, dim = patches.shape
        
        # Get actual ns from input shape, not self.ns (for leave-one-out compatibility)
        if img.ndim == 5:
            actual_ns = img.shape[1]  # (b, ns, c, h, w)
        else:
            actual_ns = self.ns  # fallback to config value
        
        cls_tokens = jnp.repeat(self.cls_token, b, axis=0)  # (b,1,dim)

        # TIME token (giống paper)
        if t_emb is None:
            t_tok = jnp.zeros((b, 1, dim), dtype=patches.dtype)
            # DEBUG: Log if no timestep
            if not hasattr(sViT.forward_set, "_logged_no_t"):
                import sys
                print(f"\n[DEBUG sViT.forward_set] ⚠️  t_emb=None, using zero time token", file=sys.stderr)
                sViT.forward_set._logged_no_t = True
        else:
            # DEBUG: Log timestep usage (only first call)
            if not hasattr(sViT.forward_set, "_logged_with_t"):
                import sys
                print(f"\n[DEBUG sViT.forward_set] ✓ Received t_emb:", file=sys.stderr)
                print(f"  - t_emb shape: {t_emb.shape}", file=sys.stderr)
                print(f"  - img shape: {img.shape}, actual_ns: {actual_ns}", file=sys.stderr)
                print(f"  - dropout: {self.dropout}, emb_dropout: {self.emb_dropout}", file=sys.stderr)
                print(f"  - train mode: {train}", file=sys.stderr)
                sViT.forward_set._logged_with_t = True
            
            # safety check: t_emb should match actual number of images in batch
            expected_t_emb_size = b * actual_ns
            assert t_emb.shape[0] == expected_t_emb_size, \
                f"t_emb must be (b*actual_ns, t_dim), got {t_emb.shape}, expected first dim={expected_t_emb_size} (b={b}, actual_ns={actual_ns})"
            t_tok = self.to_time_embedding(t_emb)         # (b*actual_ns, dim)
            t_tok = t_tok.reshape(b, actual_ns, -1)       # (b, actual_ns, dim)
            t_tok = t_tok[:, 0:1, :]                      # (b,1,dim) lấy phần tử đầu, y như paper

        # concat: [CLS | TIME | PATCHES]
        x = jnp.concatenate([cls_tokens, t_tok, patches], axis=1)  # (b, np+2, dim)
        x = x + self.pos_embedding[:, : (n + 2), :]
        x = self.emb_dropout_layer(x, deterministic=not train)

        x_set = self.transformer(x, train=train)  # (b, np+2, dim)

        # pooling / output
        if self.pool == "mean":
            x_out = x_set.mean(axis=1)
        elif self.pool == "sum":
            x_out = x_set.sum(axis=1)
        elif self.pool == "cls":
            x_out = x_set[:, 0]
        else:
            # FIX D: nếu cần tokens để condition (cross-attn), trả patch tokens thôi (bỏ CLS & TIME)
            x_out = x_set[:, 2:, :]   # (b, np, dim)

        # FIX A: cộng c_old TRƯỚC LayerNorm (giống PyTorch)
        if c_old is not None:
            x_out = x_out + c_old

        x_out = self.mlp_norm(x_out)
        hc = self.mlp_dense(x_out)

        return hc, x_set, x_set[:, 0]
