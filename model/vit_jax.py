import jax.numpy as jnp
from flax import linen as nn
from einops import rearrange, repeat
from typing import Literal, Optional, Tuple


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


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


class TransformerBlock(nn.Module):
    dim: int
    heads: int
    dim_head: int
    mlp_dim: int
    dropout: float = 0.0
    attn_dropout: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool = True):
        # pre-norm attention
        y = nn.LayerNorm()(x)
        y = nn.SelfAttention(
            num_heads=self.heads,
            qkv_features=self.heads * self.dim_head,
            out_features=self.dim,
            dropout_rate=self.attn_dropout,
            deterministic=not train,
        )(y)
        y = nn.Dropout(self.dropout)(y, deterministic=not train)
        x = x + y

        # pre-norm feedforward
        y = nn.LayerNorm()(x)
        y = FeedForward(
            dim=self.dim,
            hidden_dim=self.mlp_dim,
            dropout=self.dropout
        )(y, train=train)
        x = x + y
        return x


class Transformer(nn.Module):
    dim: int
    depth: int
    heads: int
    dim_head: int
    mlp_dim: int
    dropout: float = 0.0
    attn_dropout: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool = True):
        for _ in range(self.depth):
            x = TransformerBlock(
                dim=self.dim,
                heads=self.heads,
                dim_head=self.dim_head,
                mlp_dim=self.mlp_dim,
                dropout=self.dropout,
                attn_dropout=self.attn_dropout,
            )(x, train=train)
        return x


class ViT(nn.Module):
    # các tham số giống bản PyTorch
    image_size: Tuple[int, int]
    patch_size: Tuple[int, int]
    num_classes: int
    dim: int
    depth: int
    heads: int
    mlp_dim: int

    pool: Literal['cls', 'mean', 'mean_patch', 'agg', 'none'] = 'cls'
    channels: int = 3
    dim_head: int = 64
    dropout: float = 0.0
    emb_dropout: float = 0.0
    ns: int = 1
    t_dim: int = 256
    hierarchical_patch_embedding: bool = False  # chưa dùng, chỉ để tương thích

    def setup(self):
        ih, iw = pair(self.image_size)
        ph, pw = pair(self.patch_size)

        assert ih % ph == 0 and iw % pw == 0, 'Image dimensions must be divisible by the patch size.'
        self.image_height, self.image_width = ih, iw
        self.patch_height, self.patch_width = ph, pw

        self.num_patches = (ih // ph) * (iw // pw)
        self.k = 2  # cls + t

        # linear cho patch + thời gian
        self.to_patch_embedding = nn.Dense(self.dim)
        self.to_time_embedding = nn.Dense(self.dim)

        # positional embedding + cls token
        self.pos_embedding = self.param(
            'pos_embedding',
            nn.initializers.normal(stddev=1.0),
            (1, self.num_patches + self.k, self.dim),
        )
        self.cls_token = self.param(
            'cls_token',
            nn.initializers.normal(stddev=1.0),
            (1, 1, self.dim),
        )

        # transformer backbone
        self.transformer = Transformer(
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            dim_head=self.dim_head,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            attn_dropout=self.dropout,
        )

        # head
        self.cls_norm = nn.LayerNorm()
        self.cls_dense = nn.Dense(self.num_classes)

    # tương đương forward(self, img)
    def __call__(self, img, train: bool = True):
        """
        img: (b, c, H, W)
        """
        ph, pw = self.patch_height, self.patch_width

        # b c (h p1) (w p2) -> b (h w) (p1 p2 c)
        x = rearrange(
            img,
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
            p1=ph, p2=pw
        )
        x = self.to_patch_embedding(x)  # (b, num_patches, dim)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)
        x = jnp.concatenate([cls_tokens, x], axis=1)  # (b, n+1, dim)

        x = x + self.pos_embedding[:, : (n + 1), :]
        x = nn.Dropout(self.emb_dropout)(x, deterministic=not train)

        x_set = self.transformer(x, train=train)

        if self.pool == 'mean':
            x = x_set.mean(axis=1)
        else:  # 'cls'
            x = x_set[:, 0]

        x = self.cls_norm(x)
        x = self.cls_dense(x)
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
        img: (b, ns, c, H, W) hoặc (b, c, H, W)
        t_emb: (b, t_dim) – time embedding trước, sẽ linear vào dim
        """
        ph, pw = self.patch_height, self.patch_width

        set_to_patch_embeddings = []

        if img.ndim > 4:
            # (b, ns, c, h, w)
            b, ns, c, h, w = img.shape
            for i in range(ns):
                inpt = img[:, i, ...]  # (b, c, h, w)
                patch_tmp = rearrange(
                    inpt,
                    'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                    p1=ph, p2=pw
                )
                patch_tmp = self.to_patch_embedding(patch_tmp)  # (b, np, dim)
                set_to_patch_embeddings.append(patch_tmp)

            patches = jnp.concatenate(
                set_to_patch_embeddings, axis=1)  # (b, np*ns, dim)

            if self.pool == 'agg':
                p = patches.shape[1]
                patches = patches.reshape(b, p // ns, ns, -1).mean(axis=2)
        else:
            ns = 1
            b, c, h, w = img.shape
            patches = rearrange(
                img,
                'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                p1=ph, p2=pw
            )
            patches = self.to_patch_embedding(patches)

        b, np, dim = patches.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)

        # xử lý time embedding
        if t_emb is None:
            t_tok = jnp.zeros((b, 1, dim), dtype=patches.dtype)
        else:
            # t_emb: (b, t_dim) -> (b, dim) -> (b, 1, dim)
            t_tok = self.to_time_embedding(t_emb)  # (b, dim)
            t_tok = t_tok[:, None, :]  # (b, 1, dim) - add sequence dimension

        # concat cls + t + patches
        x = jnp.concatenate([cls_tokens, t_tok, patches],
                            axis=1)  # (b, 2+np, dim)

        if self.pool == "agg":
            x = x + self.pos_embedding[:, : (np + self.k), :]
        else:
            # repeat pos cho set
            # (1, num_patches, dim)
            tmp_pos = self.pos_embedding[:, self.k:, :]
            # (1, num_patches*ns, dim)
            patches_pos = jnp.repeat(tmp_pos, repeats=ns, axis=1)

            cls_pos = self.pos_embedding[:, 0:1, :]
            t_pos = self.pos_embedding[:, 1:2, :]

            pos_embedding_patches = jnp.concatenate(
                [cls_pos, t_pos, patches_pos], axis=1
            )
            x = x + pos_embedding_patches[:, : (np + self.k), :]

        x = nn.Dropout(self.emb_dropout)(x, deterministic=not train)
        x_set = self.transformer(x, train=train)

        if self.pool == "agg":
            x_out = x_set
        else:
            if self.pool == 'mean':
                x_out = x_set.mean(axis=1)
            elif self.pool == 'sum':
                x_out = x_set.sum(axis=1)
            elif self.pool == 'cls':
                x_out = x_set[:, 0]
            elif self.pool == "mean_patch":
                x_p = x_set[:, self.k:, :]
                x_p = x_p.reshape(b, np // ns, ns, -1).mean(axis=2)
                x_out = self.transformer(x_p, train=train)
            elif self.pool == "sum_patch":
                x_p = x_set[:, self.k:, :]
                x_p = x_p.reshape(b, np // ns, ns, -1).sum(axis=2)
                x_out = self.transformer(x_p, train=train)
            else:  # 'none' -> trả nguyên token
                x_out = x_set

        # head giống bản PyTorch (LayerNorm + Linear)
        if x_out.ndim == 3:
            # nếu là token map (b, n, dim) thì dùng mean
            x_vec = x_out.mean(axis=1)
        else:
            x_vec = x_out

        x_vec = self.cls_norm(x_vec)
        if c_old is not None:
            x_vec = x_vec + c_old
        hc = self.cls_dense(x_vec)

        # Return tuple instead of dict for JAX tracing compatibility
        return hc, x_set, x_set[:, 0]
