"""
Variational Few-Shot DDPM in JAX/Flax using DiT backbone and Gaussian Diffusion.

Supports:
- Conditioning modes: film (vector) and lag (tokens).
- Context modes: deterministic and variational Gaussian (continuous KL).

This is a functional port of the PyTorch VFSDDPM to JAX, targeting DiT defined
in `model/set_diffusion/dit_jax.py` and diffusion in
`model/set_diffusion/gaussian_diffusion_jax.py`.
"""

import dataclasses
from typing import Any, Dict, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

from model.set_diffusion.script_util_jax import (
    create_model_and_diffusion,
)
from model.set_diffusion.dit_jax import DiT
from model.set_diffusion.gaussian_diffusion_jax import GaussianDiffusion
from model.set_diffusion.nn_jax import mean_flat
from model.vit_set_jax import sViT
from model.vit_jax import ViT


Array = jnp.ndarray
PRNGKey = jax.Array


def timestep_embedding_jax(timesteps: Array, dim: int, max_period: int = 10000) -> Array:
    """
    Sin/cos timestep embedding (guided-diffusion style).
    
    Args:
        timesteps: (b,) integer timesteps
        dim: embedding dimension
        max_period: maximum period for sinusoidal embedding
        
    Returns:
        (b, dim) timestep embeddings
    """
    half = dim // 2
    freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half)
    args = timesteps.astype(jnp.float32)[:, None] * freqs[None, :]
    emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


class TimeEmbed(nn.Module):
    """
    MLP giống UNet.time_embed: Linear -> SiLU -> Linear.
    Input dim = base_dim, output dim = out_dim.
    """
    base_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = nn.Dense(self.out_dim)(x)
        x = nn.silu(x)
        x = nn.Dense(self.out_dim)(x)
        return x


@dataclasses.dataclass
class VFSDDPMConfig:
    # data
    image_size: int = 32
    in_channels: int = 3
    sample_size: int = 5  # number of elements per set (ns)
    # encoder
    encoder_mode: str = "vit_set"  # "vit" or "vit_set"
    hdim: int = 256
    pool: str = "cls"
    dropout: float = 0.0  # dropout for encoder and denoiser (applied uniformly)
    # encoder architecture (configurable from CLI)
    encoder_depth: int = 3
    encoder_heads: int = 8
    encoder_dim_head: int = 56
    encoder_mlp_ratio: float = 1.0  # mlp_dim = int(hdim * encoder_mlp_ratio)
    encoder_tokenize_mode: str = "stack"  # for sViT: "stack" | "per_sample_mean"
    # DiT
    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    patch_size: int = 2
    context_channels: int = 256
    mode_conditioning: str = "film"  # "film" or "lag"
    class_cond: bool = False
    # diffusion
    diffusion_steps: int = 1000
    noise_schedule: str = "linear"
    learn_sigma: bool = False
    timestep_respacing: str = ""
    use_kl: bool = False
    predict_xstart: bool = False
    rescale_timesteps: bool = False
    rescale_learned_sigmas: bool = False
    # context mode
    mode_context: str = "deterministic"  # "deterministic" or "variational"


def build_encoder(cfg: VFSDDPMConfig) -> nn.Module:
    mlp_dim = int(cfg.hdim * cfg.encoder_mlp_ratio)
    if cfg.encoder_mode == "vit":
        return ViT(
            image_size=(cfg.image_size, cfg.image_size),
            patch_size=(cfg.patch_size, cfg.patch_size),
            num_classes=cfg.hdim,
            dim=cfg.hdim,
            depth=cfg.encoder_depth,
            heads=cfg.encoder_heads,
            dim_head=cfg.encoder_dim_head,
            mlp_dim=mlp_dim,
            pool=cfg.pool,
            channels=cfg.in_channels,
            ns=1,
            dropout=cfg.dropout,
            emb_dropout=cfg.dropout,
        )
    return sViT(
        image_size=(cfg.image_size, cfg.image_size),
        patch_size=cfg.patch_size,
        num_classes=cfg.hdim,
        dim=cfg.hdim,
        depth=cfg.encoder_depth,
        heads=cfg.encoder_heads,
        dim_head=cfg.encoder_dim_head,
        mlp_dim=mlp_dim,
        pool=cfg.pool,
        channels=cfg.in_channels,
        ns=cfg.sample_size,
        t_dim=cfg.hdim,  # khớp với t_emb dim
        sample_size=cfg.sample_size,
        dropout=cfg.dropout,
        emb_dropout=cfg.dropout,
        tokenize_mode=cfg.encoder_tokenize_mode,
    )


class PosteriorGaussian(nn.Module):
    """Simple diagonal Gaussian posterior network."""

    hdim: int

    @nn.compact
    def __call__(self, h: Array) -> Tuple[Array, Array]:
        h = nn.Dense(self.hdim)(h)
        h = nn.silu(h)
        h = nn.Dense(2 * self.hdim)(h)
        mu, logvar = jnp.split(h, 2, axis=-1)
        return mu, logvar


def gaussian_kl(qm: Array, qlogvar: Array, pm: Array, plogvar: Array) -> Array:
    """
    KL(N(qm, qv) || N(pm, pv)) with qlogvar/plogvar as log-variance.
    """
    qv = jnp.exp(qlogvar)
    pv = jnp.exp(plogvar)
    return 0.5 * (
        (qv / pv)
        + ((qm - pm) ** 2) / pv
        - 1.0
        + (plogvar - qlogvar)
    )


def init_models(rng: PRNGKey, cfg: VFSDDPMConfig):
    """
    Initialize encoder, DiT, diffusion, and posterior (if variational).
    Returns:
        params: dict with encoder, dit, posterior (optional)
        modules: dict with encoder, dit, diffusion, posterior (optional)
    """
    enc = build_encoder(cfg)
    dit, diffusion = create_model_and_diffusion(
        image_size=cfg.image_size,
        in_channels=cfg.in_channels,
        class_cond=cfg.class_cond,
        learn_sigma=cfg.learn_sigma,
        hidden_size=cfg.hidden_size,
        depth=cfg.depth,
        context_channels=cfg.context_channels,
        mode_conditioning=cfg.mode_conditioning,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        patch_size=cfg.patch_size,
        dropout=cfg.dropout,
        diffusion_steps=cfg.diffusion_steps,
        noise_schedule=cfg.noise_schedule,
        timestep_respacing=cfg.timestep_respacing,
        use_kl=cfg.use_kl,
        predict_xstart=cfg.predict_xstart,
        rescale_timesteps=cfg.rescale_timesteps,
        rescale_learned_sigmas=cfg.rescale_learned_sigmas,
        # compatibility placeholders
        channel_mult="",
        num_head_channels=-1,
        num_heads_upsample=-1,
        attention_resolutions="",
        use_checkpoint=False,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        class_dropout_prob=0.1,
        use_fp16=False,
    )

    rng, rng_enc, rng_dit = jax.random.split(rng, 3)

    # encoder params - MUST use forward_set to init to_time_embedding layer
    dummy_set = jnp.zeros(
        (1, cfg.sample_size, cfg.in_channels, cfg.image_size, cfg.image_size),
        dtype=jnp.float32,
    )
    dummy_t_emb = jnp.zeros((1 * cfg.sample_size, cfg.hdim), dtype=jnp.float32)
    
    if cfg.encoder_mode == "vit":
        # ViT also uses forward_set with time embedding
        enc_params = enc.init(
            rng_enc,
            dummy_set[:, 0],  # single image for ViT
            t_emb=dummy_t_emb[:1],  # (1, hdim)
            train=False,
            method=enc.forward_set
        )
    else:
        # sViT uses forward_set with full set
        enc_params = enc.init(
            rng_enc, 
            dummy_set, 
            t_emb=dummy_t_emb,  # (b*ns, hdim)
            train=False, 
            method=enc.forward_set
        )

    # DiT params
    dummy_x = jnp.zeros(
        (1, cfg.in_channels, cfg.image_size, cfg.image_size), dtype=jnp.float32
    )
    dummy_t = jnp.zeros((1,), dtype=jnp.int32)
    
    # CRITICAL: For lag mode, must provide dummy context to init cross-attention layers
    if cfg.mode_conditioning == "lag":
        # Create dummy context tokens: (b, num_patches, context_channels)
        num_patches = (cfg.image_size // cfg.patch_size) ** 2
        dummy_c = jnp.zeros((1, num_patches, cfg.context_channels), dtype=jnp.float32)
    else:
        dummy_c = None
    
    dit_params = dit.init(rng_dit, dummy_x, dummy_t,
                          c=dummy_c, y=None, train=False)

    params = {"encoder": enc_params, "dit": dit_params}
    modules = {"encoder": enc, "dit": dit, "diffusion": diffusion}

    # --- time_embed for encoder (mimic PyTorch generative_model.time_embed) ---
    base_dim = cfg.hdim // 4  # giống model_channels=64 nếu hdim=256
    time_embed = TimeEmbed(base_dim=base_dim, out_dim=cfg.hdim)
    rng, rng_te = jax.random.split(rng)
    dummy_te_in = jnp.zeros((1, base_dim), dtype=jnp.float32)
    time_embed_params = time_embed.init(rng_te, dummy_te_in)
    params["time_embed"] = time_embed_params
    modules["time_embed"] = time_embed

    if cfg.mode_context == "variational":
        posterior = PosteriorGaussian(cfg.hdim)
        rng_post = jax.random.split(rng, 1)[0]
        dummy_h = jnp.zeros((1, cfg.hdim), dtype=jnp.float32)
        post_params = posterior.init(rng_post, dummy_h)
        params["posterior"] = post_params
        modules["posterior"] = posterior

    return params, modules


def encode_set(
    params_enc: Any,
    encoder: nn.Module,
    x_set: Array,
    cfg: VFSDDPMConfig,
    train: bool,
    t_emb: Optional[Array] = None,
    return_tokens: bool = False,
) -> Union[Array, Tuple[Array, Array]]:
    """
    Encode a set (or single image) into a set-level representation hc.
    
    Args:
        t_emb: Optional timestep embedding (b*ns, t_dim)
               If provided, encoder becomes time-aware
        return_tokens: If True and mode_conditioning=="lag", also return patch tokens
                      Returns (hc, tokens) tuple instead of just hc
    Returns:
        hc: (b, hdim) pooled representation
        tokens: (b, num_patches, hdim) patch tokens (only if return_tokens=True and lag mode)
    """
        # DEBUG: Log encode_set call (only first call)
    if not hasattr(encode_set, "_logged"):
        import sys
        print(f"\n[DEBUG encode_set] Called with:", file=sys.stderr)
        print(f"  - x_set shape: {x_set.shape}", file=sys.stderr)
        print(f"  - t_emb: {'None' if t_emb is None else f'shape={t_emb.shape}'}", file=sys.stderr)
        print(f"  - train: {train}", file=sys.stderr)
        print(f"  - encoder_mode: {cfg.encoder_mode}", file=sys.stderr)
        print(f"  - mode_conditioning: {cfg.mode_conditioning}", file=sys.stderr)
        print(f"  - return_tokens: {return_tokens}", file=sys.stderr)
        encode_set._logged = True
    
    if cfg.encoder_mode == "vit":
        # ViT also has forward_set method similar to sViT
        b, ns = x_set.shape[:2]
        # Expand t_emb for ViT: (b*ns, hdim) -> (b, hdim) by taking first per batch
        if t_emb is not None:
            # t_emb: (b*ns, hdim), reshape to (b, ns, hdim) and take first
            t_emb_reshaped = t_emb.reshape(b, ns, -1)
            t_emb_vit = t_emb_reshaped[:, 0, :]  # (b, hdim) - take first image's t_emb
        else:
            t_emb_vit = None
        
        # Use forward_set to get tokens if needed
        if return_tokens and cfg.mode_conditioning == "lag":
            hc, x_set_tokens, cls = encoder.apply(
                params_enc, x_set, t_emb=t_emb_vit, c_old=None, 
                train=train, method=encoder.forward_set
            )
            if hc.ndim == 3:
                hc = hc.mean(axis=1)
            # Extract patch tokens (skip CLS and TIME tokens)
            # ViT.forward_set always has TIME token (even if t_emb=None, it creates zero token)
            # So offset is always 2: CLS + TIME
            tokens = x_set_tokens[:, 2:, :]  # (b, np, dim) - patch tokens only (skip CLS=0, TIME=1)
            
            # ASSERT 1-4: Validate token shapes for ViT
            b_actual = tokens.shape[0]
            num_patches_actual = tokens.shape[1]
            hdim_actual = tokens.shape[2]
            num_patches_expected = (cfg.image_size // cfg.patch_size) ** 2
            
            assert tokens.ndim == 3, f"ViT tokens must be 3D, got {tokens.ndim}D"
            assert hdim_actual == cfg.hdim, \
                f"ViT tokens dim mismatch: got {hdim_actual}, expected {cfg.hdim}"
            assert num_patches_actual == num_patches_expected, \
                f"ViT num_patches mismatch: got {num_patches_actual}, expected {num_patches_expected}"
            
            return hc, tokens
        else:
            # Use forward_set but don't return tokens
            hc, _, _ = encoder.apply(
                params_enc, x_set, t_emb=t_emb_vit, c_old=None,
                train=train, method=encoder.forward_set
            )
            if hc.ndim == 3:
                hc = hc.mean(axis=1)
            return hc
    else:
        # encoder returns (hc, patches, cls) for forward_set
        # Must explicitly call forward_set method (apply() defaults to __call__)
        hc, x_set_tokens, cls = encoder.apply(
            params_enc, x_set, t_emb=t_emb, train=train, method=encoder.forward_set)
        if hc.ndim == 3:
            hc = hc.mean(axis=1)
        
        # Extract patch tokens for lag mode (skip CLS and TIME tokens)
        if return_tokens and cfg.mode_conditioning == "lag":
            # sViT.forward_set always has TIME token (even if t_emb=None, it creates zero token)
            # So offset is always 2: CLS=0, TIME=1, then patches start at idx=2
            # x_set_tokens: (b, np+2, dim) where first 2 are CLS and TIME
            # Extract only patch tokens: (b, np, dim)
            tokens = x_set_tokens[:, 2:, :]  # Skip CLS (idx 0) and TIME (idx 1)
            
            # ASSERT 1-4: Validate token shapes
            b_actual = tokens.shape[0]
            num_patches_actual = tokens.shape[1]
            hdim_actual = tokens.shape[2]
            num_patches_expected = (cfg.image_size // cfg.patch_size) ** 2
            
            assert tokens.ndim == 3, f"tokens must be 3D, got {tokens.ndim}D"
            assert hdim_actual == cfg.hdim, \
                f"tokens dim mismatch: got {hdim_actual}, expected {cfg.hdim}"
            assert num_patches_actual == num_patches_expected, \
                f"num_patches mismatch: got {num_patches_actual}, expected {num_patches_expected} (image_size={cfg.image_size}, patch_size={cfg.patch_size})"
            
            # Log for debugging
            if not hasattr(encode_set, "_logged_tokens"):
                import sys
                print(f"\n[DEBUG encode_set] Token extraction (lag mode):", file=sys.stderr)
                print(f"  - x_set_tokens shape: {x_set_tokens.shape}", file=sys.stderr)
                print(f"  - tokens shape: {tokens.shape}", file=sys.stderr)
                print(f"  - Expected num_patches: {num_patches_expected}", file=sys.stderr)
                encode_set._logged_tokens = True
            
            return hc, tokens
        
        return hc


def sample_context(
    rng: PRNGKey,
    hc: Array,
    cfg: VFSDDPMConfig,
    posterior: Optional[PosteriorGaussian],
    params_post: Optional[Any],
) -> Tuple[Array, Optional[Array]]:
    """
    Deterministic or variational Gaussian sample of context vector(s).
    Returns (c, klc_or_None)
    """
    if cfg.mode_context != "variational":
        return hc, None

    assert posterior is not None and params_post is not None
    mu, logvar = posterior.apply(params_post, hc)
    rng, eps_key = jax.random.split(rng)
    eps = jax.random.normal(eps_key, mu.shape)
    c = mu + jnp.exp(0.5 * logvar) * eps
    pm = jnp.zeros_like(mu)
    plogvar = jnp.zeros_like(logvar)
    kl = gaussian_kl(mu, logvar, pm, plogvar)
    klc = mean_flat(kl) / jnp.log(2.0)
    return c, klc


def leave_one_out_c(
    rng: PRNGKey,
    params: Dict[str, Any],
    modules: Dict[str, Any],
    batch_set: Array,
    cfg: VFSDDPMConfig,
    train: bool,
    t: Array,  # (b,) integer timesteps
) -> Tuple[Array, Optional[Array]]:
    """
    Build conditioning c for each element via leave-one-out over the set.
    Returns (c, klc_optional).
    Shapes:
        film -> c: (bs * ns, hdim) - pooled vector
        lag  -> c: (bs * ns, num_patches, hdim) - patch tokens for cross-attention
    """
    b, ns = batch_set.shape[:2]
    enc = modules["encoder"]
    posterior = modules.get("posterior")
    params_post = params.get("posterior")

    # --- make t_emb like PyTorch: time_embed(timestep_embedding(t)) ---
    base_dim = cfg.hdim // 4
    t_base = timestep_embedding_jax(t, base_dim)  # (b, base_dim)
    t_emb_set = modules["time_embed"].apply(params["time_embed"], t_base)  # (b, hdim)
    
        # DEBUG: Log timestep embedding info (only first call)
    if not hasattr(leave_one_out_c, "_logged"):
        import sys
        print(f"\n[DEBUG leave_one_out_c] Timestep embedding for encoder:", file=sys.stderr)
        print(f"  - t shape: {t.shape}, values: {t[:min(3, len(t))]}", file=sys.stderr)
        print(f"  - t_emb_set shape: {t_emb_set.shape}", file=sys.stderr)
        print(f"  - Encoder mode: {cfg.encoder_mode}", file=sys.stderr)
        print(f"  - Mode conditioning: {cfg.mode_conditioning}", file=sys.stderr)
        print(f"  - Dropout: {cfg.dropout}", file=sys.stderr)
        leave_one_out_c._logged = True

    kl_list = []
    c_list = []
    token_list = []  # For lag mode: collect patch tokens
    rngs = jax.random.split(rng, ns)
    
    # Check if we need tokens for lag mode
    need_tokens = cfg.mode_conditioning == "lag"
    
    for i in range(ns):
        idx = [k for k in range(ns) if k != i]
        x_subset = batch_set[:, idx]  # (b, ns-1, C, H, W)
        
        # CRITICAL FIX: For sViT with SPT stacking, pad subset back to sample_size
        # SPT expects fixed sample_size for patch_dim calculation
        subset_ns_original = ns - 1
        if cfg.encoder_mode == "vit_set" and x_subset.shape[1] < cfg.sample_size:
            # Pad with zeros to maintain sample_size
            pad_size = cfg.sample_size - x_subset.shape[1]
            pad_images = jnp.zeros((b, pad_size, *x_subset.shape[2:]), dtype=x_subset.dtype)
            x_subset = jnp.concatenate([x_subset, pad_images], axis=1)  # (b, sample_size, C, H, W)
            subset_ns = cfg.sample_size
        else:
            subset_ns = subset_ns_original
        
        # Expand t_emb for this subset: (b, hdim) -> (b * subset_ns, hdim)
        # If we padded, repeat the last t_emb for padded images
        t_emb_subset = jnp.repeat(t_emb_set[:, None, :], subset_ns_original, axis=1).reshape(
            b * subset_ns_original, cfg.hdim
        )
        if subset_ns > subset_ns_original:
            # Pad t_emb to match padded x_subset
            pad_t_emb = jnp.repeat(t_emb_set[:, None, :], subset_ns - subset_ns_original, axis=1).reshape(
                b * (subset_ns - subset_ns_original), cfg.hdim
            )
            t_emb_subset = jnp.concatenate([t_emb_subset, pad_t_emb], axis=0)  # (b * subset_ns, hdim)
        
        if need_tokens:
            # Get both hc and tokens for lag mode
            hc, tokens = encode_set(
                params["encoder"], enc, x_subset, cfg, train=train, 
                t_emb=t_emb_subset, return_tokens=True
            )
            # tokens: (b, num_patches, hdim)
            token_list.append(tokens)  # Will reshape later
        else:
            # Only get hc for film mode
            hc = encode_set(
                params["encoder"], enc, x_subset, cfg, train=train, 
                t_emb=t_emb_subset, return_tokens=False
            )
        
        # Apply posterior if variational (works on pooled hc ONLY, not tokens)
        # CRITICAL: c_vec is only for KL loss, NOT used for conditioning in lag mode
        c_vec, klc = sample_context(rngs[i], hc, cfg, posterior, params_post)
        c_list.append(c_vec[:, None, ...])  # keep set slot: (b, 1, hdim) - only for logging/debug
        if klc is not None:
            kl_list.append(klc)

    if need_tokens:
        # For lag mode: use patch tokens for cross-attention
        # token_list: list of (b, num_patches, hdim), length=ns
        # Stack: (b, ns, num_patches, hdim)
        token_set = jnp.stack(token_list, axis=1)  # (b, ns, num_patches, hdim)
        num_patches = token_set.shape[2]
        
        # ASSERT 5: Validate final conditioning shape
        assert token_set.shape[0] == b, f"token_set batch mismatch: got {token_set.shape[0]}, expected {b}"
        assert token_set.shape[1] == ns, f"token_set ns mismatch: got {token_set.shape[1]}, expected {ns}"
        assert token_set.shape[3] == cfg.hdim, f"token_set hdim mismatch: got {token_set.shape[3]}, expected {cfg.hdim}"
        
        # Reshape to (b*ns, num_patches, hdim) for DiT cross-attention
        c = token_set.reshape(b * ns, num_patches, cfg.hdim)
        
        # ASSERT 6: Final conditioning shape
        assert c.shape == (b * ns, num_patches, cfg.hdim), \
            f"Final c shape mismatch: got {c.shape}, expected ({b*ns}, {num_patches}, {cfg.hdim})"
        
        # DEBUG: Log token shapes
        if not hasattr(leave_one_out_c, "_logged_tokens"):
            import sys
            print(f"\n[DEBUG leave_one_out_c] Using patch tokens for lag mode:", file=sys.stderr)
            print(f"  - token_set shape: {token_set.shape}", file=sys.stderr)
            print(f"  - c (final) shape: {c.shape}", file=sys.stderr)
            print(f"  - num_patches: {num_patches}", file=sys.stderr)
            print(f"  - Expected: (b*ns={b*ns}, num_patches={num_patches}, hdim={cfg.hdim})", file=sys.stderr)
            leave_one_out_c._logged_tokens = True
    else:
        # For film mode: use pooled vectors
        c_set = jnp.concatenate(c_list, axis=1)  # (b, ns, hdim)
        c = c_set.reshape(b * ns, c_set.shape[-1])  # (b*ns, hdim)

    klc_total = None
    if kl_list:
        klc_total = jnp.stack(kl_list, axis=0).mean()

    return c, klc_total


def vfsddpm_loss(
    rng: PRNGKey,
    params: Dict[str, Any],
    modules: Dict[str, Any],
    batch_set: Array,  # (bs, ns, C, H, W), values in [-1, 1]
    cfg: VFSDDPMConfig,
    train: bool = True,
) -> Dict[str, Array]:
    """
    Compute VFSDDPM loss (diffusion + optional KL).
    """
    diffusion: GaussianDiffusion = modules["diffusion"]
    dit: DiT = modules["dit"]

    b, ns = batch_set.shape[:2]
    rng, t_key, noise_key = jax.random.split(rng, 3)
    t = jax.random.randint(t_key, (b,), 0, diffusion.num_timesteps)
    t_rep = jnp.repeat(t, ns, axis=0)

    # conditioning
    rng_c, rng_loss = jax.random.split(noise_key)
    c, klc = leave_one_out_c(rng_c, params, modules,
                             batch_set, cfg, train=train, t=t)

    # flatten images
    x = batch_set.reshape(b * ns, *batch_set.shape[2:])

    def model_fn(x_in, t_in, _c_unused, **kwargs):
        # ASSERT 6 (continued): Verify x and c have matching batch dimension
        if cfg.mode_conditioning == "lag":
            assert x_in.shape[0] == c.shape[0], \
                f"Batch mismatch: x_in.shape[0]={x_in.shape[0]}, c.shape[0]={c.shape[0]}"
            assert c.ndim == 3, f"Lag mode c must be 3D (b*ns, num_patches, hdim), got {c.ndim}D with shape {c.shape}"
        return dit.apply(params["dit"], x_in, t_in, c=c, train=train, **kwargs)

    losses = diffusion.training_losses(
        rng_loss, model_fn, x, t_rep, c=None, model_kwargs={}
    )
    # aggregate
    total = mean_flat(losses["loss"]).mean()
    if klc is not None:
        total = total + klc
        losses["klc"] = klc
    losses["loss"] = total
    # Add context for debugging
    losses["context"] = c
    return losses


def example_train_step_usage():
    """
    Pseudocode for integrating with train_util_jax.py:

    cfg = VFSDDPMConfig()
    rng = jax.random.PRNGKey(0)
    params, modules = init_models(rng, cfg)

    # You can wrap train_step by passing a loss_fn that uses vfsddpm_loss.
    # Example (functional, single device):
    #
    # from model.set_diffusion.train_util_jax import train_step, TrainState, create_train_state
    #
    # state = create_train_state(rng, modules['dit'], modules['diffusion'], learning_rate=1e-4)
    #
    # def loss_adapter(rng, params, batch_set):
    #     return vfsddpm_loss(rng, params, modules, batch_set, cfg, train=True)["loss"]
    #
    # grad_fn = jax.value_and_grad(lambda p, rng, batch: loss_adapter(rng, {"dit": p, "encoder": params["encoder"]}, batch))
    # ...
    """
    return None
