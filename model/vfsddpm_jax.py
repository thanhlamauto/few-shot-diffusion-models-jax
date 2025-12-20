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
import jax.lax as lax
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
    # VAE (latent space)
    use_vae: bool = False  # Enable VAE for latent space diffusion
    latent_channels: int = 4  # Latent space channels (when use_vae=True)
    latent_size: int = 0  # Latent space size (computed from image_size / downscale_factor, 0 = auto)
    original_image_size: int = 0  # Original image size before VAE encoding (set when use_vae=True)
    # Control whether encoder (ViT / sViT) operates on latents or original images when use_vae=True.
    # Backwards-compatible default: True → encoder also uses latents (old behavior).
    encoder_uses_vae: bool = True
    # Separate patch sizes for encoder and DiT (0 = fall back to patch_size)
    encoder_patch_size: int = 0
    dit_patch_size: int = 0
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
    # Encoder output head configuration
    encoder_use_mlp_head: bool = False  # Use MLP head (LayerNorm + MLP) instead of single Dense layer
    encoder_mlp_head_hidden_dim: int = 512  # Hidden dimension for MLP head (if enabled)
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
    # Input-dependent vs input-independent context (FSDM paper)
    # True: input-dependent (context includes the sample being generated) - better OOD performance
    # False: input-independent/LOO (context excludes the sample being generated) - better in-distribution
    input_dependent: bool = False  # Default to LOO (input-independent) for backward compatibility
    # Memory optimization for lag mode
    context_pool_size: int = 0  # If > 0, pool context tokens to this size (reduces Nk, saves memory)
    cross_attn_layers: str = "all"  # "all" or comma-separated layer indices (e.g., "2,3,4,5") to enable cross-attn only at specific layers
    # Debug / logging
    debug_metrics: bool = False  # Gate heavy debug reductions in vfsddpm_loss
    # Context LayerNorm control for lag mode cross-attention
    use_context_layernorm: bool = True


def build_encoder(cfg: VFSDDPMConfig) -> nn.Module:
    mlp_dim = int(cfg.hdim * cfg.encoder_mlp_ratio)
    # Effective patch size for encoder: override if encoder_patch_size > 0
    patch_size_enc = cfg.encoder_patch_size if cfg.encoder_patch_size > 0 else cfg.patch_size
    if cfg.encoder_mode == "vit":
        return ViT(
            image_size=(cfg.image_size, cfg.image_size),
            patch_size=(patch_size_enc, patch_size_enc),
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
            use_mlp_head=cfg.encoder_use_mlp_head,
            mlp_head_hidden_dim=cfg.encoder_mlp_head_hidden_dim,
        )
    return sViT(
        image_size=(cfg.image_size, cfg.image_size),
            patch_size=patch_size_enc,
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
        use_mlp_head=cfg.encoder_use_mlp_head,
        mlp_head_hidden_dim=cfg.encoder_mlp_head_hidden_dim,
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
        params: dict with encoder, dit, posterior (optional), vae (optional)
        modules: dict with encoder, dit, diffusion, posterior (optional), vae (optional)
        cfg: updated config (with latent_size and original_image_size set if VAE enabled)
    """
    # Initialize VAE if enabled
    vae = None
    vae_params = None
    # effective_* are the spatial size and channels seen by the DiT (diffusion backbone).
    # By default they match the data config. When use_vae=True, DiT will operate in latent space.
    effective_image_size = cfg.image_size
    effective_in_channels = cfg.in_channels
    
    if cfg.use_vae:
        from model.vae_jax import StableVAE
        vae = StableVAE.create()
        vae_params = vae.params
        
        # Store original image size before converting to latent space
        original_image_size = cfg.image_size
        
        # Calculate latent size if not set
        if cfg.latent_size <= 0:
            latent_size = cfg.image_size // vae.downscale_factor
        else:
            latent_size = cfg.latent_size
        
        # DiT / diffusion always operate in latent space when use_vae=True
        effective_image_size = latent_size
        effective_in_channels = cfg.latent_channels
        
        # Update cfg with latent metadata and, depending on encoder_uses_vae, either:
        # - keep encoder on latents (old behavior), or
        # - keep encoder on original images (new behavior).
        if cfg.encoder_uses_vae:
            # Old behavior: encoder also processes latents
            cfg = dataclasses.replace(
                cfg,
                latent_size=latent_size,
                original_image_size=original_image_size,
                image_size=effective_image_size,
                in_channels=effective_in_channels,
            )
        else:
            # New behavior: encoder stays in image space (32x32x3 for CIFAR),
            # DiT operates in latent space only.
            cfg = dataclasses.replace(
                cfg,
                latent_size=latent_size,
                original_image_size=original_image_size,
                # Leave image_size and in_channels as original
            )
    
    enc = build_encoder(cfg)
    # Effective patch size for DiT: override if dit_patch_size > 0
    dit_patch = cfg.dit_patch_size if cfg.dit_patch_size > 0 else cfg.patch_size

    dit, diffusion = create_model_and_diffusion(
        image_size=effective_image_size,
        in_channels=effective_in_channels,
        class_cond=cfg.class_cond,
        learn_sigma=cfg.learn_sigma,
        hidden_size=cfg.hidden_size,
        depth=cfg.depth,
        context_channels=cfg.context_channels,
        mode_conditioning=cfg.mode_conditioning,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        patch_size=dit_patch,
        dropout=cfg.dropout,
        cross_attn_layers=getattr(cfg, "cross_attn_layers", "all"),
        use_context_layernorm=getattr(cfg, "use_context_layernorm", True),
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
    # Encoder always uses cfg.image_size / cfg.in_channels (image space) when
    # encoder_uses_vae=False, otherwise it uses latent space (old behavior).
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

    # DiT params - uses effective (latent) sizes
    dummy_x = jnp.zeros(
        (1, effective_in_channels, effective_image_size, effective_image_size), dtype=jnp.float32
    )
    dummy_t = jnp.zeros((1,), dtype=jnp.int32)
    
    # CRITICAL: For lag mode, must provide dummy context to init cross-attention layers
    if cfg.mode_conditioning == "lag":
        # Create dummy context tokens: (b, num_patches, context_channels)
        num_patches = (effective_image_size // cfg.patch_size) ** 2
        dummy_c = jnp.zeros((1, num_patches, cfg.context_channels), dtype=jnp.float32)
    else:
        dummy_c = None
    
    dit_params = dit.init(rng_dit, dummy_x, dummy_t,
                          c=dummy_c, y=None, train=False)

    params = {"encoder": enc_params, "dit": dit_params}
    modules = {"encoder": enc, "dit": dit, "diffusion": diffusion}
    
    # Add VAE if enabled
    if cfg.use_vae:
        params["vae"] = vae_params
        modules["vae"] = vae

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

    return params, modules, cfg


def load_pretrained_encoder_weights(
    encoder_params: Any,
    pretrained_path: str,
    strict: bool = False,
) -> Any:
    """
    Load pretrained encoder weights from npz file and merge into encoder_params.
    
    Args:
        encoder_params: Current encoder parameters (from init)
        pretrained_path: Path to pretrained weights npz file
        strict: If True, raise error if keys don't match. If False, only load matching keys.
    
    Returns:
        Updated encoder_params with pretrained weights loaded
    """
    import numpy as np
    import os
    import sys
    
    # Try to import from convert script (might not exist)
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from convert_vit_pytorch_to_jax import unflatten_dict
    except ImportError:
        # Fallback: define unflatten_dict locally
        def unflatten_dict(flat_dict, sep='/'):
            """Reconstruct nested dict from flattened dict"""
            result = {}
            for key, value in flat_dict.items():
                parts = key.split(sep)
                d = result
                for part in parts[:-1]:
                    if part not in d:
                        d[part] = {}
                    d = d[part]
                d[parts[-1]] = value
            return result
    
    print(f"Loading pretrained encoder weights from: {pretrained_path}")
    
    # Load pretrained weights
    pretrained_data = np.load(pretrained_path, allow_pickle=True)
    flat_pretrained = {k: v for k, v in pretrained_data.items()}
    
    # Unflatten to nested structure
    pretrained_nested = unflatten_dict(flat_pretrained)
    
    # Extract encoder params from pretrained (might be under 'params' key)
    if 'params' in pretrained_nested:
        pretrained_encoder = pretrained_nested['params']
    else:
        pretrained_encoder = pretrained_nested
    
    # Function to recursively update params
    def update_params(current, pretrained, path=""):
        if isinstance(current, dict) and isinstance(pretrained, dict):
            updated = {}
            for key in current.keys():
                current_path = f"{path}.{key}" if path else key
                if key in pretrained:
                    updated[key] = update_params(current[key], pretrained[key], current_path)
                else:
                    if strict:
                        raise KeyError(f"Key '{current_path}' not found in pretrained weights")
                    else:
                        print(f"Warning: Key '{current_path}' not found in pretrained weights, keeping initialized value")
                        updated[key] = current[key]
            return updated
        else:
            # Leaf node: replace with pretrained value
            if strict and current.shape != pretrained.shape:
                raise ValueError(f"Shape mismatch at {path}: {current.shape} vs {pretrained.shape}")
            elif current.shape != pretrained.shape:
                print(f"Warning: Shape mismatch at {path}: {current.shape} vs {pretrained.shape}, skipping")
                return current
            return pretrained
    
    updated_params = update_params(encoder_params, pretrained_encoder)
    print("✅ Pretrained encoder weights loaded successfully")
    
    return updated_params


def encode_set(
    params_enc: Any,
    encoder: nn.Module,
    x_set: Array,
    cfg: VFSDDPMConfig,
    train: bool,
    t_emb: Optional[Array] = None,
    return_tokens: bool = False,
    rng: Optional[PRNGKey] = None,
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
            # Pass rngs for dropout if train=True and dropout > 0
            apply_kwargs = {
                "train": train,
                "method": encoder.forward_set,
            }
            if train and cfg.dropout > 0 and rng is not None:
                apply_kwargs["rngs"] = {"dropout": rng}
            hc, x_set_tokens, cls = encoder.apply(
                params_enc, x_set, t_emb=t_emb_vit, c_old=None, **apply_kwargs
            )
            if hc.ndim == 3:
                hc = hc.mean(axis=1)
            # Extract patch tokens (skip CLS and TIME tokens)
            # ViT.forward_set always has TIME token (even if t_emb=None, it creates zero token)
            # So offset is always 2: CLS + TIME
            tokens_all = x_set_tokens[:, 2:, :]  # (b, np*ns, dim) - all patch tokens concatenated
            
            # For ViT with set of images, patches are concatenated: (b, np*ns, dim)
            # Need to reshape to (b*ns, np, dim) for lag mode
            patch_size_enc = cfg.encoder_patch_size if cfg.encoder_patch_size > 0 else cfg.patch_size
            num_patches_per_image = (cfg.image_size // patch_size_enc) ** 2
            
            # Reshape: (b, np*ns, dim) -> (b, ns, np, dim) -> (b*ns, np, dim)
            b_actual = tokens_all.shape[0]
            total_patches = tokens_all.shape[1]
            assert total_patches % num_patches_per_image == 0, \
                f"Total patches ({total_patches}) must be divisible by patches per image ({num_patches_per_image})"
            ns_actual = total_patches // num_patches_per_image
            
            tokens = tokens_all.reshape(b_actual, ns_actual, num_patches_per_image, -1)  # (b, ns, np, dim)
            tokens = tokens.reshape(b_actual * ns_actual, num_patches_per_image, -1)  # (b*ns, np, dim)
            
            # ASSERT 1-4: Validate token shapes for ViT
            hdim_actual = tokens.shape[2]
            
            assert tokens.ndim == 3, f"ViT tokens must be 3D, got {tokens.ndim}D"
            assert hdim_actual == cfg.hdim, \
                f"ViT tokens dim mismatch: got {hdim_actual}, expected {cfg.hdim}"
            assert tokens.shape[1] == num_patches_per_image, \
                f"ViT num_patches mismatch: got {tokens.shape[1]}, expected {num_patches_per_image}"
            
            return hc, tokens
        else:
            # Use forward_set but don't return tokens
            apply_kwargs = {
                "train": train,
                "method": encoder.forward_set,
            }
            if train and cfg.dropout > 0 and rng is not None:
                apply_kwargs["rngs"] = {"dropout": rng}
            hc, _, _ = encoder.apply(
                params_enc, x_set, t_emb=t_emb_vit, c_old=None, **apply_kwargs
            )
            if hc.ndim == 3:
                hc = hc.mean(axis=1)
            return hc
    else:
        # encoder returns (hc, patches, cls) for forward_set
        # Must explicitly call forward_set method (apply() defaults to __call__)
        apply_kwargs = {
            "train": train,
            "method": encoder.forward_set
        }
        if train and cfg.dropout > 0 and rng is not None:
            apply_kwargs["rngs"] = {"dropout": rng}
        hc, x_set_tokens, cls = encoder.apply(
            params_enc, x_set, t_emb=t_emb, **apply_kwargs)
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
            patch_size_enc = cfg.encoder_patch_size if cfg.encoder_patch_size > 0 else cfg.patch_size
            num_patches_expected = (cfg.image_size // patch_size_enc) ** 2
            
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
    Build conditioning c for each element.
    
    Two modes (FSDM paper):
    - input_dependent=False (LOO): context excludes the sample being generated
    - input_dependent=True: context includes the sample being generated (better OOD)
    
    Returns (c, klc_optional).
    Shapes:
        film -> c: (bs * ns, hdim) - pooled vector
        lag  -> c: (bs * ns, num_patches, hdim) - patch tokens for cross-attention
    """
    b, ns = batch_set.shape[:2]
    enc = modules["encoder"]
    posterior = modules.get("posterior")
    params_post = params.get("posterior")

    # Encode images to latents for encoder if VAE is enabled, encoder_uses_vae=True,
    # and batch_set currently contains images (3 channels).
    # Note: If called from vfsddpm_loss with encoder_uses_vae=True, batch_set may
    # already be latents (4 channels) and this block will be skipped.
    if cfg.use_vae and cfg.encoder_uses_vae and batch_set.shape[2] == 3:
        vae = modules["vae"]
        vae_params = params["vae"]
        # Use original_image_size from cfg (set during init_models)
        # Fallback to calculating from latent_size if not set (for backward compatibility)
        if cfg.original_image_size > 0:
            original_image_size = cfg.original_image_size
        else:
            original_image_size = cfg.latent_size * vae.downscale_factor
        
        # Reshape to HWC format: (b, ns, 3, H, W) -> (b*ns, H, W, 3)
        bs, ns, C, H, W = batch_set.shape
        # Validate image dimensions (allow some tolerance for rounding)
        assert abs(H - original_image_size) <= 1 and abs(W - original_image_size) <= 1, \
            f"Expected image size {original_image_size}x{original_image_size}, got {H}x{W}"
        assert C == 3, f"Expected 3 image channels for VAE encoding, got {C}"
        
        # Log encoding (only first time)
        if not hasattr(leave_one_out_c, "_logged_vae_encode"):
            import sys
            print(f"\n[VAE ENCODE] leave_one_out_c: Encoding images → latents", file=sys.stderr)
            print(f"  Input shape (images): {batch_set.shape} (bs={bs}, ns={ns}, C={C}, H={H}, W={W})", file=sys.stderr)
            leave_one_out_c._logged_vae_encode = True
        
        batch_hwc = batch_set.transpose(0, 1, 3, 4, 2).reshape(bs * ns, H, W, C)
        
        # Encode to latents: (b*ns, H, W, 3) -> (b*ns, latent_H, latent_W, 4)
        rng, vae_rng = jax.random.split(rng)
        latents_hwc = vae.encode(vae_rng, batch_hwc, scale=True)
        
        # Reshape back to CHW format: (b*ns, latent_H, latent_W, 4) -> (b, ns, 4, latent_H, latent_W)
        latent_H, latent_W = latents_hwc.shape[1], latents_hwc.shape[2]
        batch_set = latents_hwc.transpose(0, 3, 1, 2).reshape(bs, ns, 4, latent_H, latent_W)
        
        # Log after encoding (only first time)
        if not hasattr(leave_one_out_c, "_logged_vae_encode_after"):
            import sys
            print(f"  Output shape (latents): {batch_set.shape} (bs={bs}, ns={ns}, C=4, H={latent_H}, W={latent_W})", file=sys.stderr)
            print(f"  ✅ Successfully encoded to latent space\n", file=sys.stderr)
            leave_one_out_c._logged_vae_encode_after = True

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

    # CRITICAL: Ensure ns matches cfg.sample_size (should be guaranteed by fix_set_size in vfsddpm_loss)
    assert ns == cfg.sample_size, f"ns={ns} != cfg.sample_size={cfg.sample_size}. This will cause JIT to compile multiple versions!"
    
    # Check if we need tokens for lag mode
    need_tokens = cfg.mode_conditioning == "lag"
    
    # Pre-allocate arrays for both modes (fixed shapes for all iterations)
    if need_tokens:
        patch_size_enc = cfg.encoder_patch_size if cfg.encoder_patch_size > 0 else cfg.patch_size
        num_patches = (cfg.image_size // patch_size_enc) ** 2
        init_token_set = jnp.zeros((b, ns, num_patches, cfg.hdim), dtype=jnp.float32)
        init_c_list = None
    else:
        init_token_set = None
        init_c_list = jnp.zeros((b, ns, cfg.hdim), dtype=jnp.float32)
    
    # Pre-allocate KL list (will be masked later if not variational)
    init_kl_list = jnp.zeros((ns,), dtype=jnp.float32)

    # Dummy-slot refactor:
    # Pad batch_set once with a dummy sample (index ns), and build indices that
    # either include all real samples (input_dependent=True) or leave-one-out
    # by redirecting one position to the dummy index.
    dummy = jnp.zeros_like(batch_set[:, :1])  # (b, 1, C, H, W)
    batch_set_pad = jnp.concatenate([batch_set, dummy], axis=1)  # (b, ns+1, C, H, W)

    # Build full timestep embedding tensor once, then pad with dummy slot
    t_emb_full = jnp.repeat(t_emb_set[:, None, :], ns, axis=1)  # (b, ns, hdim)
    t_dummy = jnp.zeros_like(t_emb_full[:, :1])  # (b, 1, hdim)
    t_emb_pad = jnp.concatenate([t_emb_full, t_dummy], axis=1)  # (b, ns+1, hdim)

    # Static positions [0, 1, ..., ns-1]
    pos = jnp.arange(ns)

    def body(i, carry):
        """Body function for lax.fori_loop - processes one iteration"""
        token_set_carry, c_list_carry, kl_list_carry = carry
        
        # Per-iteration RNG derived via fold_in to avoid dynamic indexing
        key_i = jax.random.fold_in(rng, i)

        if cfg.input_dependent:
            # Input-dependent: use full set including sample i (no dummy)
            idx = pos  # (ns,)
        else:
            # Input-independent (LOO) with dummy slot:
            # For each position j in [0, ns-1], map to source index:
            # - If j < i: use j
            # - If j >= i: use j+1 (skip i, point to either real or dummy at ns)
            idx = jnp.where(pos < i, pos, pos + 1)  # (ns,), exactly one element == ns

        # Take subset with fixed shape (b, ns, C, H, W)
        x_subset = jnp.take(batch_set_pad, idx, axis=1)

        # Timestep embedding subset: (b, ns, hdim) -> (b*ns, hdim)
        t_emb_subset = jnp.take(t_emb_pad, idx, axis=1).reshape(
            b * ns, cfg.hdim
        )
        
        # Encode subset
        if need_tokens:
            # Get both hc and tokens for lag mode
            hc, tokens = encode_set(
                params["encoder"],
                enc,
                x_subset,
                cfg,
                train=train,
                t_emb=t_emb_subset,
                return_tokens=True,
                rng=key_i,
            )
            # tokens: (b*ns, num_patches, hdim) from encode_set
            # Reshape to (b, ns, num_patches, hdim) and extract tokens for image i
            b_actual = tokens.shape[0] // ns
            num_patches = tokens.shape[1]
            hdim_actual = tokens.shape[2]
            tokens_reshaped = tokens.reshape(b_actual, ns, num_patches, hdim_actual)  # (b, ns, np, dim)
            # Extract tokens for image i (the target image in this iteration)
            tokens_i = tokens_reshaped[:, i, :, :]  # (b, np, dim)
            # Update token_set at index i
            token_set_carry = token_set_carry.at[:, i, :, :].set(tokens_i)
        else:
            # Only get hc for film mode
            hc = encode_set(
                params["encoder"],
                enc,
                x_subset,
                cfg,
                train=train,
                t_emb=t_emb_subset,
                return_tokens=False,
                rng=key_i,
            )
        
        # Apply posterior if variational (works on pooled hc ONLY, not tokens)
        # CRITICAL: c_vec is only for KL loss, NOT used for conditioning in lag mode
        c_vec, klc = sample_context(key_i, hc, cfg, posterior, params_post)
        
        # Update c_list for film mode
        if not need_tokens:
            # Store c_vec in pre-allocated array: (b, hdim) -> (b, 1, hdim) -> store at index i
            c_list_carry = c_list_carry.at[:, i, :].set(c_vec)
        
        # Update KL list if variational
        if klc is not None:
            kl_list_carry = kl_list_carry.at[i].set(klc)
        
        return (token_set_carry, c_list_carry, kl_list_carry)
    
    # Run the loop using lax.fori_loop (prevents unrolling in JIT)
    final_token_set, final_c_list, final_kl_list = lax.fori_loop(
        0, ns, body, (init_token_set, init_c_list, init_kl_list)
    )

    if need_tokens:
        # For lag mode: use patch tokens for cross-attention
        # final_token_set: (b, ns, num_patches, hdim)
        num_patches = final_token_set.shape[2]
        
        # ASSERT 5: Validate final conditioning shape
        assert final_token_set.shape[0] == b, f"token_set batch mismatch: got {final_token_set.shape[0]}, expected {b}"
        assert final_token_set.shape[1] == ns, f"token_set ns mismatch: got {final_token_set.shape[1]}, expected {ns}"
        assert final_token_set.shape[3] == cfg.hdim, f"token_set hdim mismatch: got {final_token_set.shape[3]}, expected {cfg.hdim}"
        
        # Reshape to (b*ns, num_patches, hdim) for DiT cross-attention
        c = final_token_set.reshape(b * ns, num_patches, cfg.hdim)
        
        # MEMORY OPTIMIZATION: Pool context tokens to reduce Nk (attention memory scales as Nk^2)
        if cfg.context_pool_size > 0 and cfg.context_pool_size < num_patches:
            # Validate divisibility before pooling
            if num_patches % cfg.context_pool_size != 0:
                raise ValueError(
                    f"num_patches ({num_patches}) must be divisible by context_pool_size ({cfg.context_pool_size}). "
                    f"Current config: image_size={cfg.image_size}, patch_size={cfg.patch_size}, "
                    f"num_patches={(cfg.image_size // cfg.patch_size) ** 2}. "
                    f"Please adjust context_pool_size to be a divisor of {num_patches}."
                )
            
            # Average pool tokens to reduce from num_patches to context_pool_size
            # This reduces attention memory by (num_patches/context_pool_size)^2
            pool_factor = num_patches // cfg.context_pool_size
            # Reshape: (b*ns, num_patches, hdim) -> (b*ns, context_pool_size, pool_factor, hdim)
            c_pooled = c.reshape(b * ns, cfg.context_pool_size, pool_factor, cfg.hdim)
            # Average pool: (b*ns, context_pool_size, hdim)
            c = jnp.mean(c_pooled, axis=2)
            num_patches = cfg.context_pool_size
            if not hasattr(leave_one_out_c, "_logged_pooling"):
                import sys
                print(f"\n[INFO] Context token pooling: {final_token_set.shape[2]} -> {num_patches} tokens (memory reduction: {(final_token_set.shape[2]/num_patches)**2:.1f}x)", file=sys.stderr)
                leave_one_out_c._logged_pooling = True
        
        # ASSERT 6: Final conditioning shape
        assert c.shape == (b * ns, num_patches, cfg.hdim), \
            f"Final c shape mismatch: got {c.shape}, expected ({b*ns}, {num_patches}, {cfg.hdim})"
        
        # DEBUG: Log token shapes
        if not hasattr(leave_one_out_c, "_logged_tokens"):
            import sys
            print(f"\n[DEBUG leave_one_out_c] Using patch tokens for lag mode:", file=sys.stderr)
            print(f"  - token_set shape: {final_token_set.shape}", file=sys.stderr)
            print(f"  - c (final) shape: {c.shape}", file=sys.stderr)
            print(f"  - num_patches: {num_patches}", file=sys.stderr)
            print(f"  - Expected: (b*ns={b*ns}, num_patches={num_patches}, hdim={cfg.hdim})", file=sys.stderr)
            leave_one_out_c._logged_tokens = True
    else:
        # For film mode: use pooled vectors
        # final_c_list: (b, ns, hdim)
        c = final_c_list.reshape(b * ns, cfg.hdim)

    # Compute KL total (mask out unused entries if not variational)
    klc_total = None
    if cfg.mode_context == "variational":
        # All entries in final_kl_list are valid
        klc_total = final_kl_list.mean()

    return c, klc_total


def fix_set_size(batch_set: Array, target_ns: int) -> Array:
    """
    Fix batch_set to have exactly target_ns images per set.
    This ensures consistent shapes for JIT compilation.
    
    Args:
        batch_set: (b, ns, C, H, W)
        target_ns: Target number of images per set
        
    Returns:
        batch_set: (b, target_ns, C, H, W)
    """
    b, ns, C, H, W = batch_set.shape
    if ns > target_ns:
        # Crop to target_ns - take first target_ns images
        return batch_set[:, :target_ns]
    elif ns < target_ns:
        # Pad with zeros to reach target_ns
        pad = jnp.zeros((b, target_ns - ns, C, H, W), dtype=batch_set.dtype)
        return jnp.concatenate([batch_set, pad], axis=1)
    else:
        # Already correct size, but return explicitly to ensure shape consistency
        return batch_set


def vfsddpm_loss(
    rng: PRNGKey,
    params: Dict[str, Any],
    modules: Dict[str, Any],
    batch_set: Array,  # (bs, ns, C, H, W), values in [-1, 1] (images if use_vae=False, latents if use_vae=True)
    cfg: VFSDDPMConfig,
    train: bool = True,
) -> Dict[str, Array]:
    """
    Compute VFSDDPM loss (diffusion + optional KL).
    """
    diffusion: GaussianDiffusion = modules["diffusion"]
    dit: DiT = modules["dit"]

    # CRITICAL: Always normalize batch_set to cfg.sample_size to prevent JIT recompilation
    # This must be done even if ns == cfg.sample_size to ensure consistent shapes
    # Some batches might have ns != cfg.sample_size due to dataset variations
    batch_set = fix_set_size(batch_set, cfg.sample_size)
    b, ns = batch_set.shape[:2]
    # Double-check after fix_set_size
    assert ns == cfg.sample_size, f"After fix_set_size: batch_set ns={ns} != cfg.sample_size={cfg.sample_size}. This will cause JAX to compile multiple versions!"
    
    # Keep a copy of images for encoder when encoder_uses_vae is False.
    batch_images = batch_set
    batch_latents = batch_set

    # Encode images to latents if VAE is enabled (for diffusion / DiT input).
    # Note: batch_set comes in as images (bs, ns, 3, H, W) when use_vae=True.
    if cfg.use_vae:
        vae = modules["vae"]
        vae_params = params["vae"]
        # Use original_image_size from cfg (set during init_models)
        # Fallback to calculating from latent_size if not set (for backward compatibility)
        if cfg.original_image_size > 0:
            original_image_size = cfg.original_image_size
        else:
            original_image_size = cfg.latent_size * vae.downscale_factor  # Recover original size
        
        # Reshape to HWC format for VAE: (bs, ns, C, H, W) -> (bs*ns, H, W, C)
        bs, ns, C, H, W = batch_set.shape
        # Validate image dimensions (allow some tolerance for rounding)
        assert abs(H - original_image_size) <= 1 and abs(W - original_image_size) <= 1, \
            f"Expected image size {original_image_size}x{original_image_size}, got {H}x{W}"
        assert C == 3, f"Expected 3 image channels for VAE encoding, got {C}"
        
        # Log encoding (only first time to avoid spam)
        if not hasattr(vfsddpm_loss, "_logged_vae_encode"):
            import sys
            print(f"\n[VAE ENCODE] vfsddpm_loss: Encoding images → latents", file=sys.stderr)
            print(f"  Input shape (images): {batch_set.shape} (bs={bs}, ns={ns}, C={C}, H={H}, W={W})", file=sys.stderr)
            print(f"  Original image size: {original_image_size}×{original_image_size}", file=sys.stderr)
            vfsddpm_loss._logged_vae_encode = True
        
        batch_hwc = batch_set.transpose(0, 1, 3, 4, 2).reshape(bs * ns, H, W, C)
        
        # Encode to latents: (bs*ns, H, W, 3) -> (bs*ns, latent_H, latent_W, 4)
        rng, vae_rng = jax.random.split(rng)
        latents_hwc = vae.encode(vae_rng, batch_hwc, scale=True)
        
        # Reshape back to CHW format: (bs*ns, latent_H, latent_W, 4) -> (bs, ns, 4, latent_H, latent_W)
        latent_H, latent_W = latents_hwc.shape[1], latents_hwc.shape[2]
        batch_latents = latents_hwc.transpose(0, 3, 1, 2).reshape(bs, ns, 4, latent_H, latent_W)
        
        # Log after encoding (only first time)
        if not hasattr(vfsddpm_loss, "_logged_vae_encode_after"):
            import sys
            print(f"  Output shape (latents): {batch_latents.shape} (bs={bs}, ns={ns}, C=4, H={latent_H}, W={latent_W})", file=sys.stderr)
            print(f"  Latent size: {latent_H}×{latent_W} (downscale: {H//latent_H}x)", file=sys.stderr)
            print(f"  ✅ Successfully encoded to latent space\n", file=sys.stderr)
            vfsddpm_loss._logged_vae_encode_after = True
    else:
        # No VAE: latents are just the original images
        batch_latents = batch_set
    
    rng, t_key, noise_key = jax.random.split(rng, 3)
    t = jax.random.randint(t_key, (b,), 0, diffusion.num_timesteps)
    t_rep = jnp.repeat(t, ns, axis=0)

    # conditioning
    rng_c, rng_loss, rng_dit_dropout = jax.random.split(noise_key, 3)
    # Encoder sees images if encoder_uses_vae=False, otherwise it sees latents.
    if cfg.use_vae and not cfg.encoder_uses_vae:
        cond_batch = batch_images
    else:
        cond_batch = batch_latents
    c, klc = leave_one_out_c(
        rng_c, params, modules, cond_batch, cfg, train=train, t=t
    )

    # flatten images/latents for diffusion model input
    x = batch_latents.reshape(b * ns, *batch_latents.shape[2:])

    def model_fn(x_in, t_in, _c_unused, **kwargs):
        # ASSERT 6 (continued): Verify x and c have matching batch dimension
        if cfg.mode_conditioning == "lag":
            assert x_in.shape[0] == c.shape[0], \
                f"Batch mismatch: x_in.shape[0]={x_in.shape[0]}, c.shape[0]={c.shape[0]}"
            assert c.ndim == 3, f"Lag mode c must be 3D (b*ns, num_patches, hdim), got {c.ndim}D with shape {c.shape}"
        # Pass rngs for dropout - Flax nn.Dropout needs PRNG key even when deterministic=True
        apply_kwargs = {"train": train}
        if train:
            apply_kwargs["rngs"] = {"dropout": rng_dit_dropout}
        return dit.apply(params["dit"], x_in, t_in, c=c, **apply_kwargs, **kwargs)

    losses = diffusion.training_losses(
        rng_loss, model_fn, x, t_rep, c=None, model_kwargs={}
    )
    # aggregate
    total = mean_flat(losses["loss"]).mean()
    if klc is not None:
        total = total + klc
        losses["klc"] = klc
    losses["loss"] = total
    # Heavy debug metrics (context/data/timestep stats) are gated to avoid huge
    # reduction graphs in XLA, especially for lag mode where c is very large.
    if getattr(cfg, "debug_metrics", False):
        # Add debug metrics from context (scalars only, don't store large tensor)
        # This avoids memory leak from storing large context tensor in lag mode
        if cfg.mode_conditioning == "lag":
            # For lag mode: c is (b*ns, num_patches, hdim) - very large!
            # Only compute scalar metrics, don't store tensor
            losses["debug/context_norm"] = jnp.linalg.norm(c)
            losses["debug/context_mean"] = jnp.mean(jnp.abs(c))
            losses["debug/context_max"] = jnp.max(jnp.abs(c))
            losses["debug/context_std"] = jnp.std(c)
        else:
            # For film mode: c is (b*ns, hdim) - smaller, but still avoid storing
            losses["debug/context_norm"] = jnp.linalg.norm(c)
            losses["debug/context_mean"] = jnp.mean(jnp.abs(c))
        
        # --- DEBUG LOGGING BLOCK START ---
        # 1. Input stats
        losses["debug/data_min"] = jnp.min(batch_set)
        losses["debug/data_max"] = jnp.max(batch_set)
        losses["debug/data_mean"] = jnp.mean(batch_set)
        
        # 2. Context stats (Encoder output)
        # c shape: (B*ns, hdim) for film or (B*ns, num_patches, hdim) for lag
        losses["debug/c_mean"] = jnp.mean(c)
        losses["debug/c_std"] = jnp.std(c)
        # Compute norm: mean(sqrt(sum(c^2, axis=-1)))
        # For film: (b*ns, hdim) -> (b*ns,) -> scalar
        # For lag: (b*ns, num_patches, hdim) -> (b*ns, num_patches) -> scalar
        losses["debug/c_norm"] = jnp.mean(jnp.sqrt(jnp.sum(c**2, axis=-1)))
        
        # 3. Time embedding stats
        # Re-compute t_emb for comparison
        base_dim = cfg.hdim // 4
        t_base = timestep_embedding_jax(t, base_dim)  # (b, base_dim)
        t_emb_chk = modules["time_embed"].apply(params["time_embed"], t_base)  # (b, hdim)
        # Compute norm per sample
        t_norm_per_sample = jnp.sqrt(jnp.sum(t_emb_chk**2, axis=-1))  # (b,)
        losses["debug/t_norm"] = jnp.mean(t_norm_per_sample)
        
        # 4. Signal Ratio (Quan trọng)
        # If ratio < 0.01, Encoder quá yếu so với Time Embedding
        losses["debug/signal_ratio_c_t"] = (
            losses["debug/c_norm"] / (losses["debug/t_norm"] + 1e-6)
        )
        
        # 5. Additional magnitude comparisons
        mag_t = jnp.mean(jnp.abs(t_emb_chk))
        mag_c = jnp.mean(jnp.abs(c))
        losses["debug/magnitude_time"] = mag_t
        losses["debug/magnitude_context"] = mag_c
        losses["debug/ratio_c_over_t"] = mag_c / (mag_t + 1e-6)
        
        # 6. Context after LayerNorm (simulate what happens in DiTBlock)
        # Compute c_norm after LayerNorm (same as DiTBlock)
        if cfg.mode_conditioning == "film":
            # For film mode: c is (b*ns, hdim)
            # Simulate LayerNorm: normalize to mean=0, std=1
            c_mean = jnp.mean(c, axis=-1, keepdims=True)  # (b*ns, 1)
            c_std = jnp.std(c, axis=-1, keepdims=True)  # (b*ns, 1)
            c_norm = (c - c_mean) / (c_std + 1e-6)
            
            # Expand t_emb to match c shape: (b, hdim) -> (b*ns, hdim)
            t_emb_expanded = jnp.repeat(t_emb_chk, ns, axis=0)  # (b*ns, hdim)
            
            # Compute norms after layer norm
            c_norm_norm = jnp.mean(jnp.sqrt(jnp.sum(c_norm**2, axis=-1)))  # scalar
            t_emb_norm = jnp.mean(jnp.sqrt(jnp.sum(t_emb_expanded**2, axis=-1)))  # scalar
            
            # Ratio: c_norm / t_emb (after layer norm)
            losses["debug/c_norm_after_ln"] = c_norm_norm
            losses["debug/t_norm_for_ratio"] = t_emb_norm
            losses["debug/ratio_c_norm_t"] = c_norm_norm / (t_emb_norm + 1e-6)
            
            # Magnitude ratio after layer norm
            c_norm_mag = jnp.mean(jnp.abs(c_norm))
            t_emb_mag = jnp.mean(jnp.abs(t_emb_expanded))
            losses["debug/magnitude_c_norm"] = c_norm_mag
            losses["debug/magnitude_t_emb"] = t_emb_mag
            losses["debug/ratio_mag_c_norm_t"] = c_norm_mag / (t_emb_mag + 1e-6)
            
            # Context norm std (should be ~1.0 after LayerNorm)
            c_norm_std = jnp.std(c_norm, axis=-1)  # (b*ns,)
            losses["debug/c_norm_std"] = jnp.mean(c_norm_std)  # Should be close to 1.0
            
        elif cfg.mode_conditioning == "lag":
            # For lag mode: c is (b*ns, num_patches, hdim)
            # Normalize per token (along last dimension)
            c_mean = jnp.mean(c, axis=-1, keepdims=True)  # (b*ns, num_patches, 1)
            c_std = jnp.std(c, axis=-1, keepdims=True)
            c_norm = (c - c_mean) / (c_std + 1e-6)
            
            # Expand t_emb: (b, hdim) -> (b*ns, hdim)
            t_emb_expanded = jnp.repeat(t_emb_chk, ns, axis=0)  # (b*ns, hdim)
            
            # For lag mode, compare per-token norm with t_emb
            c_norm_per_token = jnp.sqrt(jnp.sum(c_norm**2, axis=-1))  # (b*ns, num_patches)
            c_norm_norm = jnp.mean(c_norm_per_token)  # scalar
            t_emb_norm = jnp.mean(jnp.sqrt(jnp.sum(t_emb_expanded**2, axis=-1)))  # scalar
            
            losses["debug/c_norm_after_ln"] = c_norm_norm
            losses["debug/t_norm_for_ratio"] = t_emb_norm
            losses["debug/ratio_c_norm_t"] = c_norm_norm / (t_emb_norm + 1e-6)
            
            # Context norm std (should be ~1.0 after LayerNorm)
            c_norm_std = jnp.std(c_norm, axis=-1)  # (b*ns, num_patches)
            losses["debug/c_norm_std"] = jnp.mean(c_norm_std)  # Should be close to 1.0
        
        # 7. Log context_scale parameters from DiT blocks
        # Extract context_scale from DiT params (if available)
        try:
            dit_params = params.get("dit", {})
            if dit_params:
                # DiT params structure: blocks are in a list or dict
                # We need to extract context_scale from each block
                # Flax stores params as nested dict, blocks might be in 'blocks' or similar
                def extract_context_scale(tree, prefix=""):
                    """Recursively extract context_scale parameters"""
                    scales = []
                    if isinstance(tree, dict):
                        for key, value in tree.items():
                            if key == "context_scale" or key == "context_scale_lag":
                                # Found a context_scale parameter
                                if hasattr(value, 'shape'):  # It's a JAX array
                                    scales.append(value)
                                elif isinstance(value, dict) and 'params' in value:
                                    # Flax param structure: {'params': {...}, 'params_axes': {...}}
                                    scales.append(value['params'])
                            else:
                                scales.extend(extract_context_scale(value, f"{prefix}/{key}" if prefix else key))
                    elif isinstance(tree, (list, tuple)):
                        for i, item in enumerate(tree):
                            scales.extend(extract_context_scale(item, f"{prefix}[{i}]"))
                    return scales
                
                context_scales = extract_context_scale(dit_params)
                if context_scales:
                    # Aggregate all context_scale values
                    # Flatten each scale array and concatenate
                    all_scales_list = []
                    for scale in context_scales:
                        if hasattr(scale, 'shape'):  # JAX array
                            all_scales_list.append(jnp.ravel(scale))
                        elif isinstance(scale, dict):
                            # Handle nested dict structure
                            for v in scale.values():
                                if hasattr(v, 'shape'):
                                    all_scales_list.append(jnp.ravel(v))
                    
                    if all_scales_list:
                        all_scales = jnp.concatenate(all_scales_list)
                        losses["debug/context_scale_mean"] = jnp.mean(all_scales)
                        losses["debug/context_scale_std"] = jnp.std(all_scales)
                        losses["debug/context_scale_min"] = jnp.min(all_scales)
                        losses["debug/context_scale_max"] = jnp.max(all_scales)
        except Exception as e:
            # If extraction fails, skip (params structure might be different)
            # Don't log error to avoid cluttering output
            pass
        
        # --- DEBUG LOGGING BLOCK END ---
    
    # Don't store c tensor - it's too large and causes memory leak
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
