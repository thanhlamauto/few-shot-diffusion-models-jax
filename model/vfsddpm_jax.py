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
from typing import Any, Dict, Optional, Tuple

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
    dit_params = dit.init(rng_dit, dummy_x, dummy_t,
                          c=None, y=None, train=False)

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
) -> Array:
    """
    Encode a set (or single image) into a set-level representation hc.
    
    Args:
        t_emb: Optional timestep embedding (b*ns, t_dim)
               If provided, encoder becomes time-aware
    """
    # DEBUG: Log encode_set call (only first call)
    if not hasattr(encode_set, "_logged"):
        import sys
        print(f"\n[DEBUG encode_set] Called with:", file=sys.stderr)
        print(f"  - x_set shape: {x_set.shape}", file=sys.stderr)
        print(f"  - t_emb: {'None' if t_emb is None else f'shape={t_emb.shape}'}", file=sys.stderr)
        print(f"  - train: {train}", file=sys.stderr)
        print(f"  - encoder_mode: {cfg.encoder_mode}", file=sys.stderr)
        encode_set._logged = True
    
    if cfg.encoder_mode == "vit":
        # flatten and average
        b, ns = x_set.shape[:2]
        x_flat = x_set.reshape(b * ns, *x_set.shape[2:])
        # encoder returns just hc for __call__
        hc_flat = encoder.apply(params_enc, x_flat, train=train)
        hc = hc_flat.reshape(b, ns, -1).mean(axis=1)
    else:
        # encoder returns (hc, patches, cls) for forward_set
        # Must explicitly call forward_set method (apply() defaults to __call__)
        hc, _, _ = encoder.apply(
            params_enc, x_set, t_emb=t_emb, train=train, method=encoder.forward_set)
        if hc.ndim == 3:
            hc = hc.mean(axis=1)
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
        film -> c: (bs * ns, hdim)
        lag  -> c: (bs * ns, tokens, hdim)  (tokens=1 here)
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
        print(f"  - Dropout: {cfg.dropout}", file=sys.stderr)
        leave_one_out_c._logged = True

    kl_list = []
    c_list = []
    rngs = jax.random.split(rng, ns)
    for i in range(ns):
        idx = [k for k in range(ns) if k != i]
        x_subset = batch_set[:, idx]  # (b, ns-1, C, H, W)
        # Expand t_emb for this subset: (b, hdim) -> (b * (ns-1), hdim)
        subset_ns = ns - 1
        t_emb_subset = jnp.repeat(t_emb_set[:, None, :], subset_ns, axis=1).reshape(
            b * subset_ns, cfg.hdim
        )
        hc = encode_set(
            params["encoder"], enc, x_subset, cfg, train=train, t_emb=t_emb_subset
        )
        c_vec, klc = sample_context(rngs[i], hc, cfg, posterior, params_post)
        c_list.append(c_vec[:, None, ...])  # keep set slot
        if klc is not None:
            kl_list.append(klc)

    c_set = jnp.concatenate(c_list, axis=1)  # (b, ns, hdim)

    if cfg.mode_conditioning == "lag":
        # provide one token per element (can be extended)
        c = c_set.reshape(b * ns, 1, c_set.shape[-1])
    else:
        c = c_set.reshape(b * ns, c_set.shape[-1])

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
