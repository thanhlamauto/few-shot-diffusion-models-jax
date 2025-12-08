from typing import Any, Dict, Tuple

import jax

from model.vfsddpm_jax import VFSDDPMConfig, init_models


def select_model_jax(args, rng: jax.Array) -> Tuple[Dict[str, Any], Dict[str, Any], VFSDDPMConfig]:
    """
    Build params, modules, cfg for the requested JAX model.
    Currently supports: vfsddpm_jax.
    """
    if args.model != "vfsddpm_jax":
        raise ValueError(f"Unsupported JAX model: {args.model}")

    cfg = VFSDDPMConfig(
        image_size=args.image_size,
        in_channels=args.in_channels,
        sample_size=args.sample_size,
        encoder_mode=args.encoder_mode,
        hdim=args.hdim,
        pool=getattr(args, "pool", "cls"),
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        patch_size=args.patch_size,
        context_channels=args.context_channels,
        mode_conditioning=args.mode_conditioning,
        class_cond=False,
        diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        learn_sigma=args.learn_sigma,
        timestep_respacing=args.timestep_respacing,
        use_kl=args.use_kl,
        predict_xstart=args.predict_xstart,
        rescale_timesteps=args.rescale_timesteps,
        rescale_learned_sigmas=args.rescale_learned_sigmas,
        mode_context=args.mode_context,
    )
    params, modules = init_models(rng, cfg)
    return params, modules, cfg
