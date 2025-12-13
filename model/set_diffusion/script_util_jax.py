"""
Script utilities for DiT (Diffusion Transformer) model in JAX/Flax.
Chuyên dụng cho denoiser transformer thay thế UNet.
"""

import argparse
import inspect
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp

from . import gaussian_diffusion_jax as gd_jax
from .dit_jax import DiT

NUM_CLASSES = 1000


def diffusion_defaults():
    """
    Defaults for diffusion process (JAX version).
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def dit_model_defaults():
    """
    Defaults for DiT (Diffusion Transformer) model.
    Thay thế các tham số UNet bằng tham số DiT.
    """
    res = dict(
        image_size=32,
        in_channels=3,
        hidden_size=468,  # DiT hidden dimension (must be divisible by num_heads=9, 468/9=52 head_dim)
        depth=6,  # DiT depth (reduced from 12 for faster training & better gradient flow)
        num_heads=9,  # DiT attention heads (468/9 = 52 head_dim)
        mlp_ratio=3.0,  # DiT MLP expansion ratio (reduced from 4.0 for efficiency)
        patch_size=2,  # DiT patch size
        context_channels=448,  # Few-shot conditioning dimension (MUST MATCH hdim from encoder!)
        mode_conditioning="film",  # "film" or "lag"
        dropout=0.0,
        class_cond=False,
        class_dropout_prob=0.1,  # DiT class dropout for classifier-free guidance
        use_fp16=False,
    )
    res.update(diffusion_defaults())
    return res


def model_and_diffusion_defaults():
    """
    Alias for dit_model_defaults() để tương thích với code hiện tại.
    """
    return dit_model_defaults()


def create_model_and_diffusion(
    image_size,
    in_channels,
    class_cond,
    learn_sigma,
    hidden_size,  # DiT: hidden_size thay cho num_channels
    depth,  # DiT: depth thay cho num_res_blocks
    context_channels,
    mode_conditioning,
    num_heads,  # DiT: num_heads
    mlp_ratio,  # DiT: mlp_ratio
    patch_size,  # DiT: patch_size
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    class_dropout_prob=0.1,
    use_fp16=False,
    # Các tham số không dùng trong DiT (giữ để tương thích)
    channel_mult="",
    num_head_channels=-1,
    num_heads_upsample=-1,
    attention_resolutions="",
    use_checkpoint=False,
    use_scale_shift_norm=False,
    resblock_updown=False,
    use_new_attention_order=False,
):
    """
    Create DiT model and diffusion process (JAX version).

    Args:
        image_size: Image size (must be divisible by patch_size)
        in_channels: Input channels
        class_cond: Whether to use class conditioning
        learn_sigma: Whether to learn sigma
        hidden_size: DiT hidden dimension
        depth: DiT transformer depth
        context_channels: Few-shot conditioning dimension
        mode_conditioning: "film" or "lag"
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        patch_size: Patch size for image patching
        dropout: Dropout rate
        diffusion_steps: Number of diffusion steps
        noise_schedule: Noise schedule ("linear" or "cosine")
        timestep_respacing: Timestep respacing
        use_kl: Whether to use KL loss
        predict_xstart: Whether to predict x_start
        rescale_timesteps: Whether to rescale timesteps
        rescale_learned_sigmas: Whether to rescale learned sigmas
        class_dropout_prob: Class dropout probability for classifier-free guidance
        use_fp16: Whether to use float16 (not fully supported in JAX, kept for compatibility)
        ... (các tham số không dùng, giữ để tương thích với UNet interface)

    Returns:
        model: DiT Flax module
        diffusion: GaussianDiffusion instance (JAX)
    """
    model = create_dit_model(
        image_size=image_size,
        in_channels=in_channels,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        patch_size=patch_size,
        context_channels=context_channels,
        mode_conditioning=mode_conditioning,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        class_dropout_prob=class_dropout_prob,
        dropout=dropout,
    )

    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )

    return model, diffusion


def create_dit_model(
    image_size,
    in_channels,
    hidden_size,
    depth,
    num_heads,
    mlp_ratio,
    patch_size,
    context_channels=0,
    mode_conditioning="film",
    learn_sigma=False,
    class_cond=False,
    class_dropout_prob=0.1,
    dropout=0.0,
):
    """
    Create DiT (Diffusion Transformer) model.

    Args:
        image_size: Image size (must be divisible by patch_size)
        in_channels: Input channels
        hidden_size: DiT hidden dimension
        depth: Number of DiT blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        patch_size: Patch size for image patching
        context_channels: Few-shot conditioning dimension (0 if not used)
        mode_conditioning: "film" or "lag" for few-shot conditioning
        learn_sigma: Whether to learn sigma (output channels = 2 * in_channels if True)
        class_cond: Whether to use class conditioning
        class_dropout_prob: Class dropout probability for classifier-free guidance
        dropout: Dropout rate

    Returns:
        DiT Flax module
    """
    # Validate image_size is divisible by patch_size
    if image_size % patch_size != 0:
        # Handle 28x28 images (MNIST) by padding to 32x32
        if image_size == 28:
            # Will be handled inside DiT forward pass
            pass
        else:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
            )

    num_classes = NUM_CLASSES if class_cond else 0

    return DiT(
        patch_size=patch_size,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        class_dropout_prob=class_dropout_prob,
        num_classes=num_classes,
        learn_sigma=learn_sigma,
        context_channels=context_channels,
        mode_conditioning=mode_conditioning,
        dropout_rate=dropout,
    )


# Alias for compatibility
create_model = create_dit_model


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    """
    Create Gaussian diffusion process (JAX version).

    Args:
        steps: Number of diffusion steps
        learn_sigma: Whether to learn sigma
        sigma_small: Whether to use small sigma
        noise_schedule: Noise schedule ("linear" or "cosine")
        use_kl: Whether to use KL loss
        predict_xstart: Whether to predict x_start
        rescale_timesteps: Whether to rescale timesteps
        rescale_learned_sigmas: Whether to rescale learned sigmas
        timestep_respacing: Timestep respacing (empty string means no respacing)

    Returns:
        GaussianDiffusion instance (JAX)
    """
    betas = gd_jax.get_named_beta_schedule(noise_schedule, steps)

    if use_kl:
        loss_type = gd_jax.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd_jax.LossType.RESCALED_MSE
    else:
        loss_type = gd_jax.LossType.MSE

    # Handle timestep respacing
    # Note: JAX version doesn't have SpacedDiffusion wrapper yet
    # For now, we'll use all timesteps
    # TODO: Implement timestep respacing for JAX version if needed

    return gd_jax.GaussianDiffusion(
        betas=betas,
        model_mean_type=(
            gd_jax.ModelMeanType.EPSILON
            if not predict_xstart
            else gd_jax.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd_jax.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd_jax.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd_jax.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    """
    Add dictionary to argument parser.
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    """
    Convert argparse args to dictionary.
    """
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    Convert string to boolean.
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
