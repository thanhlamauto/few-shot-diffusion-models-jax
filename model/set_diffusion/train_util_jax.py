"""
Minimal training utilities for the JAX/Flax DiT diffusion stack
defined in `script_util_jax.py`.

Notes:
- Single-host, single-device reference implementation (no pmap/sharding).
- Uses optax for AdamW + EMA tracking of parameters.
- Expected model signature matches DiT.__call__(x, t, c=None, y=None, train=False).
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax import struct

from .nn_jax import update_ema, mean_flat


Array = jnp.ndarray
PRNGKey = jax.Array
ModelApply = Callable[..., Array]


class TrainState(train_state.TrainState):
    """
    Extends Flax TrainState to also keep EMA parameters.
    Flax TrainState is frozen; inherit without @dataclass.
    """
    ema_params: Any


def create_train_state(
    rng: PRNGKey,
    model,
    diffusion,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
):
    """
    Initialize model parameters and optimizer/EMA state.
    """
    dummy_x = jnp.zeros((1, 3, model.patch_size * 4, model.patch_size * 4), dtype=jnp.float32)
    dummy_t = jnp.zeros((1,), dtype=jnp.int32)
    params = model.init(rng, dummy_x, dummy_t, c=None, y=None, train=True)

    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx, ema_params=params)
    state = state.replace(step=jnp.array(0, dtype=jnp.int32))
    return state


def _loss_fn(
    params: Any,
    model_apply: ModelApply,
    diffusion,
    rng: PRNGKey,
    batch: Array,
    conditioning: Optional[Any],
) -> Tuple[Array, Dict[str, Array]]:
    """
    Compute diffusion training loss and return aux stats.
    """
    rng, t_key, noise_key = jax.random.split(rng, 3)
    b = batch.shape[0]
    t = jax.random.randint(t_key, (b,), 0, diffusion.num_timesteps)

    def model_fn(x, t_in, c, **kwargs):
        return model_apply(params, x, t_in, c, **kwargs)

    losses = diffusion.training_losses(
        noise_key,
        model_fn,
        batch,
        t,
        c=conditioning,
    )
    total_loss = mean_flat(losses["loss"]).mean()
    return total_loss, {"losses": losses, "t": t}


@jax.jit
def train_step(
    state: TrainState,
    diffusion,
    rng: PRNGKey,
    batch: Array,
    conditioning: Optional[Any] = None,
    ema_rate: float = 0.999,
):
    """
    Single JIT-ed training step (no gradient accumulation, single device).
    """
    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(
        state.params,
        state.apply_fn,
        diffusion,
        rng,
        batch,
        conditioning,
    )
    state = state.apply_gradients(grads=grads)
    new_ema_params = update_ema(state.ema_params, state.params, rate=ema_rate)
    state = state.replace(ema_params=new_ema_params, step=state.step + 1)
    metrics = {"loss": loss, "t_mean": aux["t"].mean()}
    for k, v in aux["losses"].items():
        metrics[f"{k}_mean"] = mean_flat(v).mean()
    return state, metrics


def sample(
    rng: PRNGKey,
    state: TrainState,
    diffusion,
    shape: Tuple[int, ...],
    conditioning: Optional[Any] = None,
    use_ddim: bool = False,
    eta: float = 0.0,
    clip_denoised: bool = True,
):
    """
    Run ancestral (DDPM) or DDIM sampling using EMA params.
    """
    model_apply = lambda x, t, c=None, **kw: state.apply_fn(state.ema_params, x, t, c, train=False, **kw)
    if use_ddim:
        return diffusion.ddim_sample_loop(
            rng,
            model_apply,
            shape,
            c=conditioning,
            clip_denoised=clip_denoised,
            eta=eta,
        )
    else:
        return diffusion.p_sample_loop(
            rng,
            model_apply,
            shape,
            c=conditioning,
            clip_denoised=clip_denoised,
        )


# ----------------------- Pmap multi-device utilities ----------------------- #


class TrainStatePmap(struct.PyTreeNode):
    """
    Pmap-friendly train state holding params, EMA params, opt state, and step.
    """

    params: Any
    ema_params: Any
    opt_state: optax.OptState
    step: jnp.ndarray


def create_train_state_pmap(
    params: Any,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    encoder_lr: Optional[float] = None,
    dit_lr: Optional[float] = None,
):
    """
    Create TrainStatePmap.
    Note: encoder_lr / dit_lr are handled in train_step_pmap via gradient scaling.
    Here we keep a single AdamW optimizer for numerical stability.
    """
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    opt_state = tx.init(params)
    return TrainStatePmap(
        params=params,
        ema_params=params,
        opt_state=opt_state,
        step=jnp.array(0, dtype=jnp.int32),
    ), tx


def _tree_update_ema(ema_params, params, rate: float):
    return jax.tree.map(lambda e, p: update_ema(e, p, rate=rate), ema_params, params)


def shard_batch(batch: Array, n_devices: int) -> Array:
    """
    Reshape leading batch dim to (n_devices, per_device, ...).
    """
    b = batch.shape[0]
    assert b % n_devices == 0, "Batch size must be divisible by number of devices"
    per_dev = b // n_devices
    new_shape = (n_devices, per_dev) + batch.shape[1:]
    return batch.reshape(new_shape)


def train_step_pmap(
    tx: optax.GradientTransformation,
    loss_fn: Callable[[Any, Array, PRNGKey], Dict[str, Array]],
    ema_rate: float = 0.999,
    freeze_dit_steps: int = 0,
    base_lr: float = 1e-4,
    encoder_lr: Optional[float] = None,
    dit_lr: Optional[float] = None,
):
    """
    Returns a pmapped train step: state, batch, rng -> (state, metrics).
    loss_fn(params, batch, rng) should return dict with key 'loss' and optionally 'context'.
    
    Args:
        freeze_dit_steps: If > 0, freeze DiT parameters for the first N steps (only train encoder)
    """

    # Compute LR scales (relative to base_lr) for encoder vs DiT
    def _to_float_or_none(x):
        # Args from argparse may come as strings; normalize here.
        if x is None:
            return None
        if isinstance(x, (float, int)):
            return float(x)
        if isinstance(x, str):
            xs = x.strip()
            if xs == "" or xs.lower() in ("none", "null"):
                return None
            return float(xs)
        # Fallback: try cast
        return float(x)

    enc_lr_val = _to_float_or_none(encoder_lr)
    dit_lr_val = _to_float_or_none(dit_lr)

    enc_scale = (
        (enc_lr_val / base_lr) if (enc_lr_val is not None and base_lr > 0) else 1.0
    )
    dit_scale = (
        (dit_lr_val / base_lr) if (dit_lr_val is not None and base_lr > 0) else 1.0
    )

    def step(state: TrainStatePmap, batch, rng):
        def loss_wrap(p):
            losses = loss_fn(p, batch, rng)
            return losses["loss"], losses

        (loss, losses), grads = jax.value_and_grad(loss_wrap, has_aux=True)(
            state.params
        )
        
        # Freeze DiT for first N steps if enabled
        if freeze_dit_steps > 0:
            should_freeze_dit = state.step < freeze_dit_steps
            
            # Create masked gradients: zero out DiT grads if frozen
            def mask_dit_grads(grad_dict):
                if "dit" in grad_dict:
                    # Zero out DiT gradients if frozen
                    dit_grads_masked = jax.lax.cond(
                        should_freeze_dit,
                        lambda g: jax.tree.map(jnp.zeros_like, g),  # Freeze: zero grads
                        lambda g: g,  # Normal: keep grads
                        grad_dict["dit"]
                    )
                    grad_dict_masked = {**grad_dict, "dit": dit_grads_masked}
                    return grad_dict_masked
                return grad_dict
            
            grads = mask_dit_grads(grads)
        
        # Apply per-module learning rate scaling via gradient scaling
        def scale_module_grads(grad_dict):
            if not isinstance(grad_dict, dict):
                return grad_dict

            new_grads = dict(grad_dict)
            # Encoder-related params
            if "encoder" in new_grads:
                new_grads["encoder"] = jax.tree.map(
                    lambda g: g * enc_scale, new_grads["encoder"]
                )
            if "posterior" in new_grads:
                new_grads["posterior"] = jax.tree.map(
                    lambda g: g * enc_scale, new_grads["posterior"]
                )

            # DiT-related params
            if "dit" in new_grads:
                new_grads["dit"] = jax.tree.map(
                    lambda g: g * dit_scale, new_grads["dit"]
                )
            if "time_embed" in new_grads:
                new_grads["time_embed"] = jax.tree.map(
                    lambda g: g * dit_scale, new_grads["time_embed"]
                )

            return new_grads

        grads = scale_module_grads(grads)

        updates, new_opt_state = tx.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        new_ema_params = _tree_update_ema(state.ema_params, new_params, rate=ema_rate)

        new_state = TrainStatePmap(
            params=new_params,
            ema_params=new_ema_params,
            opt_state=new_opt_state,
            step=state.step + 1,
        )

        # mean metrics over devices for logging outside
        metrics = {"loss": loss}
        if isinstance(losses, dict):
            for k, v in losses.items():
                metrics[k] = v
        
        # Compute debug metrics
        # Note: Context metrics are now computed in loss function to avoid memory leak
        # Just copy scalar metrics if they exist
        if "debug/context_norm" in losses:
            metrics["debug/context_norm"] = losses["debug/context_norm"]
        if "debug/context_mean" in losses:
            metrics["debug/context_mean"] = losses["debug/context_mean"]
        if "debug/context_max" in losses:
            metrics["debug/context_max"] = losses["debug/context_max"]
        if "debug/context_std" in losses:
            metrics["debug/context_std"] = losses["debug/context_std"]
        
        # 2. Gradient norms (layer-wise)
        def get_norm(tree):
            """Helper to compute L2 norm of all leaves in a tree"""
            leaves = jax.tree_util.tree_leaves(tree)
            return jnp.sqrt(sum(jnp.sum(g**2) for g in leaves if g is not None))
        
        if hasattr(grads, 'keys'):
            # Compute gradient norm for each parameter group
            grad_norm_dit = None
            grad_norm_encoder = None
            
            for key in ['dit', 'encoder']:
                if key in grads and grads[key] is not None:
                    grad_tree = grads[key]
                    grad_norm = get_norm(grad_tree)
                    metrics[f"debug/grad_norm_{key}"] = grad_norm
                    
                    if key == 'dit':
                        grad_norm_dit = grad_norm
                    elif key == 'encoder':
                        grad_norm_encoder = grad_norm
            
            # Compute ratio: DiT gradient / Encoder gradient
            # If ratio > 1000, DiT đang học nhưng không truyền tin về Encoder
            if grad_norm_dit is not None and grad_norm_encoder is not None:
                metrics["debug/grad_norm_dit_total"] = grad_norm_dit
                metrics["debug/grad_norm_encoder_total"] = grad_norm_encoder
                metrics["debug/ratio_grad_dit_enc"] = grad_norm_dit / (grad_norm_encoder + 1e-8)
        
        # 3. Parameter norms (for reference)
        for key in ['dit', 'encoder']:
            if key in new_params and new_params[key] is not None:
                param_tree = new_params[key]
                flat_params = jax.tree_util.tree_leaves(param_tree)
                if flat_params:
                    param_norm = jnp.sqrt(sum(jnp.sum(p**2) for p in flat_params if p is not None))
                    metrics[f"debug/param_norm_{key}"] = param_norm
        
        return new_state, metrics

    return jax.pmap(step, axis_name="devices")


def train_step_single_device(
    tx: optax.GradientTransformation,
    loss_fn: Callable[[Any, Array, PRNGKey], Dict[str, Array]],
    ema_rate: float = 0.999,
    freeze_dit_steps: int = 0,
):
    """
    Returns a JIT-compiled train step for SINGLE DEVICE (no pmap).
    Useful for debugging compile-OOM issues.
    
    Args:
        freeze_dit_steps: If > 0, freeze DiT parameters for the first N steps (only train encoder)
    """
    def step(state: TrainStatePmap, batch, rng):
        def loss_wrap(p):
            losses = loss_fn(p, batch, rng)
            return losses["loss"], losses

        (loss, losses), grads = jax.value_and_grad(loss_wrap, has_aux=True)(state.params)
        
        # Freeze DiT for first N steps if enabled
        if freeze_dit_steps > 0:
            should_freeze_dit = state.step < freeze_dit_steps
            
            # Create masked gradients: zero out DiT grads if frozen
            def mask_dit_grads(grad_dict):
                if "dit" in grad_dict:
                    # Zero out DiT gradients if frozen
                    dit_grads_masked = jax.lax.cond(
                        should_freeze_dit,
                        lambda g: jax.tree.map(jnp.zeros_like, g),  # Freeze: zero grads
                        lambda g: g,  # Normal: keep grads
                        grad_dict["dit"]
                    )
                    grad_dict_masked = {**grad_dict, "dit": dit_grads_masked}
                    return grad_dict_masked
                return grad_dict
            
            grads = mask_dit_grads(grads)
        
        updates, new_opt_state = tx.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        new_ema_params = _tree_update_ema(state.ema_params, new_params, rate=ema_rate)

        new_state = TrainStatePmap(
            params=new_params,
            ema_params=new_ema_params,
            opt_state=new_opt_state,
            step=state.step + 1,
        )

        # metrics
        metrics = {"loss": loss}
        if isinstance(losses, dict):
            for k, v in losses.items():
                metrics[k] = v
        
        # Compute debug metrics
        if "debug/context_norm" in losses:
            metrics["debug/context_norm"] = losses["debug/context_norm"]
        if "debug/context_mean" in losses:
            metrics["debug/context_mean"] = losses["debug/context_mean"]
        if "debug/context_max" in losses:
            metrics["debug/context_max"] = losses["debug/context_max"]
        if "debug/context_std" in losses:
            metrics["debug/context_std"] = losses["debug/context_std"]
        
        # Gradient norms
        def get_norm(tree):
            """Helper to compute L2 norm of all leaves in a tree"""
            leaves = jax.tree_util.tree_leaves(tree)
            return jnp.sqrt(sum(jnp.sum(g**2) for g in leaves if g is not None))
        
        if hasattr(grads, 'keys'):
            grad_norm_dit = None
            grad_norm_encoder = None
            
            for key in ['dit', 'encoder']:
                if key in grads and grads[key] is not None:
                    grad_tree = grads[key]
                    grad_norm = get_norm(grad_tree)
                    metrics[f"debug/grad_norm_{key}"] = grad_norm
                    
                    if key == 'dit':
                        grad_norm_dit = grad_norm
                    elif key == 'encoder':
                        grad_norm_encoder = grad_norm
            
            # Compute ratio: DiT gradient / Encoder gradient
            # If ratio > 1000, DiT đang học nhưng không truyền tin về Encoder
            if grad_norm_dit is not None and grad_norm_encoder is not None:
                metrics["debug/grad_norm_dit_total"] = grad_norm_dit
                metrics["debug/grad_norm_encoder_total"] = grad_norm_encoder
                metrics["debug/ratio_grad_dit_enc"] = grad_norm_dit / (grad_norm_encoder + 1e-8)
        
        # Parameter norms
        for key in ['dit', 'encoder']:
            if key in new_params and new_params[key] is not None:
                param_tree = new_params[key]
                flat_params = jax.tree_util.tree_leaves(param_tree)
                if flat_params:
                    param_norm = jnp.sqrt(sum(jnp.sum(p**2) for p in flat_params if p is not None))
                    metrics[f"debug/param_norm_{key}"] = param_norm
        
        return new_state, metrics

    return jax.jit(step)  # Single device: use jit instead of pmap


def sample_ema(
    rng: PRNGKey,
    ema_params: Any,
    diffusion,
    model_apply: Callable,
    shape: Tuple[int, ...],
    conditioning: Optional[Any] = None,
    use_ddim: bool = False,
    eta: float = 0.0,
    clip_denoised: bool = True,
    ddim_num_steps: Optional[int] = None,
):
    """
    Sampling helper that uses provided EMA params and model_apply.
    
    Args:
        ddim_num_steps: Number of DDIM sampling steps. If None, use all timesteps.
                       Typical values: 50, 100, 250 for faster sampling.
    """
    apply = lambda x, t, c=None, **kw: model_apply(ema_params, x, t, c, train=False, **kw)
    if use_ddim:
        return diffusion.ddim_sample_loop(
            rng,
            apply,
            shape,
            c=conditioning,
            clip_denoised=clip_denoised,
            eta=eta,
            ddim_num_steps=ddim_num_steps,
        )
    return diffusion.p_sample_loop(
        rng,
        apply,
        shape,
        c=conditioning,
        clip_denoised=clip_denoised,
    )

