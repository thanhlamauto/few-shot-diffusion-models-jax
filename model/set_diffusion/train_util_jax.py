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
):
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
):
    """
    Returns a pmapped train step: state, batch, rng -> (state, metrics).
    loss_fn(params, batch, rng) should return dict with key 'loss'.
    """

    def step(state: TrainStatePmap, batch, rng):
        def loss_wrap(p):
            losses = loss_fn(p, batch, rng)
            return losses["loss"], losses

        (loss, losses), grads = jax.value_and_grad(loss_wrap, has_aux=True)(state.params)
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
        return new_state, metrics

    return jax.pmap(step, axis_name="devices")


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
):
    """
    Sampling helper that uses provided EMA params and model_apply.
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
        )
    return diffusion.p_sample_loop(
        rng,
        apply,
        shape,
        c=conditioning,
        clip_denoised=clip_denoised,
    )

