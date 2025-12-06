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

from .nn_jax import update_ema, mean_flat


Array = jnp.ndarray
PRNGKey = jax.Array
ModelApply = Callable[..., Array]


@dataclass
class TrainState(train_state.TrainState):
    """
    Extends Flax TrainState to also keep EMA parameters.
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

