# jax_fsdm/gaussian_diffusion.py

import enum
import math
from typing import Any, Dict, Tuple, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .nn import mean_flat   # JAX version
from .losses import normal_kl, discretized_gaussian_log_likelihood  # JAX version
from .bernoulli import BernoulliLikelihood  # JAX version

Array = jnp.ndarray
ModelFn = Callable[..., Array]


def get_named_beta_schedule(schedule_name: str, num_diffusion_timesteps: int) -> Array:
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
        return jnp.asarray(betas)
    elif schedule_name == "cosine":
        betas = betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
        return jnp.asarray(betas)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(
    num_diffusion_timesteps: int,
    alpha_bar: Callable[[float], float],
    max_beta: float = 0.999,
) -> np.ndarray:
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float64)


class ModelMeanType(enum.Enum):
    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()


class ModelVarType(enum.Enum):
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()

    def is_vb(self):
        return self in (LossType.KL, LossType.RESCALED_KL)


class GaussianDiffusion:
    """
    JAX version of GaussianDiffusion.
    """

    def __init__(
        self,
        *,
        betas: Array,
        model_mean_type: ModelMeanType,
        model_var_type: ModelVarType,
        loss_type: LossType,
        rescale_timesteps: bool = False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        betas = jnp.asarray(betas, dtype=jnp.float64)
        assert betas.ndim == 1, "betas must be 1-D"
        assert jnp.all(betas > 0) and jnp.all(betas <= 1.0)

        self.betas = betas
        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = jnp.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = jnp.concatenate(
            [jnp.array([1.0], dtype=jnp.float64), self.alphas_cumprod[:-1]], axis=0
        )
        self.alphas_cumprod_next = jnp.concatenate(
            [self.alphas_cumprod[1:], jnp.array([0.0], dtype=jnp.float64)], axis=0
        )

        # q(x_t | x_0)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = jnp.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = jnp.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = jnp.sqrt(1.0 / self.alphas_cumprod - 1.0)

        # q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = jnp.log(
            jnp.concatenate(
                [self.posterior_variance[1:2], self.posterior_variance[1:]], axis=0
            )
        )
        self.posterior_mean_coef1 = (
            betas * jnp.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * jnp.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.bernoulli = BernoulliLikelihood()

    # ------------------------------------------------------------------
    # q(x_t | x_0) and posterior
    # ------------------------------------------------------------------
    def q_mean_variance(self, x_start: Array, t: Array) -> Tuple[Array, Array, Array]:
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(
        self,
        rng: jax.Array,
        x_start: Array,
        t: Array,
        noise: Optional[Array] = None,
    ) -> Array:
        if noise is None:
            noise = jax.random.normal(rng, x_start.shape)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )
            * noise
        )

    def q_posterior_mean_variance(
        self, x_start: Array, x_t: Array, t: Array
    ) -> Tuple[Array, Array, Array]:
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(
            self.posterior_variance, t, x_t.shape
        )
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # ------------------------------------------------------------------
    # Helpers for eps/x0/x_{t-1}
    # ------------------------------------------------------------------
    def _predict_xstart_from_eps(self, x_t: Array, t: Array, eps: Array) -> Array:
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(
                self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
            )
            * eps
        )

    def _predict_xstart_from_xprev(
        self, x_t: Array, t: Array, xprev: Array
    ) -> Array:
        assert x_t.shape == xprev.shape
        return (
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(
        self, x_t: Array, t: Array, pred_xstart: Array
    ) -> Array:
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

    def _scale_timesteps(self, t: Array) -> Array:
        if self.rescale_timesteps:
            return t.astype(jnp.float32) * (1000.0 / self.num_timesteps)
        return t

    # ------------------------------------------------------------------
    # p(x_{t-1} | x_t)
    # ------------------------------------------------------------------
    def p_mean_variance(
        self,
        model: ModelFn,
        x: Array,
        t: Array,
        c: Optional[Any] = None,
        clip_denoised: bool = True,
        denoised_fn: Optional[Callable[[Array], Array]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Array]:
        if model_kwargs is None:
            model_kwargs = {}
        B = x.shape[0]
        assert t.shape == (B,)

        model_output = model(x, self._scale_timesteps(t), c, **model_kwargs)

        # variance
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            C = x.shape[1]
            assert model_output.shape[:2] == (B, C * 2)
            model_output, model_var_values = jnp.split(model_output, 2, axis=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = jnp.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(jnp.log(self.betas), t, x.shape)
                frac = (model_var_values + 1.0) / 2.0
                model_log_variance = frac * max_log + (1.0 - frac) * min_log
                model_variance = jnp.exp(model_log_variance)
        else:
            model_variance_arr, model_log_variance_arr = {
                ModelVarType.FIXED_LARGE: (
                    jnp.concatenate(
                        [self.posterior_variance[1:2], self.betas[1:]], axis=0
                    ),
                    jnp.log(
                        jnp.concatenate(
                            [self.posterior_variance[1:2], self.betas[1:]], axis=0
                        )
                    ),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance_arr, t, x.shape)
            model_log_variance = _extract_into_tensor(
                model_log_variance_arr, t, x.shape
            )

        def process_xstart(x0: Array) -> Array:
            if denoised_fn is not None:
                x0 = denoised_fn(x0)
            if clip_denoised:
                return jnp.clip(x0, -1.0, 1.0)
            return x0

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    # ------------------------------------------------------------------
    # Sampling DDPM / DDIM
    # ------------------------------------------------------------------
    def p_sample(
        self,
        rng: jax.Array,
        model: ModelFn,
        x: Array,
        t: Array,
        c: Optional[Any] = None,
        clip_denoised: bool = True,
        denoised_fn: Optional[Callable[[Array], Array]] = None,
        cond_fn: Optional[Callable[..., Array]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Array]:
        out = self.p_mean_variance(
            model,
            x,
            t,
            c,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = jax.random.normal(rng, x.shape)
        nonzero_mask = (t != 0).astype(jnp.float32).reshape(
            (-1,) + (1,) * (x.ndim - 1)
        )
        if cond_fn is not None:
            gradient = cond_fn(x, self._scale_timesteps(t), **(model_kwargs or {}))
            out_mean = out["mean"].astype(jnp.float32) + out["variance"] * gradient.astype(
                jnp.float32
            )
        else:
            out_mean = out["mean"]
        sample = out_mean + nonzero_mask * jnp.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        rng: jax.Array,
        model: ModelFn,
        shape: Tuple[int, ...],
        c: Optional[Any] = None,
        noise: Optional[Array] = None,
        clip_denoised: bool = True,
        denoised_fn: Optional[Callable[[Array], Array]] = None,
        cond_fn: Optional[Callable[..., Array]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Array:
        if noise is not None:
            img = noise
        else:
            rng, sub = jax.random.split(rng)
            img = jax.random.normal(sub, shape)

        for i in reversed(range(self.num_timesteps)):
            rng, sub = jax.random.split(rng)
            t = jnp.full((shape[0],), i, dtype=jnp.int32)
            out = self.p_sample(
                sub,
                model,
                img,
                t,
                c=c,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
            )
            img = out["sample"]
        return img

    def ddim_sample(
        self,
        rng: jax.Array,
        model: ModelFn,
        x: Array,
        t: Array,
        c: Optional[Any] = None,
        clip_denoised: bool = True,
        denoised_fn: Optional[Callable[[Array], Array]] = None,
        cond_fn: Optional[Callable[..., Array]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        eta: float = 0.0,
    ) -> Dict[str, Array]:
        out = self.p_mean_variance(
            model,
            x,
            t,
            c,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
            eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
            eps = eps - jnp.sqrt(1.0 - alpha_bar) * cond_fn(
                x, self._scale_timesteps(t), **(model_kwargs or {})
            )
            out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
            out["mean"], _, _ = self.q_posterior_mean_variance(
                x_start=out["pred_xstart"], x_t=x, t=t
            )

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * jnp.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar))
            * jnp.sqrt(1.0 - alpha_bar / alpha_bar_prev)
        )
        rng, sub = jax.random.split(rng)
        noise = jax.random.normal(sub, x.shape)
        mean_pred = (
            out["pred_xstart"] * jnp.sqrt(alpha_bar_prev)
            + jnp.sqrt(1.0 - alpha_bar_prev - sigma**2) * eps
        )
        nonzero_mask = (t != 0).astype(jnp.float32).reshape(
            (-1,) + (1,) * (x.ndim - 1)
        )
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        rng: jax.Array,
        model: ModelFn,
        shape: Tuple[int, ...],
        c: Optional[Any] = None,
        noise: Optional[Array] = None,
        clip_denoised: bool = True,
        denoised_fn: Optional[Callable[[Array], Array]] = None,
        cond_fn: Optional[Callable[..., Array]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        eta: float = 0.0,
        ddim_num_steps: Optional[int] = None,
    ) -> Array:
        """
        DDIM sampling loop with optional timestep respacing.
        
        Args:
            ddim_num_steps: Number of DDIM sampling steps. If None, use all timesteps.
                           Typical values: 50, 100, 250 for faster sampling.
        """
        if noise is not None:
            img = noise
        else:
            rng, sub = jax.random.split(rng)
            img = jax.random.normal(sub, shape)

        # Get timesteps to use (with respacing if ddim_num_steps is specified)
        timesteps = self._get_ddim_timesteps(ddim_num_steps)

        for i in timesteps:
            rng, sub = jax.random.split(rng)
            t = jnp.full((shape[0],), int(i), dtype=jnp.int32)
            out = self.ddim_sample(
                sub,
                model,
                img,
                t,
                c=c,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                eta=eta,
            )
            img = out["sample"]
        return img

    # ------------------------------------------------------------------
    # Training losses
    # ------------------------------------------------------------------
    def _vb_terms_bpd(
        self,
        model: ModelFn,
        x_start: Array,
        x_t: Array,
        t: Array,
        c: Optional[Any] = None,
        clip_denoised: bool = True,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Array]:
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, c, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / jnp.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        decoder_nll = mean_flat(decoder_nll) / jnp.log(2.0)
        output = jnp.where(t == 0, decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(
        self,
        rng: jax.Array,
        model: ModelFn,
        x_start: Array,
        t: Array,
        c: Optional[Any] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        noise: Optional[Array] = None,
    ) -> Dict[str, Array]:
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            rng, sub = jax.random.split(rng)
            noise = jax.random.normal(sub, x_start.shape)

        x_t = self.q_sample(rng, x_start, t, noise=noise)
        terms: Dict[str, Array] = {}

        if self.loss_type in [LossType.KL, LossType.RESCALED_KL]:
            vb_out = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                c=c,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                vb_out = vb_out * self.num_timesteps
            terms["loss"] = vb_out

        elif self.loss_type in [LossType.MSE, LossType.RESCALED_MSE]:
            model_output = model(x_t, self._scale_timesteps(t), c, **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = jnp.split(model_output, 2, axis=1)

                frozen_out = jnp.concatenate(
                    [jax.lax.stop_gradient(model_output), model_var_values], axis=1
                )

                def frozen_model(x_in, t_in, c_in, **kw):
                    return frozen_out

                vb = self._vb_terms_bpd(
                    model=frozen_model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    c=c,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    vb = vb * (self.num_timesteps / 1000.0)
                terms["vb"] = vb

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]

            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms


def _extract_into_tensor(arr: Array, timesteps: Array, broadcast_shape: Tuple[int, ...]) -> Array:
    arr = jnp.asarray(arr)
    timesteps = timesteps.astype(jnp.int32)
    res = arr[timesteps]
    while res.ndim < len(broadcast_shape):
        res = res[..., None]
    return jnp.broadcast_to(res, broadcast_shape)
# jax_fsdm/gaussian_diffusion.py

import enum
import math
from typing import Any, Dict, Tuple, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np  # chỉ dùng cho tiện tạo array, sau đó convert sang jnp

from .nn import mean_flat   # JAX version bạn đã port
from .losses import normal_kl, discretized_gaussian_log_likelihood  # JAX version
from .bernoulli import BernoulliLikelihood  # JAX version

Array = jnp.ndarray
ModelFn = Callable[..., Array]


def get_named_beta_schedule(schedule_name: str, num_diffusion_timesteps: int) -> Array:
    """
    Get a pre-defined beta schedule for the given name.
    Returns: jnp.ndarray shape [T]
    """
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
        return jnp.asarray(betas)
    elif schedule_name == "cosine":
        betas = betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
        return jnp.asarray(betas)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(
    num_diffusion_timesteps: int,
    alpha_bar: Callable[[float], float],
    max_beta: float = 0.999,
) -> np.ndarray:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function.
    Returns: numpy array (sau đó convert sang jnp phía trên).
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float64)


class ModelMeanType(enum.Enum):
    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()     # the model predicts x_0
    EPSILON = enum.auto()     # the model predicts epsilon


class ModelVarType(enum.Enum):
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()

    def is_vb(self):
        return self in (LossType.KL, LossType.RESCALED_KL)


class GaussianDiffusion:
    """
    JAX version of GaussianDiffusion.

    This class is framework-agnostic: it only uses jnp and expects a Python
    callable `model(x, t_scaled, c, **model_kwargs)` that returns a tensor
    with the same shape as x.
    """

    def __init__(
        self,
        *,
        betas: Array,
        model_mean_type: ModelMeanType,
        model_var_type: ModelVarType,
        loss_type: LossType,
        rescale_timesteps: bool = False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        betas = jnp.asarray(betas, dtype=jnp.float64)
        assert betas.ndim == 1, "betas must be 1-D"
        assert jnp.all(betas > 0) and jnp.all(betas <= 1.0)

        self.betas = betas
        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = jnp.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = jnp.concatenate(
            [jnp.array([1.0], dtype=jnp.float64), self.alphas_cumprod[:-1]], axis=0
        )
        self.alphas_cumprod_next = jnp.concatenate(
            [self.alphas_cumprod[1:], jnp.array([0.0], dtype=jnp.float64)], axis=0
        )

        # q(x_t | x_0) stuff
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = jnp.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = jnp.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = jnp.sqrt(1.0 / self.alphas_cumprod - 1.0)

        # q(x_{t-1} | x_t, x_0) stuff
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = jnp.log(
            jnp.concatenate(
                [self.posterior_variance[1:2], self.posterior_variance[1:]], axis=0
            )
        )
        self.posterior_mean_coef1 = (
            betas * jnp.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * jnp.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.bernoulli = BernoulliLikelihood()

    # ----------------------------------------------------------------------
    # q(x_t | x_0) and posterior
    # ----------------------------------------------------------------------

    def q_mean_variance(self, x_start: Array, t: Array) -> Tuple[Array, Array, Array]:
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(
        self,
        rng: jax.Array,
        x_start: Array,
        t: Array,
        noise: Optional[Array] = None,
    ) -> Array:
        """
        Sample from q(x_t | x_0).
        rng only used if noise is None.
        """
        if noise is None:
            noise = jax.random.normal(rng, x_start.shape)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )
            * noise
        )

    def q_posterior_mean_variance(
        self, x_start: Array, x_t: Array, t: Array
    ) -> Tuple[Array, Array, Array]:
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(
            self.posterior_variance, t, x_t.shape
        )
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # ----------------------------------------------------------------------
    # Helpers for converting between eps/x0/x_{t-1}
    # ----------------------------------------------------------------------

    def _predict_xstart_from_eps(self, x_t: Array, t: Array, eps: Array) -> Array:
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(
                self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
            )
            * eps
        )

    def _predict_xstart_from_xprev(
        self, x_t: Array, t: Array, xprev: Array
    ) -> Array:
        assert x_t.shape == xprev.shape
        return (
            _extract_into_tensor(
                1.0 / self.posterior_mean_coef1, t, x_t.shape
            )
            * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1,
                t,
                x_t.shape,
            )
            * x_t
        )

    def _predict_eps_from_xstart(
        self, x_t: Array, t: Array, pred_xstart: Array
    ) -> Array:
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

    def _scale_timesteps(self, t: Array) -> Array:
        if self.rescale_timesteps:
            return t.astype(jnp.float32) * (1000.0 / self.num_timesteps)
        return t

    # ----------------------------------------------------------------------
    # p(x_{t-1} | x_t) core
    # ----------------------------------------------------------------------

    def p_mean_variance(
        self,
        model: ModelFn,
        x: Array,
        t: Array,
        c: Optional[Any] = None,
        clip_denoised: bool = True,
        denoised_fn: Optional[Callable[[Array], Array]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Array]:
        if model_kwargs is None:
            model_kwargs = {}

        B = x.shape[0]
        assert t.shape == (B,)

        model_output = model(x, self._scale_timesteps(t), c, **model_kwargs)

        # variance
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            C = x.shape[1]
            assert model_output.shape[:2] == (B, C * 2)
            model_output, model_var_values = jnp.split(model_output, 2, axis=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = jnp.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(jnp.log(self.betas), t, x.shape)
                frac = (model_var_values + 1.0) / 2.0
                model_log_variance = frac * max_log + (1.0 - frac) * min_log
                model_variance = jnp.exp(model_log_variance)
        else:
            model_variance_arr, model_log_variance_arr = {
                ModelVarType.FIXED_LARGE: (
                    jnp.concatenate(
                        [self.posterior_variance[1:2], self.betas[1:]], axis=0
                    ),
                    jnp.log(
                        jnp.concatenate(
                            [self.posterior_variance[1:2], self.betas[1:]], axis=0
                        )
                    ),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance_arr, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance_arr, t, x.shape)

        def process_xstart(x0: Array) -> Array:
            if denoised_fn is not None:
                x0 = denoised_fn(x0)
            if clip_denoised:
                return jnp.clip(x0, -1.0, 1.0)
            return x0

        # mean & pred_xstart
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    # ----------------------------------------------------------------------
    # Sampling (DDPM)
    # ----------------------------------------------------------------------

    def p_sample(
        self,
        rng: jax.Array,
        model: ModelFn,
        x: Array,
        t: Array,
        c: Optional[Any] = None,
        clip_denoised: bool = True,
        denoised_fn: Optional[Callable[[Array], Array]] = None,
        cond_fn: Optional[Callable[..., Array]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Array]:
        out = self.p_mean_variance(
            model,
            x,
            t,
            c,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = jax.random.normal(rng, x.shape)
        nonzero_mask = (t != 0).astype(jnp.float32).reshape(
            (-1,) + (1,) * (x.ndim - 1)
        )
        if cond_fn is not None:
            # condition_mean strategy
            gradient = cond_fn(x, self._scale_timesteps(t), **(model_kwargs or {}))
            out_mean = out["mean"].astype(jnp.float32) + out["variance"] * gradient.astype(
                jnp.float32
            )
        else:
            out_mean = out["mean"]
        sample = out_mean + nonzero_mask * jnp.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        rng: jax.Array,
        model: ModelFn,
        shape: Tuple[int, ...],
        c: Optional[Any] = None,
        noise: Optional[Array] = None,
        clip_denoised: bool = True,
        denoised_fn: Optional[Callable[[Array], Array]] = None,
        cond_fn: Optional[Callable[..., Array]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Array:
        """
        Non-jitted sampling loop in pure Python. For JIT, wrap with lax.scan.
        """
        if noise is not None:
            img = noise
        else:
            rng, sub = jax.random.split(rng)
            img = jax.random.normal(sub, shape)

        for i in reversed(range(self.num_timesteps)):
            rng, sub = jax.random.split(rng)
            t = jnp.full((shape[0],), i, dtype=jnp.int32)
            out = self.p_sample(
                sub,
                model,
                img,
                t,
                c=c,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
            )
            img = out["sample"]
        return img

    # ----------------------------------------------------------------------
    # DDIM sampling (deterministic when eta=0)
    # ----------------------------------------------------------------------

    def _get_ddim_timesteps(self, ddim_num_steps: Optional[int] = None) -> Array:
        """
        Get timesteps for DDIM sampling with respacing.
        
        Args:
            ddim_num_steps: Number of DDIM sampling steps. If None, use all timesteps.
        
        Returns:
            Array of timestep indices to use (in reverse order, from num_timesteps-1 to 0)
        """
        if ddim_num_steps is None or ddim_num_steps >= self.num_timesteps:
            # Use all timesteps
            return jnp.arange(self.num_timesteps - 1, -1, -1)
        
        # Calculate stride to get exactly ddim_num_steps
        # We want to sample from [0, num_timesteps-1] with ddim_num_steps steps
        # Stride = (num_timesteps - 1) / (ddim_num_steps - 1)
        if ddim_num_steps == 1:
            # Only sample at the last timestep
            return jnp.array([self.num_timesteps - 1])
        
        # Calculate evenly spaced timesteps in reverse order
        # Start from num_timesteps-1, go down to 0
        stride = (self.num_timesteps - 1) / (ddim_num_steps - 1)
        timesteps = []
        for i in range(ddim_num_steps):
            t = round(i * stride)
            # Convert to reverse order (from num_timesteps-1 down to 0)
            timestep = self.num_timesteps - 1 - t
            timesteps.append(timestep)
        
        # Ensure we have exactly ddim_num_steps unique timesteps
        timesteps = sorted(set(timesteps), reverse=True)
        if len(timesteps) != ddim_num_steps:
            # Fallback: use evenly spaced indices
            indices = np.linspace(0, self.num_timesteps - 1, ddim_num_steps, dtype=int)
            timesteps = (self.num_timesteps - 1 - indices).tolist()
            timesteps = sorted(set(timesteps), reverse=True)
        
        return jnp.array(timesteps)

    def ddim_sample(
        self,
        rng: jax.Array,
        model: ModelFn,
        x: Array,
        t: Array,
        c: Optional[Any] = None,
        clip_denoised: bool = True,
        denoised_fn: Optional[Callable[[Array], Array]] = None,
        cond_fn: Optional[Callable[..., Array]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        eta: float = 0.0,
    ) -> Dict[str, Array]:
        out = self.p_mean_variance(
            model,
            x,
            t,
            c,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            # condition_score (Song et al.)
            alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
            eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
            eps = eps - jnp.sqrt(1.0 - alpha_bar) * cond_fn(
                x, self._scale_timesteps(t), **(model_kwargs or {})
            )
            out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
            out["mean"], _, _ = self.q_posterior_mean_variance(
                x_start=out["pred_xstart"], x_t=x, t=t
            )

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * jnp.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar))
            * jnp.sqrt(1.0 - alpha_bar / alpha_bar_prev)
        )
        rng, sub = jax.random.split(rng)
        noise = jax.random.normal(sub, x.shape)
        mean_pred = (
            out["pred_xstart"] * jnp.sqrt(alpha_bar_prev)
            + jnp.sqrt(1.0 - alpha_bar_prev - sigma**2) * eps
        )
        nonzero_mask = (t != 0).astype(jnp.float32).reshape(
            (-1,) + (1,) * (x.ndim - 1)
        )
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        rng: jax.Array,
        model: ModelFn,
        shape: Tuple[int, ...],
        c: Optional[Any] = None,
        noise: Optional[Array] = None,
        clip_denoised: bool = True,
        denoised_fn: Optional[Callable[[Array], Array]] = None,
        cond_fn: Optional[Callable[..., Array]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        eta: float = 0.0,
        ddim_num_steps: Optional[int] = None,
    ) -> Array:
        """
        DDIM sampling loop with optional timestep respacing.
        
        Args:
            ddim_num_steps: Number of DDIM sampling steps. If None, use all timesteps.
                           Typical values: 50, 100, 250 for faster sampling.
        """
        if noise is not None:
            img = noise
        else:
            rng, sub = jax.random.split(rng)
            img = jax.random.normal(sub, shape)

        # Get timesteps to use (with respacing if ddim_num_steps is specified)
        timesteps = self._get_ddim_timesteps(ddim_num_steps)

        for i in timesteps:
            rng, sub = jax.random.split(rng)
            t = jnp.full((shape[0],), int(i), dtype=jnp.int32)
            out = self.ddim_sample(
                sub,
                model,
                img,
                t,
                c=c,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                eta=eta,
            )
            img = out["sample"]
        return img

    # ----------------------------------------------------------------------
    # Training losses
    # ----------------------------------------------------------------------

    def _vb_terms_bpd(
        self,
        model: ModelFn,
        x_start: Array,
        x_t: Array,
        t: Array,
        c: Optional[Any] = None,
        clip_denoised: bool = True,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Array]:
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, c, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / jnp.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        decoder_nll = mean_flat(decoder_nll) / jnp.log(2.0)
        output = jnp.where(t == 0, decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(
        self,
        rng: jax.Array,
        model: ModelFn,
        x_start: Array,
        t: Array,
        c: Optional[Any] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        noise: Optional[Array] = None,
    ) -> Dict[str, Array]:
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            rng, sub = jax.random.split(rng)
            noise = jax.random.normal(sub, x_start.shape)

        x_t = self.q_sample(rng, x_start, t, noise=noise)
        terms: Dict[str, Array] = {}

        if self.loss_type in [LossType.KL, LossType.RESCALED_KL]:
            vb_out = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                c=c,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                vb_out = vb_out * self.num_timesteps
            terms["loss"] = vb_out

        elif self.loss_type in [LossType.MSE, LossType.RESCALED_MSE]:
            model_output = model(x_t, self._scale_timesteps(t), c, **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = jnp.split(model_output, 2, axis=1)

                # VB term for variance (frozen mean)
                frozen_out = jnp.concatenate(
                    [jax.lax.stop_gradient(model_output), model_var_values], axis=1
                )

                def frozen_model(x_in, t_in, c_in, **kw):
                    # ignore real input, just return frozen_out broadcasted
                    # NOTE: assume x_in.shape == model_output.shape
                    return frozen_out

                vb = self._vb_terms_bpd(
                    model=frozen_model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    c=c,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    vb = vb * (self.num_timesteps / 1000.0)
                terms["vb"] = vb

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]

            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms


def _extract_into_tensor(arr: Array, timesteps: Array, broadcast_shape: Tuple[int, ...]) -> Array:
    """
    Extract values from a 1-D array for a batch of indices and broadcast.
    arr: [T]
    timesteps: [B] int32
    broadcast_shape: shape of result
    """
    arr = jnp.asarray(arr)
    timesteps = timesteps.astype(jnp.int32)
    res = arr[timesteps]  # [B]
    while res.ndim < len(broadcast_shape):
        res = res[..., None]
    return jnp.broadcast_to(res, broadcast_shape)
