import jax
import jax.numpy as jnp
import flax.linen as nn

class BernoulliLikelihood(nn.Module):
    eps: float = 1e-8  # tránh log(0)

    @nn.compact
    def __call__(self, x, xp):
        """
        x:   ground-truth Bernoulli samples (0/1), shape (...,)
        xp:  probs dự đoán trong [0,1], shape như x
        return: log p(x | xp), cùng shape với x
        """
        xp = jnp.clip(xp, self.eps, 1.0 - self.eps)
        logpx = x * jnp.log(xp) + (1.0 - x) * jnp.log(1.0 - xp)
        return logpx

    def sample(self, xp, binary: bool = False):
        """
        Lấy mẫu từ Bernoulli(probs=xp) nếu binary=True,
        ngược lại trả lại xp (giống kiểu dùng 'xp' như kỳ vọng).

        Lưu ý: cần rng 'sample' khi gọi apply:
        model.apply(params, xp, rngs={'sample': key})
        """
        key = self.make_rng('sample')
        if binary:
            xp = jax.random.bernoulli(key, p=xp).astype(xp.dtype)
        return xp
