"""
StableVAE wrapper for JAX/Flax using diffusers FlaxAutoencoderKL.

Provides encode/decode functions to convert between pixel space and latent space.
"""

from functools import partial, cached_property

import jax
from diffusers import FlaxAutoencoderKL
from einops import rearrange
from flax import struct
from jaxtyping import Array, PyTree, Key, Float, Shaped, Int, UInt8, jaxtyped
from typeguard import typechecked

typecheck = partial(jaxtyped, typechecker=typechecked)


@struct.dataclass
class StableVAE:
    params: PyTree[Float[Array, "..."]]
    module: FlaxAutoencoderKL = struct.field(pytree_node=False)

    @classmethod
    def create(cls) -> "StableVAE":
        """
        Create StableVAE instance with pretrained weights.
        Uses pcuenq/sd-vae-ft-mse-flax model.
        """
        module, params = FlaxAutoencoderKL.from_pretrained(
            "pcuenq/sd-vae-ft-mse-flax"
        )
        params = jax.device_get(params)
        return cls(
            params=params,
            module=module,
        )

    @partial(jax.jit, static_argnames="scale")
    def encode(
        self, key: Key[Array, ""], images: Float[Array, "b h w 3"], scale: bool = True
    ) -> Float[Array, "b lh lw 4"]:
        """
        Encode images to latents.
        
        Args:
            key: JAX random key for sampling
            images: Images in HWC format, shape (b, h, w, 3), values in [-1, 1]
            scale: Whether to scale latents by VAE scaling_factor
            
        Returns:
            Latents in HWC format, shape (b, lh, lw, 4)
        """
        images = rearrange(images, "b h w c -> b c h w")
        latents = self.module.apply(
            {"params": self.params}, images, method=self.module.encode
        ).latent_dist.sample(key)
        if scale:
            latents *= self.module.config.scaling_factor
        return latents

    @partial(jax.jit, static_argnames="scale")
    def decode(
        self, latents: Float[Array, "b lh lw 4"], scale: bool = True
    ) -> Float[Array, "b h w 3"]:
        """
        Decode latents to images.
        
        Args:
            latents: Latents in HWC format, shape (b, lh, lw, 4)
            scale: Whether to unscale latents by VAE scaling_factor
            
        Returns:
            Images in HWC format, shape (b, h, w, 3), values in [-1, 1]
        """
        if scale:
            latents /= self.module.config.scaling_factor
        images = self.module.apply(
            {"params": self.params}, latents, method=self.module.decode
        ).sample
        # convert to channels-last
        images = rearrange(images, "b c h w -> b h w c")
        return images

    @cached_property
    def downscale_factor(self) -> int:
        """
        Get the downscale factor of the VAE.
        For sd-vae-ft-mse-flax, this is typically 8.
        """
        return 2 ** (len(self.module.block_out_channels) - 1)

