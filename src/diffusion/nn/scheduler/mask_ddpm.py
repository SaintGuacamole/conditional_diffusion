from typing import Optional

import safetensors.torch
import torch
from diffusers import DDPMScheduler



class MaskDDPMScheduler(DDPMScheduler):

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
        mask_noise_factor: float = 1.0,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        # Calculate alpha-scaled noise coefficient
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # Scale the (1-α) term with the provided alpha
        scaled_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps])
        sqrt_scaled_one_minus_alpha_prod = scaled_one_minus_alpha_prod ** 0.5
        sqrt_scaled_one_minus_alpha_prod = sqrt_scaled_one_minus_alpha_prod.flatten()
        while len(sqrt_scaled_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_scaled_one_minus_alpha_prod = sqrt_scaled_one_minus_alpha_prod.unsqueeze(-1)

        if mask is not None:
            masked_sqrt_one_minus_alpha = sqrt_scaled_one_minus_alpha_prod * (
                mask * mask_noise_factor + (1 - mask)
            )
            noisy_samples = (
                sqrt_alpha_prod * original_samples +
                masked_sqrt_one_minus_alpha * noise
            )
        else:
            noisy_samples = (
                sqrt_alpha_prod * original_samples +
                sqrt_scaled_one_minus_alpha_prod * noise
            )

        return noisy_samples