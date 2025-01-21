import logging
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from diffusers import SchedulerMixin
from diffusers.utils import BaseOutput

from torch import Tensor


@dataclass
class NoiseOutput(BaseOutput):
    noisy_x: Tensor
    noise: Tensor
    timesteps: Tensor


class NoiseMixin:

    noise_scheduler: SchedulerMixin
    def add_noise(
        self,
        x: Tensor,
        *,
        device: Literal["cpu", "cuda", "mps"],
        input_perturbation: Optional[float] = None,
        noise_mask: Optional[Tensor] = None,
        timesteps: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
    ) -> NoiseOutput:

        if not hasattr(self, 'noise_scheduler') or self.noise_scheduler is None:
            raise ValueError("noise_scheduler must be initialized in the parent class.")

        size = x.size()

        if noise is None:
            noise = torch.randn(size, device=device)
        else:
            assert noise.shape == size
        noise = noise * self.noise_scheduler.init_noise_sigma

        if timesteps is None:
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (size[0],), device=noise.device)
        else:
            assert timesteps.shape[0] == x.shape[0]
        timesteps = timesteps.long()

        if noise_mask is not None:
            noise = noise * noise_mask

        if input_perturbation is not None:
            new_noise = noise + input_perturbation * torch.randn_like(noise)
            noisy_x = self.noise_scheduler.add_noise(x, new_noise, timesteps)
        else:
            noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)

        noisy_x = self.noise_scheduler.scale_model_input(noisy_x, timestep=timesteps)

        return NoiseOutput(
            noisy_x=noisy_x,
            noise=noise,
            timesteps=timesteps
        )

    def create_classifier_free_guidance_inputs(
        self,
        x: Tensor,
        *,
        device: Literal["cpu", "cuda", "mps"],
        input_perturbation: Optional[float] = None,
        noise_mask: Optional[Tensor] = None,
    ) -> NoiseOutput:

        if not hasattr(self, 'noise_scheduler') or self.noise_scheduler is None:
            raise ValueError("noise_scheduler must be initialized in the parent class.")

        size = x.size()

        noise = torch.randn(size, device=device)
        noise = noise * self.noise_scheduler.init_noise_sigma

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (size[0],), device=noise.device)
        timesteps = timesteps.long()

        conditional_noise = noise
        unconditional_noise = noise

        if noise_mask is not None:
            conditional_noise = conditional_noise * noise_mask

        if input_perturbation is not None:
            perturbed_conditinoal_noise = conditional_noise + input_perturbation * torch.randn_like(conditional_noise)
            conditional_noisy_x = self.noise_scheduler.add_noise(x, perturbed_conditinoal_noise, timesteps)

            perturbed_unconditinoal_noise = conditional_noise + input_perturbation * torch.randn_like(conditional_noise)
            unconditional_noisy_x = self.noise_scheduler.add_noise(x, perturbed_unconditinoal_noise, timesteps)
        else:
            conditional_noisy_x = self.noise_scheduler.add_noise(x, conditional_noise, timesteps)
            unconditional_noisy_x = self.noise_scheduler.add_noise(x, unconditional_noise, timesteps)

        conditional_noisy_x = self.noise_scheduler.scale_model_input(conditional_noisy_x, timestep=timesteps)
        unconditional_noisy_x = self.noise_scheduler.scale_model_input(unconditional_noisy_x, timestep=timesteps)

        return NoiseOutput(
            noisy_x=torch.cat([conditional_noisy_x, unconditional_noisy_x], dim=0),
            noise=torch.cat([conditional_noise, unconditional_noise], dim=0),
            timesteps=torch.cat([timesteps, timesteps], dim=0)
        )


    def get_log_variance(self, t):
        prev_t = self.noise_scheduler.previous_timestep(t)

        alpha_prod_t = self.noise_scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.noise_scheduler.alphas_cumprod[prev_t]
        alpha_prod_t_prev = torch.where(prev_t < 0, self.noise_scheduler.one, alpha_prod_t_prev)

        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        variance = torch.clamp(variance, min=1e-20)

        log_variance = torch.log(variance)

        while len(log_variance.shape) < 4:
            log_variance = log_variance.unsqueeze(-1)

        return log_variance

    @torch.no_grad()
    def get_uncertainty(
            self,
            model_output: Tensor,
    ):
        assert model_output.size(1) % 2 == 0, f"Expected number of channels to be multiple of 2, got {model_output.size(1)}"
        _, log_variance = torch.chunk(model_output, 2, dim=1)

        # Convert log variance to standard deviation
        uncertainty = torch.exp(0.5 * log_variance)  # sqrt(variance) = std
        return uncertainty

    def get_confidence(
            self,
            *,
            model_output: Optional[Tensor] = None,
            uncertainty: Optional[Tensor] = None,
    ):
        if uncertainty is None:
            assert model_output is not None, "Either model_output or uncertainty must be provided."
            uncertainty = self.get_uncertainty(model_output)
        else:
            if model_output is not None:
                logging.warning("Warning: model_output is ignored when uncertainty is provided.")
        avg_uncertainty = uncertainty.mean(dim=[1, 2, 3])

        confidence = 1 / (1 + avg_uncertainty)
        return confidence