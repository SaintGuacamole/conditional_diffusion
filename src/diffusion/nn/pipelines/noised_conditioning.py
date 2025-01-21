from typing import Optional, Union, List

import torch
from diffusers import DiffusionPipeline, SchedulerMixin
from torch import Generator, Tensor, Size, nn

from diffusion.functional import normalize
from diffusion.functional.conditioning_functions import fast_sampling
from diffusion.models import SinogramConditionedUnet, UNetModel
from diffusion.nn.leap_projector_wrapper import SimpleProjector


class WeakNoisePipeline(DiffusionPipeline):

    def __init__(
        self,
        *,
        unet: SinogramConditionedUnet,
        scheduler: SchedulerMixin,
        projector: SimpleProjector,
        extract_into_rotations: nn.Module
    ):
        super(WeakNoisePipeline, self).__init__()
        self.projector = projector
        self.extract_into_rotations = extract_into_rotations
        self.register_modules(
            unet=unet,
            scheduler=scheduler
        )
        self.register_to_config()


    @torch.no_grad()
    def __call__(
        self,
        masked_sinogram,
        mask,
        weak_noise_factor,
        image_size: Size,
        generator: Optional[Union[Generator, List[Generator]]] = None,
        num_inference_steps: int = 50,
        memb: Optional[torch.Tensor] = None,

    ):
        self.unet.eval()
        image = torch.randn(
            image_size,
            device=masked_sinogram.device,
        )

        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            t.to(masked_sinogram.device)


            if weak_noise_factor is not None:
                noise = torch.randn_like(masked_sinogram)
                noise = noise * self.scheduler.init_noise_sigma

                strong_noise_sinogram = self.scheduler.add_noise(
                    masked_sinogram.clone(),
                    noise,
                    timesteps=t
                )
                strong_noise_sinogram = self.scheduler.scale_model_input(
                    strong_noise_sinogram,
                    timestep=t
                )

                weak_noise_sinogram = self.scheduler.add_noise(
                    masked_sinogram.clone(),
                    noise,
                    timesteps=t // weak_noise_factor
                )
                weak_noise_sinogram = self.scheduler.scale_model_input(
                    weak_noise_sinogram,
                    timestep=t // weak_noise_factor
                )

                conditioning = mask * weak_noise_sinogram + (1 - mask) * strong_noise_sinogram
            else:
                conditioning = masked_sinogram

            extracted = self.extract_into_rotations(conditioning).detach()

            model_output = self.unet(
                sample=image,
                embeddings=(t, ),
                conditioning=extracted.clone(),
                memb=memb
            ).sample

            scheduler_output = self.scheduler.step(model_output, t, image, generator=generator)
            image = scheduler_output.prev_sample

            x_0_sino = self.projector(scheduler_output.pred_original_sample)
            masked_sinogram = mask * masked_sinogram + (1-mask) * x_0_sino

        return image



class ScaledNoisePipeline(DiffusionPipeline):

    def __init__(
        self,
        *,
        unet: Union[SinogramConditionedUnet, UNetModel],
        scheduler: SchedulerMixin,
        projector: SimpleProjector,
        extract_into_rotations: nn.Module
    ):
        super(ScaledNoisePipeline, self).__init__()
        self.projector = projector
        self.extract_into_rotations = extract_into_rotations

        self.register_modules(
            unet=unet,
            scheduler=scheduler
        )
        self.register_to_config()


    @torch.no_grad()
    def __call__(
        self,
        masked_sinogram: Tensor,
        mask: Tensor,
        target_shape: Size,
        scale: float,
        generator: Optional[Union[Generator, List[Generator]]] = None,
        num_inference_steps: int = 50,
        starting_index: int = 27,
    ):
        assert 0 <= starting_index < num_inference_steps

        noise = torch.randn(
            target_shape,
            device=masked_sinogram.device,
        )

        self.scheduler.set_timesteps(num_inference_steps)

        sparse_fbp = self.projector.fbp(masked_sinogram)
        sparse_fbp = normalize(sparse_fbp.clamp(0., 1.))

        starting_timestep = self.scheduler.timesteps[starting_index-1].to(masked_sinogram.device)

        sparse_projection = self.projector(sparse_fbp)

        sparse_fbp = sparse_fbp.mul(2.).sub(1.)

        image = self.scheduler.add_noise(sparse_fbp, noise, timesteps=starting_timestep)

        conditioning = mask * masked_sinogram + (1-mask) * sparse_projection
        conditioning = self.extract_into_rotations(conditioning * scale)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            if i < starting_index:
                continue

            t.to(masked_sinogram.device)

            model_output = self.unet(
                sample=image,
                embeddings=(t, ),
                conditioning=conditioning,
            ).sample

            scheduler_output = self.scheduler.step(model_output, t, image, generator=generator)
            image = scheduler_output.prev_sample

            x_0_sino = self.projector(scheduler_output.pred_original_sample.clamp(-1., 1.).add(1.).mul(.5))
            conditioning = mask * masked_sinogram + (1-mask) * x_0_sino
            conditioning = self.extract_into_rotations(conditioning * scale)

        return image

    def dps(
        self,
        masked_sinogram: Tensor,
        mask: Tensor,
        target_shape: Size,
        scale: float,
        n_gradient_steps: int,
        generator: Optional[Union[Generator, List[Generator]]] = None,
        num_inference_steps: int = 50,
        starting_index: int = 25,
    ):
        assert 0 < starting_index <= num_inference_steps

        noise = torch.randn(
            target_shape,
            device=masked_sinogram.device,
        )

        self.scheduler.set_timesteps(num_inference_steps)

        sparse_fbp = self.projector.fbp(masked_sinogram)
        sparse_fbp = normalize(sparse_fbp.clamp(0., 1.))

        starting_timestep = self.scheduler.timesteps[starting_index-1].to(masked_sinogram.device)

        sparse_projection = self.projector(sparse_fbp)

        sparse_fbp = sparse_fbp.mul(2.).sub(1.)

        image = self.scheduler.add_noise(sparse_fbp, noise, timesteps=starting_timestep)

        conditioning = mask * masked_sinogram + (1-mask) * sparse_projection
        conditioning = self.extract_into_rotations(conditioning * scale)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):

            with torch.no_grad():
                if i < starting_index:
                    continue

                t.to(masked_sinogram.device)

                model_output = self.unet(
                    sample=image,
                    embeddings=(t, ),
                    conditioning=conditioning,
                ).sample

                scheduler_output = self.scheduler.step(model_output, t, image, generator=generator)

                x_0_pred = scheduler_output.pred_original_sample.detach().add(1.).mul(.5)

            image = fast_sampling(
                x_0_pred.detach().clone(),
                masked_sinogram,
                lr=5e-2,
                projector=self.projector,
                mask=mask,
                n_steps=n_gradient_steps
            )

            x_0_sino = self.projector(image.clone())
            conditioning = mask * masked_sinogram + (1-mask) * x_0_sino
            conditioning = self.extract_into_rotations(conditioning * scale)

            image = image.mul(2.).sub(1.)

            prev_t = self.scheduler.previous_timestep(t)
            prev_t = prev_t if prev_t > 0 else torch.tensor(0, device=prev_t.device)

            image = self.scheduler.add_noise(image, torch.randn_like(image, device=image.device), timesteps=prev_t)

        return image