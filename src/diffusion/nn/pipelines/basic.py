from typing import Optional, Union, List, Literal

import torch
from diffusers import DiffusionPipeline, SchedulerMixin
from torch import Generator, Size, Tensor

from diffusion.functional.conditioning_functions import fast_sampling, diffusion_posterior_sampling
from diffusion.models import UNetModel
from diffusion.nn.leap_projector_wrapper import SimpleProjector


class BasicPipeline(DiffusionPipeline):

    def __init__(
            self,
            *,
            unet: UNetModel,
            scheduler: SchedulerMixin
    ):
        super(BasicPipeline, self).__init__()
        self.register_modules(
            unet=unet,
            scheduler=scheduler
        )
        self.set_progress_bar_config(disable=True)
        self.register_to_config()



    @torch.no_grad()
    def __call__(
        self,
        target_shape: Size,
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        generator: Optional[Union[Generator, List[Generator]]] = None,
        num_inference_steps: int = 50,
    ):
        self.unet.eval()
        image = torch.randn(
            target_shape,
            device=device,
            generator=generator,
        )

        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            t.to(device)

            model_output = self.unet(sample=image, embeddings=(t, )).sample

            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        return image


    @torch.no_grad()
    def partial(
        self,
        *,
        fbp: Tensor,
        target_shape: Optional[Size] = None,
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        generator: Optional[Union[Generator, List[Generator]]] = None,
        num_inference_steps: int = 50,
        starting_index: int = 25,
    ):
        assert target_shape is None or target_shape == fbp.size()
        assert 0 <= starting_index < num_inference_steps

        self.unet.eval()
        noise = torch.randn(
            fbp.size(),
            device=device,
            generator=generator,
        )

        self.scheduler.set_timesteps(num_inference_steps)
        starting_timestep = self.scheduler.timesteps[starting_index].to(device)

        image = self.scheduler.add_noise(fbp, noise, timesteps=starting_timestep)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            if i < starting_index:
                continue
            t.to(device)

            model_output = self.unet(image, embeddings=(t, )).sample

            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        return image


    def consistency_guidance_partial(
        self,
        *,
        fbp: Tensor,
        masked_sinogram: Tensor,
        mask: Tensor,
        projector: SimpleProjector,
        target_shape: Optional[Size] = None,
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        generator: Optional[Union[Generator, List[Generator]]] = None,
        num_inference_steps: int = 50,
        n_gradient_steps: int = 20,
        starting_index: int = 25,
        renoise_with_pred: bool = True,
    ):
        assert target_shape is None or target_shape == fbp.size()
        assert 0 <= starting_index < num_inference_steps

        noise = torch.randn(
            target_shape,
            device=device,
            generator=generator
        )

        self.scheduler.set_timesteps(num_inference_steps)
        starting_timestep = self.scheduler.timesteps[starting_index].to(device)

        image = self.scheduler.add_noise(fbp, noise, timesteps=starting_timestep)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            if i < starting_index:
                continue
            t.to(device)

            with torch.no_grad():
                model_output = self.unet(
                    sample=image,
                    embeddings=(t, )
                ).sample
                if model_output.shape[1] == image.shape[1] * 2 and self.scheduler.variance_type in ["learned",
                                                                                                    "learned_range"]:
                    eps_hat, _ = torch.split(model_output, image.shape[1], dim=1)
                else:
                    eps_hat = model_output
                scheduler_output = self.scheduler.step(model_output, t, image, generator=generator)



            image = fast_sampling(
                scheduler_output.pred_original_sample.detach().add(1.).mul(.5),
                masked_sinogram,
                lr=5e-2,
                projector=projector,
                mask=mask,
                n_steps=n_gradient_steps
            ).mul(2.).sub(1.)

            prev_t = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
            prev_t = prev_t if prev_t > 0 else torch.tensor(0, device=prev_t.device)

            if renoise_with_pred:
                image = self.scheduler.add_noise(image, eps_hat, timesteps=prev_t)
            else:
                image = self.scheduler.add_noise(image, torch.randn_like(image, device=image.device), timesteps=prev_t)

        return image

    def consistency_guidance(
        self,
        target_shape: Size,
        masked_sinogram: Tensor,
        mask: Tensor,
        projector: SimpleProjector,
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        generator: Optional[Union[Generator, List[Generator]]] = None,
        num_inference_steps: int = 50,
        n_gradient_steps: int = 20,
        renoise_with_pred: bool = True,
    ):
        image = torch.randn(
            target_shape,
            device=device,
            generator=generator
        )

        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            t.to(device)

            with torch.no_grad():
                model_output = self.unet(
                    sample=image,
                    embeddings=(t, )
                ).sample
                if model_output.shape[1] == image.shape[1] * 2 and self.scheduler.variance_type in ["learned",
                                                                                                    "learned_range"]:
                    eps_hat, _ = torch.split(model_output, image.shape[1], dim=1)
                else:
                    eps_hat = model_output
                scheduler_output = self.scheduler.step(model_output, t, image, generator=generator)
                scheduler_output = self.scheduler.step(model_output, t, image, generator=generator)

            image = fast_sampling(
                scheduler_output.pred_original_sample.detach().add(1.).mul(.5),
                masked_sinogram,
                lr=5e-2,
                projector=projector,
                mask=mask,
                n_steps=n_gradient_steps
            ).mul(2.).sub(1.)

            prev_t = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
            prev_t = prev_t if prev_t > 0 else torch.tensor(0, device=prev_t.device)

            if renoise_with_pred:
                image = self.scheduler.add_noise(image, eps_hat, timesteps=prev_t)
            else:
                image = self.scheduler.add_noise(image, torch.randn_like(image, device=image.device), timesteps=prev_t)

        return image


    def vanilla_dps(
        self,
        target_shape: Size,
        masked_sinogram: Tensor,
        mask: Tensor,
        projector: SimpleProjector,
        generator: Optional[Union[Generator, List[Generator]]] = None,
        num_inference_steps: int = 1000,
        device: Literal["cpu", "cuda", "mps"] = "cpu",
    ):
        self.unet.eval()

        image = torch.randn(
            target_shape,
            device=masked_sinogram.device,
            generator=generator,
        )

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            t.to(masked_sinogram.device)


            with torch.no_grad():
                model_output = self.unet(
                    sample=image,
                    embeddings=(t, ),
                ).sample

            image = image.requires_grad_(True)

            scheduler_output = self.scheduler.step(model_output, t, image, generator=generator)

            image, distance = diffusion_posterior_sampling(
                x_t_minus_one=scheduler_output.prev_sample.add(1.).mul(.5),
                measurement=masked_sinogram,
                measurement_mask=mask,
                x_t=image,
                x_0_hat=scheduler_output.pred_original_sample.add(1.).mul(.5),
                projector=projector
            )
            image = image.mul(2.).sub(1.).detach()

        return image