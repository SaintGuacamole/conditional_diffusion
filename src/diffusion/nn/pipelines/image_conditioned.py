from typing import Optional, Union, List, Callable

import numpy as np
import torch
from diffusers import DiffusionPipeline, SchedulerMixin
from torch import Generator, Tensor, Size

from diffusion.functional import compute_guidance
from diffusion.functional.conditioning_functions import alternating_proximal_gradient_method, fast_sampling, \
    diffusion_posterior_sampling
from diffusion.models import UNetModel, SinogramConditionedUnet
from diffusion.nn.leap_projector_wrapper import SimpleProjector


class ImageConditionedPipeline(DiffusionPipeline):

    def __init__(
            self,
            *,
            unet: Union[UNetModel, SinogramConditionedUnet],
            scheduler: SchedulerMixin
    ):
        super(ImageConditionedPipeline, self).__init__()
        self.register_modules(
            unet=unet,
            scheduler=scheduler
        )
        self.set_progress_bar_config(disable=True)
        self.register_to_config()


    @torch.no_grad()
    def __call__(
        self,
        conditioning,
        target_shape: Size,
        generator: Optional[Union[Generator, List[Generator]]] = None,
        num_inference_steps: int = 50,
        guidance_fn: Optional[Callable] = None,
        guidance_input_prep_fn: Optional[Callable] = None,
        guidance_scale: float = 1.,
        unconditional_input: Optional[Tensor] = None,
        trajectories: Optional[List] = None,
    ):
        assert (guidance_fn is None) == (guidance_input_prep_fn is None)

        self.unet.eval()

        image = torch.randn(
            target_shape,
            device=conditioning.device,
        )

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            scheduler_output = self.__step(
                image=image,
                conditioning=conditioning,
                t=t,
                device=conditioning.device,
                generator=generator,
                guidance_fn=guidance_fn,
                guidance_input_prep_fn=guidance_input_prep_fn,
                guidance_scale=guidance_scale,
                unconditional_input=unconditional_input,
            )
            image = scheduler_output.prev_sample

            if trajectories is not None:
                trajectories.append(scheduler_output.pred_original_sample.clone().detach().add(1.).mul(.5).cpu())

        return image

    @torch.no_grad()
    def partial(
        self,
        conditioning,
        target_shape: Size,
        partial_input: Tensor,
        partial_starting_index: int = 25,
        generator: Optional[Union[Generator, List[Generator]]] = None,
        num_inference_steps: int = 50,
        guidance_fn: Optional[Callable] = None,
        guidance_input_prep_fn: Optional[Callable] = None,
        guidance_scale: float = 1.,
        unconditional_input: Optional[Tensor] = None,
        trajectories: Optional[List] = None,
    ):
        assert (guidance_fn is None) == (guidance_input_prep_fn is None)

        self.unet.eval()

        noise = torch.randn(
            target_shape,
            device=conditioning.device,
            generator=generator,
        )

        self.scheduler.set_timesteps(num_inference_steps)
        starting_timestep = self.scheduler.timesteps[partial_starting_index].to(conditioning.device)

        image = self.scheduler.add_noise(partial_input, noise, timesteps=starting_timestep)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            if i < partial_starting_index:
                continue
            scheduler_output = self.__step(
                image=image,
                conditioning=conditioning,
                t=t,
                device=conditioning.device,
                generator=generator,
                guidance_fn=guidance_fn,
                guidance_input_prep_fn=guidance_input_prep_fn,
                guidance_scale=guidance_scale,
                unconditional_input=unconditional_input,
            )
            image = scheduler_output.prev_sample

            if trajectories is not None:
                trajectories.append(scheduler_output.pred_original_sample.clone().detach().add(1.).mul(.5).cpu())

        return image

    def noisy_classifier_sampling(
        self,
        conditioning,
        target_shape: Size,
        masked_sinogram: Tensor,
        mask: Tensor,
        projector: SimpleProjector,
        generator: Optional[Union[Generator, List[Generator]]] = None,
        num_inference_steps: int = 50,
        guidance_fn: Optional[Callable] = None,
        guidance_input_prep_fn: Optional[Callable] = None,
        guidance_scale: float = 1.,
        unconditional_input: Optional[Tensor] = None,
    ):
        # prox solver apgm from DOLCE
        # https://github.com/wustl-cig/DOLCE/blob/main/dataFidelities/CTClass.py
        assert (guidance_fn is None) == (guidance_input_prep_fn is None)

        self.unet.eval()

        image = torch.randn(
            target_shape,
            device=conditioning.device,
        )

        self.scheduler.set_timesteps(num_inference_steps)

        indices = list(range(num_inference_steps))[::-1]
        linespace_ = np.linspace(1, len(indices), len(indices), endpoint=True)[::-1]
        rhos = list(map(lambda x: 1 - np.exp(-x / (8 * len(indices))), linespace_))
        rhos = torch.tensor(rhos).to(conditioning.device)

        masked_sinogram = masked_sinogram.detach().clone().mul(2.).sub(1.)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            image = image.requires_grad_(True)
            scheduler_output = self.__step(
                image=image,
                conditioning=conditioning,
                t=t,
                device=conditioning.device,
                generator=generator,
                guidance_fn=guidance_fn,
                guidance_input_prep_fn=guidance_input_prep_fn,
                guidance_scale=guidance_scale,
                unconditional_input=unconditional_input
            )
            noisy_prox_target = self.scheduler.add_noise(
                masked_sinogram.clone(),
                torch.randn_like(masked_sinogram, device=masked_sinogram.device),
                t,
            )

            image = alternating_proximal_gradient_method(
                x=scheduler_output.prev_sample,
                target=noisy_prox_target,
                mask=mask,
                rho=rhos[i],
                device=masked_sinogram.device,
                projector=projector,
            ).detach()

        return image


    def __step(
        self,
        image: Tensor,
        conditioning: Tensor,
        t: Tensor,
        device: torch.device,
        generator: Optional[Union[Generator, List[Generator]]] = None,
        guidance_fn: Optional[Callable] = None,
        guidance_input_prep_fn: Optional[Callable] = None,
        guidance_scale: float = 1.,
        unconditional_input: Optional[Tensor] = None,
        forward_grad: bool = False
    ):
        t.to(device)
        if guidance_fn is not None and guidance_input_prep_fn is not None:
            conditioning_input = guidance_input_prep_fn(
                conditioning, unconditional_input
            )
            image_input = guidance_input_prep_fn(image, image)
        else:
            conditioning_input = conditioning
            image_input = image

        if forward_grad:
            model_output = self.unet(
                sample=image_input,
                conditioning=conditioning_input,
                embeddings=(t,),
            ).sample
        else:
            with torch.no_grad():
                model_output = self.unet(
                    sample=image_input,
                    conditioning=conditioning_input,
                    embeddings=(t,),
                ).sample

        if guidance_fn is not None and guidance_input_prep_fn is not None:
            model_output = compute_guidance(model_output, guidance_fn, guidance_scale)

        scheduler_output = self.scheduler.step(model_output, t, image, generator=generator)

        if model_output.shape[1] == image.shape[1] * 2 and self.scheduler.variance_type in ["learned", "learned_range"]:
            model_output, _ = torch.split(model_output, image.shape[1], dim=1)
        scheduler_output['eps_hat'] = model_output

        return scheduler_output

    def eps_bar(
        self,
        x_t: Tensor,
        x_0_hat: Tensor,
        t
    ):
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        return (x_t - x_0_hat * alpha_prod_t ** 0.5) / (1 - alpha_prod_t) ** 0.5

    def consistency_guidance(
        self,
        conditioning,
        target_shape: Size,
        masked_sinogram: Tensor,
        mask: Tensor,
        projector: SimpleProjector,
        do_dps: Optional[Union[Tensor, List]] = None,
        n_gradient_steps: int = 20,
        generator: Optional[Union[Generator, List[Generator]]] = None,
        num_inference_steps: int = 50,
        guidance_fn: Optional[Callable] = None,
        guidance_input_prep_fn: Optional[Callable] = None,
        guidance_scale: float = 1.,
        unconditional_input: Optional[Tensor] = None,
        trajectories: Optional[List] = None,
        renoise_with_pred: bool = True,
    ):
        assert (guidance_fn is None) == (guidance_input_prep_fn is None)

        if do_dps is None:
            do_dps = torch.ones((num_inference_steps, ), device=conditioning.device)
        elif not isinstance(do_dps, Tensor):
            do_dps = torch.tensor(do_dps, dtype=torch.bool)
        assert do_dps.shape[0] == num_inference_steps

        self.unet.eval()

        image = torch.randn(
            target_shape,
            device=conditioning.device,
            generator=generator,
        )

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            with torch.no_grad():
                scheduler_output = self.__step(
                    image=image,
                    conditioning=conditioning,
                    t=t,
                    device=conditioning.device,
                    generator=generator,
                    guidance_fn=guidance_fn,
                    guidance_input_prep_fn=guidance_input_prep_fn,
                    guidance_scale=guidance_scale,
                    unconditional_input=unconditional_input
                )

            if do_dps[i]:
                x_t = image
                x_0_hat = scheduler_output.pred_original_sample.clone().detach()
                image = fast_sampling(
                    scheduler_output.pred_original_sample.detach().add(1.).mul(.5),
                    masked_sinogram,
                    lr=5e-2,
                    projector=projector,
                    mask=mask,
                    n_steps=n_gradient_steps
                ).mul(2.).sub(1.)

                if trajectories is not None:
                    trajectories.append((x_0_hat.add(1.).mul(.5).cpu(), image.clone().detach().add(1.).mul(.5).cpu()))
                # prev_t = self.scheduler.previous_timestep(t)
                prev_t = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                prev_t = prev_t if prev_t > 0 else torch.tensor(0, device=prev_t.device)

                if renoise_with_pred:
                    # eps_bar = self.eps_bar(x_t=x_t, x_0_hat=x_0_hat, t=t)
                    eps_bar = scheduler_output.eps_hat
                    image = self.scheduler.add_noise(image, eps_bar, timesteps=prev_t)
                else:
                    image = self.scheduler.add_noise(image, torch.randn_like(image, device=image.device), timesteps=prev_t)
            else:
                image = scheduler_output.prev_sample.detach()

        return image

    def consistency_guidance_partial(
        self,
        *,
        conditioning,
        target_shape: Size,
        masked_sinogram: Tensor,
        mask: Tensor,
        projector: SimpleProjector,
        partial_input: Tensor,
        partial_starting_index: int = 25,
        do_dps: Optional[Union[Tensor, List]] = None,
        n_gradient_steps: int = 20,
        generator: Optional[Union[Generator, List[Generator]]] = None,
        num_inference_steps: int = 50,
        guidance_fn: Optional[Callable] = None,
        guidance_input_prep_fn: Optional[Callable] = None,
        guidance_scale: float = 1.,
        unconditional_input: Optional[Tensor] = None,
        trajectories: Optional[List] = None,
        renoise_with_pred: bool = True
    ):
        assert (guidance_fn is None) == (guidance_input_prep_fn is None)

        if do_dps is None:
            do_dps = torch.ones((num_inference_steps, ), device=conditioning.device)
        elif not isinstance(do_dps, Tensor):
            do_dps = torch.tensor(do_dps, dtype=torch.bool)
        assert do_dps.shape[0] == num_inference_steps

        self.unet.eval()

        noise = torch.randn(
            target_shape,
            device=conditioning.device,
            generator=generator,
        )

        self.scheduler.set_timesteps(num_inference_steps)
        starting_timestep = self.scheduler.timesteps[partial_starting_index].to(conditioning.device)

        image = self.scheduler.add_noise(partial_input, noise, timesteps=starting_timestep)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            if i < partial_starting_index:
                continue
            with torch.no_grad():
                scheduler_output = self.__step(
                    image=image,
                    conditioning=conditioning,
                    t=t,
                    device=conditioning.device,
                    generator=generator,
                    guidance_fn=guidance_fn,
                    guidance_input_prep_fn=guidance_input_prep_fn,
                    guidance_scale=guidance_scale,
                    unconditional_input=unconditional_input
                )

            if do_dps[i]:

                x_0_hat = scheduler_output.pred_original_sample.clone().detach().cpu()
                image = fast_sampling(
                    scheduler_output.pred_original_sample.detach().add(1.).mul(.5),
                    masked_sinogram,
                    lr=5e-2,
                    projector=projector,
                    mask=mask,
                    n_steps=n_gradient_steps
                ).mul(2.).sub(1.)

                if trajectories is not None:
                    trajectories.append((x_0_hat.add(1.).mul(.5), image.clone().detach().add(1.).mul(.5).cpu()))
                # prev_t = self.scheduler.previous_timestep(t)
                prev_t = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                prev_t = prev_t if prev_t > 0 else torch.tensor(0, device=prev_t.device)

                if renoise_with_pred:
                    image = self.scheduler.add_noise(image, scheduler_output.eps_hat, timesteps=prev_t)
                else:
                    image = self.scheduler.add_noise(image, torch.randn_like(image, device=image.device), timesteps=prev_t)
            else:
                image = scheduler_output.prev_sample.detach()

        return image

    def vanilla_dps(
        self,
        conditioning,
        target_shape: Size,
        masked_sinogram: Tensor,
        mask: Tensor,
        projector: SimpleProjector,
        generator: Optional[Union[Generator, List[Generator]]] = None,
        guidance_fn: Optional[Callable] = None,
        guidance_input_prep_fn: Optional[Callable] = None,
        guidance_scale: float = 1.,
        unconditional_input: Optional[Tensor] = None,
        num_inference_steps: int = 1000,
        trajectories: Optional[List] = None,
    ):
        assert (guidance_fn is None) == (guidance_input_prep_fn is None)

        self.unet.eval()

        image = torch.randn(
            target_shape,
            device=conditioning.device,
            generator=generator,
        )

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            image = image.requires_grad_(True)

            scheduler_output = self.__step(
                image=image,
                conditioning=conditioning,
                t=t,
                device=conditioning.device,
                generator=generator,
                guidance_fn=guidance_fn,
                guidance_input_prep_fn=guidance_input_prep_fn,
                guidance_scale=guidance_scale,
                unconditional_input=unconditional_input
            )
            x_0_hat = scheduler_output.pred_original_sample.clone().detach().add(1).mul(.5).cpu()
            image, distance = diffusion_posterior_sampling(
                x_t_minus_one=scheduler_output.prev_sample.add(1.).mul(.5),
                measurement=masked_sinogram,
                measurement_mask=mask,
                x_t=image,
                x_0_hat=scheduler_output.pred_original_sample.add(1.).mul(.5),
                projector=projector,
            )
            if trajectories is not None:
                trajectories.append((x_0_hat, image.clone().detach().cpu()))

            image = image.mul(2.).sub(1.).detach()

        return image