from dataclasses import dataclass, astuple
from typing import Union, Dict, Optional, Generator, Callable, Tuple, List

import torch
from diffusers import DiffusionPipeline, SchedulerMixin
from torch import Tensor, Size

from diffusion.functional.conditioning_functions import diffusion_posterior_sampling, fast_sampling
from diffusion.models import UNetModel, SinogramConditionedUnet
from diffusion.nn.leap_projector_wrapper import SimpleProjector

class VariablePipeline(DiffusionPipeline):

    def __init__(
        self,
        *,
        model: Union[UNetModel, SinogramConditionedUnet],
        scheduler: SchedulerMixin,
        progress_bar: bool = False,
    ):
        super(VariablePipeline, self).__init__()
        self.register_modules(
            model=model,
            scheduler=scheduler
        )
        self.set_progress_bar_config(disable=not progress_bar)
        self.register_to_config()


    def __step(
        self,
        *,
        model_input: Dict[str, Tensor],
        t,
        image: Tensor,
        generator: Optional[Generator] = None,
        device: torch.device,
        grad: bool = False,
        guidance: Optional[Tuple[Callable, Callable]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor]:

        if guidance is not None:
            model_input = guidance[0](model_input)

        if grad:
            model_output = self.model(**model_input).sample
        else:
            with torch.no_grad():
                model_output = self.model(**model_input).sample

        if guidance is not None:
            model_output = guidance[1](model_output)

        prev_sample, pred_original_sample = self.scheduler.step(
            model_output,
            t,
            image,
            generator=generator,
            return_dict=False,
        )

        if (
            model_output.shape[1] == image.shape[1] * 2
            and self.scheduler.variance_type in ["learned", "learned_range"]
        ):
            model_output, _ = torch.split(model_output, image.shape[1], dim=1)

        return prev_sample, pred_original_sample, model_output

    def __prev_t(
        self,
        t
    ):
        prev_t = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        return prev_t if prev_t > 0 else torch.tensor(0, device=t.device)

    def vanilla(
        self,
        *,
        target_shape: Size,
        device: torch.device,
        model_input: Callable[[Tensor, Tensor], Dict[str, Tensor]],
        generator: Optional[Generator] = None,
        n_inference_steps: int = 50,
        **kwargs
    ):
        self.model.eval()

        self.scheduler.set_timesteps(num_inference_steps=n_inference_steps, device=device)
        image = torch.randn(target_shape, device=device)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            t = t.to(device)
            prev_sample, _, _ = self.__step(
                model_input=model_input(image, t),
                t=t,
                image=image,
                generator=generator,
                device=device,
                grad=False,
                kwargs=kwargs
            )

            image = prev_sample

        return image

    def dps(
        self,
        *,
        target_shape: Size,
        device: torch.device,
        model_input: Callable[[Tensor, Tensor], Dict[str, Tensor]],
        generator: Optional[Generator] = None,
        n_inference_steps: int = 1000,
        masked_sinogram: Tensor,
        mask: Tensor,
        projector: SimpleProjector,
        **kwargs
    ):
        self.model.eval()
        self.scheduler.set_timesteps(num_inference_steps=n_inference_steps, device=device)

        image = torch.randn(target_shape, device=device)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            t = t.to(device)
            image = image.requires_grad_(True)
            prev_sample, pred_original_sample, _ = self.__step(
                model_input=model_input(image, t),
                t=t,
                image=image,
                generator=generator,
                device=device,
                grad=False,
                kwargs=kwargs
            )

            image, distance = diffusion_posterior_sampling(
                x_t_minus_one=prev_sample.add(1.).mul(.5),
                measurement=masked_sinogram,
                measurement_mask=mask,
                x_t=image,
                x_0_hat=pred_original_sample.add(1.).mul(.5),
                projector=projector,
            )
            image = image.mul(2.).sub(1.).detach()

        return image

    def cg(
        self,
        *,
        target_shape: Size,
        device: torch.device,
        model_input: Callable[[Tensor, Tuple[Tensor]], Dict[str, Tensor]],
        generator: Optional[Generator] = None,
        n_inference_steps: int = 50,
        n_consistency_steps: int = 20,
        consistency_lr: float = 5e-2,
        masked_sinogram: Tensor,
        mask: Tensor,
        projector: SimpleProjector,
        cg_mask: Optional[Union[Tensor, List]] = None,
        renoise_with_eps: bool = True,
        **kwargs
    ):
        if cg_mask is None:
            cg_mask = torch.ones((n_inference_steps, ), device=device)

        assert len(cg_mask) == n_inference_steps, f"cg mask must have length {n_inference_steps}."
        self.model.eval()
        self.scheduler.set_timesteps(num_inference_steps=n_inference_steps, device=device)

        image = torch.randn(target_shape, device=device)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            t = t.to(device)
            prev_sample, pred_original_sample, model_output = self.__step(
                model_input=model_input(image, (t,)),
                t=t,
                image=image,
                generator=generator,
                device=device,
                grad=False,
                kwargs=kwargs
            )

            if cg_mask[i]:
                image = fast_sampling(
                    x=pred_original_sample.detach().add(1.).mul(.5),
                    target=masked_sinogram,
                    lr=consistency_lr,
                    projector=projector,
                    mask=mask,
                    n_steps=n_consistency_steps
                ).mul(2.).sub(1.)

                prev_t = self.__prev_t(t)

                if renoise_with_eps:
                    z = model_output
                else:
                    z = torch.randn_like(image, device=image.device)

                image = self.scheduler.add_noise(image, z, timesteps=prev_t).detach()
            else:
                image = prev_sample.detach()

        return image

    def __eps_bar(
        self,
        x_t: Tensor,
        x_0_hat: Tensor,
        t
    ):
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        return (x_t - x_0_hat * alpha_prod_t ** 0.5) / (1 - alpha_prod_t) ** 0.5
