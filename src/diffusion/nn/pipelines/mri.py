
from typing import Optional, Union, List, Callable

import torch
from diffusers import DiffusionPipeline, SchedulerMixin
from torch import Generator, Tensor, Size

from diffusion.functional import compute_guidance
from diffusion.functional.conditioning_functions import mri_fast_sampling
from diffusion.models import UNetModel, SinogramConditionedUnet


class MRIPipeline(DiffusionPipeline):

    def __init__(
            self,
            *,
            unet: Union[UNetModel, SinogramConditionedUnet],
            scheduler: SchedulerMixin
    ):
        super(MRIPipeline, self).__init__()
        self.register_modules(
            unet=unet,
            scheduler=scheduler
        )
        self.register_to_config()


    @torch.no_grad()
    def __call__(
        self,
        conditioning,
        target_shape: Size,
        mask_emb: Optional[Tensor] = None,
        generator: Optional[Union[Generator, List[Generator]]] = None,
        num_inference_steps: int = 50,
        guidance_fn: Optional[Callable] = None,
        guidance_input_prep_fn: Optional[Callable] = None,
        guidance_scale: float = 1.,
        unconditional_input: Optional[Tensor] = None,
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
                mask_emb=mask_emb,
                device=conditioning.device,
                generator=generator,
                guidance_fn=guidance_fn,
                guidance_input_prep_fn=guidance_input_prep_fn,
                guidance_scale=guidance_scale,
                unconditional_input=unconditional_input,
            )
            image = scheduler_output.prev_sample

        return image


    def __step(
        self,
        image: Tensor,
        conditioning: Tensor,
        t: Tensor,
        device: torch.device,
        mask_emb: Optional[Tensor] = None,
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
                embeddings=(t, mask_emb) if (mask_emb is not None) else (t,),
            ).sample
        else:
            with torch.no_grad():
                model_output = self.unet(
                    sample=image_input,
                    conditioning=conditioning_input,
                    embeddings=(t, mask_emb) if (mask_emb is not None) else (t,),
                ).sample

        if guidance_fn is not None and guidance_input_prep_fn is not None:
            model_output = compute_guidance(model_output, guidance_fn, guidance_scale)

        scheduler_output = self.scheduler.step(model_output, t, image, generator=generator)
        return scheduler_output


    def consistency_guidance(
        self,
        conditioning,
        target_shape: Size,
        masked_k_space: Tensor,
        mask: Tensor,
        mask_emb: Optional[Tensor] = None,
        do_dps: Optional[Union[Tensor, List]] = None,
        n_gradient_steps: int = 20,
        generator: Optional[Union[Generator, List[Generator]]] = None,
        num_inference_steps: int = 50,
        guidance_fn: Optional[Callable] = None,
        guidance_input_prep_fn: Optional[Callable] = None,
        guidance_scale: float = 1.,
        unconditional_input: Optional[Tensor] = None,
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
                    mask_emb=mask_emb,
                    device=conditioning.device,
                    generator=generator,
                    guidance_fn=guidance_fn,
                    guidance_input_prep_fn=guidance_input_prep_fn,
                    guidance_scale=guidance_scale,
                    unconditional_input=unconditional_input
                )

            if do_dps[i]:
                x_0_hat = scheduler_output.pred_original_sample.detach().add(1.).mul(.5)
                image = mri_fast_sampling(
                    x=x_0_hat,
                    target=masked_k_space,
                    lr=5e-3,
                    mask=mask,
                    n_steps=n_gradient_steps
                ).mul(2.).sub(1.)

                # prev_t = self.scheduler.previous_timestep(t)
                prev_t = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                prev_t = prev_t if prev_t > 0 else torch.tensor(0, device=prev_t.device)

                image = self.scheduler.add_noise(image, torch.randn_like(image, device=image.device), timesteps=prev_t)
            else:
                image = scheduler_output.prev_sample.detach()

        return image
