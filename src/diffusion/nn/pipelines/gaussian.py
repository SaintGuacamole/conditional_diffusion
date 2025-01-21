from typing import Optional, Union, List

import numpy as np
import torch
from diffusers import DiffusionPipeline
from torch import Generator, Size

from diffusion.models import UNetModel
from diffusion.nn.scheduler import GaussianDiffusion


class GaussianPipeline(DiffusionPipeline):

    def __init__(
            self,
            *,
            unet: UNetModel,
            scheduler: GaussianDiffusion
    ):
        super(GaussianPipeline, self).__init__()
        self.register_modules(
            unet=unet,
            scheduler=scheduler
        )
        self.register_to_config()

    @staticmethod
    def prox(
        inp,
        noisy_sino,
        mask,
        rho,
        device,
        projector,
    ):
        x = inp.detach().clone()
        z = inp.detach().clone()

        s = x
        t = torch.tensor([1.]).float().to(device)
        for _ in range(30):

            projected = projector(s)
            diff = mask * (projected - noisy_sino)
            grad = projector.fbp(diff)

            xnext = s - 5e-6*grad - rho*(s - z)

            tnext = 0.5*(1+torch.sqrt(1+4*t*t))

            s = xnext + ((t-1)/tnext)*(xnext-x)

            t = tnext
            x = xnext
        return x


    @torch.no_grad()
    def __call__(
        self,
        sparse_fbp,
        masked_sinogram,
        mask,
        projector,
        image_size: Size,
        generator: Optional[Union[Generator, List[Generator]]] = None,
        num_inference_steps: int = 50,
        prox_solve: bool = False,
    ):
        self.unet.eval()

        image = torch.randn(
            image_size,
            device=sparse_fbp.device,
        )

        self.scheduler.set_timesteps(num_inference_steps)

        indices = list(range(num_inference_steps))[::-1]
        linespace_ = np.linspace(1, len(indices), len(indices), endpoint=True)[::-1]
        rhos = list(map(lambda x: 1 - np.exp(-x/(8*len(indices))), linespace_))
        rhos = torch.tensor(rhos).to(sparse_fbp.device)

        i = 0

        for t in self.progress_bar(self.scheduler.timesteps):
            t = torch.tensor([t] * sparse_fbp.shape[0], device=sparse_fbp.device)

            model_input = torch.cat([image, sparse_fbp], dim=1)

            model_output = self.unet(model_input, embeddings=(t, )).sample

            image = self.scheduler.p_sample(
                model=lambda *_: model_output,
                x=image,
                t=t,
            )["sample"]

            noisy_sinogram = self.scheduler.add_noise(
                masked_sinogram,
                torch.randn_like(masked_sinogram, device=masked_sinogram.device),
                t,
            )
            #
            # image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

            if prox_solve:
                image = self.prox(
                    image,
                    noisy_sinogram,
                    mask,
                    rhos[i],
                    sparse_fbp.device,
                    projector,
                )

            i = i + 1

        return image