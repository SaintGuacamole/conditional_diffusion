from typing import Literal, Optional, Union, List

import torch
from diffusers import DiffusionPipeline
from torch import Generator, Tensor

from diffusion.models import UNetModel


class ColdDiffusionPipeline(DiffusionPipeline):

    def __init__(
            self,
            *,
            unet: UNetModel,
    ):
        super(ColdDiffusionPipeline, self).__init__()
        self.register_modules(
            unet=unet,
        )
        self.register_to_config()


    @torch.no_grad()
    def __call__(
            self,
            masked_sinogram: Tensor,
            mask: Tensor,
            starting_step: int,
            end_step: int,
            generator: Optional[Union[Generator, List[Generator]]] = None,
            var_mode: Literal["mean", "max"] = "mean"
    ):
        self.unet.eval()

        b = masked_sinogram.size(0)

        for s in self.progress_bar(range(starting_step, end_step)):

            model_input = torch.cat([masked_sinogram, mask], dim=1)

            model_pred = self.unet(model_input, embeddings=(s, )).sample

            pred_mean, log_var = torch.chunk(model_pred, chunks=2, dim=1)

            log_var = torch.exp(log_var)

            masked_var = (1. - mask) * torch.clamp(log_var, min=0.) + mask * 10000

            match var_mode:
                case "mean":
                    slice_var = torch.mean(masked_var, dim=(1, -1))
                    next_slice = torch.argmin(slice_var, dim=-1)
                    pass
                case "max":
                    slice_var = torch.max(torch.max(masked_var, dim=1), dim=-1)
                    next_slice = torch.argmin(slice_var, dim=-1)
                case _:
                    raise NotImplementedError(f"Unknown variance mode: {var_mode}")

            for e in range(b):
                masked_sinogram[e, :, next_slice[e], :] = pred_mean[e, :, next_slice[e], :]
                mask[e, :, next_slice[e], :] = 1.

        return masked_sinogram
