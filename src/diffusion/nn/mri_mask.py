from dataclasses import dataclass
from functools import partial
from typing import Optional, Union

import numpy as np
import torch
from torch import nn, Tensor
from torch.fft import fftshift, fft2, ifft2, ifftshift

from diffusion.functional.mri import cartesian_k_space_mask, random_k_space_mask, gaussian_k_space_mask


@dataclass
class MRISubsampleOutput:
    sample: Tensor
    mask: Tensor
    k_space: Tensor
    masked_k_space: Tensor

class MRISubsample(nn.Module):

    def __init__(
        self,
        keep_min: int,
        keep_max: int,
        mode: str = "cartesian",
        preverse_center: int = 2,
        device=torch.device("cpu"),
    ):
        super(MRISubsample, self).__init__()
        self.mask_value = 0.
        self.mode = mode
        self.keep_min = keep_min
        self.keep_max = keep_max
        self.preverse_center = preverse_center

        self.cartesian_k_space_mask = partial(
            cartesian_k_space_mask,
            generator=np.random.Generator(np.random.PCG64(0))
        )
        self.gaussian_k_space_mask = partial(
            gaussian_k_space_mask,
            generator=np.random.Generator(np.random.PCG64(0))
        )
        self.random_k_space_mask = partial(
            random_k_space_mask,
            generator=torch.Generator(device=device).manual_seed(0)
        )


    def apply_mask(
        self,
        x: Tensor,
        mask: Tensor,
        return_dict: bool = True
    ) -> Union[Tensor, MRISubsampleOutput]:

        k_space = fftshift(fft2(x))
        masked_k_space = k_space * mask + (1-mask) * self.mask_value
        masked_x = torch.abs(ifft2(ifftshift(masked_k_space)))

        if return_dict:
            return MRISubsampleOutput(sample=masked_x, mask=mask, k_space=k_space, masked_k_space=masked_k_space)
        else:
            return masked_x

    def forward(
        self,
        x: Tensor,
        mode: Optional[str] = None,
        return_dict: bool = True,
        keep: Optional[Union[Tensor]] = None,
    ) -> Union[Tensor, MRISubsampleOutput]:
        mode = mode or self.mode

        b, c, h, w = x.shape

        match mode:
            case "cartesian":
                keep = keep if keep is not None else torch.randint(self.keep_min, self.keep_max+1, (b,))
                mask = self.cartesian_k_space_mask(
                    batch=b, channels=c, height=h, width=w,
                    device=x.device,
                    keep_n_cols=keep,
                    center_width=self.preverse_center
                )
                return self.apply_mask(x=x, mask=mask, return_dict=return_dict)
            case "gaussian":
                if keep is None:
                    assert self.keep_max <= (h // 2) * w + w + 1
                    keep = torch.randint(self.keep_min, self.keep_max+1, (b, )).to(x.device)
                mask = self.gaussian_k_space_mask(
                    batch_size=b, channels=c, height=h, width=w,
                    keep_n_pixels=keep,
                    device=x.device,
                    center_radius=self.preverse_center
                )
                return self.apply_mask(x=x, mask=mask, return_dict=return_dict)
            case "random":
                keep = keep if keep is not None else torch.randint(self.keep_min, self.keep_max+1, (b, )).to(x.device)
                mask = self.random_k_space_mask(
                    b=b, c=c, h=h, w=w,
                    keep_n_pixels=keep,
                    device=x.device
                )
                return self.apply_mask(x=x, mask=mask, return_dict=return_dict)
            case _:
                raise NotImplementedError