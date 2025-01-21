from dataclasses import dataclass
from functools import partial
from typing import Union

import torch
from torch import nn, Tensor

from diffusion.functional import generate_random_mask_expanded
from diffusion.functional.batch_slice_mask import generate_mask_from_fixed_expanded, generate_contiguous_mask_expanded, \
    generate_random_contiguous_mask_expanded


@dataclass
class SliceMaskOutput:
    sample: Tensor
    mask: Tensor
    mask_amount: Tensor


class SliceRandomMask(nn.Module):
    def __init__(
        self,
        keep_min: int,
        keep_max: int,
        device: str,
        mask_value: float = 0.,
    ):
        super(SliceRandomMask, self).__init__()

        self.mask_value = mask_value
        self.device = device

        self.keep_min = keep_min
        self.keep_max = keep_max
        self.mask_fn = None
        self.contiguous_mask_fn = None
        self.generate_mask_from_fixed_expanded = None
        self.generate_contiguous_mask_fixed_expanded = None
        self.reset_rng()

    def reset_rng(self):
        self.mask_fn = partial(
            generate_random_mask_expanded,
            keep_min=self.keep_min,
            keep_max=self.keep_max,
            device=self.device,
            generator=torch.Generator(device=self.device).manual_seed(0),
        )
        self.contiguous_mask_fn = partial(
            generate_random_contiguous_mask_expanded,
            keep_min=self.keep_min,
            keep_max=self.keep_max,
            device=self.device,
            generator=torch.Generator(device=self.device).manual_seed(0),
        )
        self.generate_mask_from_fixed_expanded = partial(
            generate_mask_from_fixed_expanded,
            generator=torch.Generator(device=self.device).manual_seed(0),
        )
        self.generate_contiguous_mask_fixed_expanded = partial(
            generate_contiguous_mask_expanded,
            generator=torch.Generator(device=self.device).manual_seed(0),
        )


    def apply_mask(
        self,
        x: Tensor,
        mask: Tensor,
        return_dict: bool = True
    ) -> Union[Tensor, SliceMaskOutput]:
        masked_x = x * mask + (1-mask) * self.mask_value
        if not return_dict:
            return masked_x
        else:
            mask_amount = mask[:, 0, :, 0].sum(dim=-1) / x.shape[2]

            return SliceMaskOutput(
                sample=masked_x,
                mask=mask,
                mask_amount=mask_amount,
            )

    def forward(
        self,
        x: Tensor,
        return_dict: bool = True
    ) -> Union[Tensor, SliceMaskOutput]:
        b, c, s, w = x.size()
        mask = self.mask_fn(batch_size=b, channels=c, sinogram_size=s, width=w)

        return self.apply_mask(x, mask, return_dict)

    def contiguous(
        self,
        x: Tensor,
        return_dict: bool = True
    ) -> Union[Tensor, SliceMaskOutput]:
        b, c, s, w = x.size()
        mask = self.contiguous_mask_fn(batch_size=b, channels=c, sinogram_size=s, width=w)

        return self.apply_mask(x, mask, return_dict)

    def fixed_sparsity_mask(
        self,
        *,
        x: Tensor,
        keep_n_angles: int,
        return_dict: bool = True
    ) -> Union[Tensor, SliceMaskOutput]:
        b, c, s, w = x.size()
        mask = self.generate_mask_from_fixed_expanded(
            batch_size=b,
            channels=c,
            sinogram_size=s,
            width=w,
            keep=torch.ones(b, device=self.device) * keep_n_angles,
            device=self.device
        )

        return self.apply_mask(x, mask, return_dict)

    def contiguous_mask(
        self,
        *,
        x: Tensor,
        keep_n_angles: int,
        return_dict: bool = True
    ):
        b, c, s, w = x.size()
        mask = self.generate_contiguous_mask_fixed_expanded(
            batch_size=b,
            channels=c,
            sinogram_size=s,
            width=w,
            keep=torch.ones(b, device=self.device) * keep_n_angles,
            device=self.device
        )

        return self.apply_mask(x, mask, return_dict)