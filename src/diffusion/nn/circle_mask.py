from typing import Union, Tuple, Literal, Optional

import numpy as np
import torch
from torch import nn, Tensor


class CircleMask(nn.Module):

    def __init__(
            self,
            *,
            size: Union[int, Tuple[int, int]],
            device: Optional[Literal["cpu", "cuda", "mps"]] = "cpu"
    ):
        super(CircleMask, self).__init__()
        if isinstance(size, int):
            size = (size, size)

        center = (int(size[0] / 2) - 0.5, int(size[1] / 2) - 0.5)
        radius = min(center[0], center[1], size[0] - center[0], size[1] - center[1])

        y, x = np.ogrid[:size[0], :size[1]]
        dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

        mask = dist_from_center <= radius

        self.register_buffer('mask', torch.from_numpy(mask).to(device))

    def forward(
            self,
            x: Tensor,
            *,
            return_mask: bool = False,
            mask_value: float = 0.0,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:

        if return_mask:
            return x * self.mask + mask_value * ~self.mask, self.mask
        else:
            return x * self.mask + mask_value * ~self.mask

