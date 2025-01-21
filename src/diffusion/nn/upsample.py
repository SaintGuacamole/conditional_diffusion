from typing import Optional

import torch
import torch.nn.functional as F

from torch import nn, Tensor

from diffusion.nn import UpSampleMode


class Upsample(nn.Module):

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        interpolate=True,
        conv: bool = False,
        mode: UpSampleMode = "nearest",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.interpolate = interpolate
        self.mode = mode
        self.conv = None

        if conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=3, padding=1, bias=True)


    def forward(
            self,
            x: Tensor,
            output_size: Optional[int] = None,
    ) -> Tensor:

        assert x.shape[1] == self.channels

        dtype = x.dtype
        if dtype == torch.bfloat16:
            x = x.to(torch.float32)

        if x.shape[0] >= 64:
            x = x.contiguous()


        if output_size is None:
            x = F.interpolate(x, scale_factor=2.0, mode=self.mode)
        else:
            x = F.interpolate(x, size=output_size, mode=self.mode)

        if dtype == torch.bfloat16:
            x = x.to(dtype)

        if self.conv is not None:
            x = self.conv(x)
        return x