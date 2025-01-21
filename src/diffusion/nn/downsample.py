from typing import Optional

from torch import nn, Tensor

from diffusion.nn import DownSampleMode


class Downsample(nn.Module):

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        mode: DownSampleMode = "avg"
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        match mode:
            case "avg":
                assert channels == out_channels
                self.down = nn.AvgPool2d(kernel_size=2, stride=2)
            case "max":
                assert channels == out_channels
                self.down = nn.MaxPool2d(kernel_size=2, stride=2)
            case "conv":
                self.down = nn.Conv2d(
                    self.channels, self.out_channels, kernel_size=3, stride=2, padding=1, bias=True
                )
            case _:
                raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:

        assert x.shape[1] == self.channels
        x = self.down(x)
        return x