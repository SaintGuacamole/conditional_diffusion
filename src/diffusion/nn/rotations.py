import math
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from diffusion.nn import CircleMask


class ExtractIntoRotations(nn.Module):
    def __init__(
        self,
        *,
        size: int,
        device: str = 'cuda',
        theta: Optional[Tensor] = None,
        n_rotations: Optional[int] = None,
        angular_range: int = 180,
        circle_mask: bool = True,
        circle_mask_value: Optional[float] = 0.
    ):

        super(ExtractIntoRotations, self).__init__()

        if n_rotations is None and theta is None:
            raise ValueError(f"Either `n_rotations` or `theta` must be specified.")

        if theta is None:
            angles = torch.linspace(0, angular_range, n_rotations, device=device)
            theta = math.pi * angles / float(angular_range) - math.pi/2
        else:
            n_rotations = theta.shape[0]

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        zeros = torch.zeros_like(cos_theta)

        self.affine_matrices = torch.stack([
            torch.stack([cos_theta, -sin_theta, zeros], dim=1),
            torch.stack([sin_theta, cos_theta, zeros], dim=1)
        ], dim=1)

        grid = F.affine_grid(
            self.affine_matrices,
            (n_rotations, 1, size, size),
            align_corners=True
        )

        if circle_mask:
            self.circle_mask = CircleMask(
                size=size,
                device=device
            )
            self.circle_mask_value = circle_mask_value
        else:
            self.circle_mask = None

        self.register_buffer('base_grid', grid)
        self.n_rotations = n_rotations
        self.size = size
        self.device = device

    def forward(self, x: Tensor) -> Tensor:

        batch_size = x.shape[0]

        if x.shape[1] == self.n_rotations:
            # x is a stack of images
            x = x.view(batch_size * self.n_rotations, 1, self.size, self.size)
        else:
            assert x.shape[1] == 1

            if x.shape[2] == self.n_rotations and x.shape[3] == self.size:
                # x should be a sinogram
                x = x.flip(dims=(2,))
                x = x.squeeze(1).unsqueeze(-1)
                x = x.expand(-1, -1, -1, self.size)
                x = x.view(batch_size * self.n_rotations, 1, self.size, self.size)

            elif x.shape[2] == self.size and x.shape[3] == self.size:
                # x is a one channel image
                x = x.repeat_interleave(self.n_rotations, dim=0)

        grid = self.base_grid.repeat(batch_size, 1, 1, 1)

        rotated = F.grid_sample(x, grid, align_corners=True, padding_mode="border")

        # (batch_size, n_rotations, height, width)
        rotated = rotated.squeeze(1).view(batch_size, self.n_rotations, self.size, self.size)
        if self.circle_mask is not None:
            rotated = self.circle_mask(rotated, mask_value=self.circle_mask_value)
        return rotated

