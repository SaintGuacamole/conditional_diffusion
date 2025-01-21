
from pathlib import Path
from typing import Union, Literal, Tuple

import torch
import torchvision
from safetensors import safe_open
from torch import Tensor
from torchvision.transforms import InterpolationMode

from diffusion.datasets import Dataset
from diffusion.nn import CircleMask


class SafetensorsImageDataset(Dataset):

    def __init__(
        self,
        *,
        directory: Union[str, Path],
        split: Literal["train", "val", "test"],
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        image_size: int = 256,
        scale: Tuple[float, float] = (1., 1.),
        shear: Tuple[float, float] = (0., 0.),
        rotation: Tuple[float, float] = (-90, 90),
        circle_mask: bool = True,
        mean: float = 0.5,
        std: float = 0.5
    ):
        super(SafetensorsImageDataset, self).__init__(
            split=split,
            collate_fn=collate_fn
        )

        self.samples = sorted(list((Path(directory) / split).glob("**/*.safetensors")))
        self.device = device

        if circle_mask:
            self.circle_mask = CircleMask(
                size=image_size,
                device=self.device
            )
        else:
            self.circle_mask = lambda x, **kwargs: x

        self.transforms = torch.nn.Sequential(
            torchvision.transforms.RandomAffine(
                degrees=rotation,
                scale=scale,
                shear=shear,
                interpolation=InterpolationMode.BILINEAR,
            ),
            torchvision.transforms.Normalize([mean], [std], inplace=True),
        )

    def __len__(self):
        return len(list(self.samples))

    def __getitem__(self, idx: int) -> Tensor:

        with safe_open(
                filename=self.samples[idx],
                framework="pt",
                device=self.device
        ) as f:
            image = f.get_tensor(name="image")

        if len(image.shape) == 2:
            image = image.unsqueeze(0)

        image = self.transforms(image)
        image = self.circle_mask(image, mask_value=-1.)

        return image


def collate_fn(batch):
    images = [b for b in batch]
    images = torch.stack(images).to(memory_format=torch.contiguous_format).float()

    return {"image": images}
