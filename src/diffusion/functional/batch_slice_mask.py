import torch
from torch import Tensor
from typing import Union, Literal, Optional

from torch.distributions import Beta


def generate_mask_from_fixed(
    batch_size: int,
    sinogram_size: int,
    keep: Tensor,
    device: Union[torch.device, Literal["cuda", "cpu", "mps"]],
    generator: Optional[torch.Generator] = None,
) -> Tensor:

    mask = torch.ones((batch_size, 1, sinogram_size, 1), device=device)

    row_indices = torch.arange(sinogram_size, device=device).expand(batch_size, sinogram_size)
    rand_perm = torch.argsort(torch.rand(batch_size, sinogram_size, device=device, generator=generator), dim=1)

    batch_mask = row_indices < keep.unsqueeze(1)

    mask.scatter_(2, rand_perm.unsqueeze(1).unsqueeze(3), batch_mask.unsqueeze(1).unsqueeze(3).float())

    return mask

def generate_mask_from_fixed_expanded(
    batch_size: int,
    channels: int,
    sinogram_size: int,
    width: int,
    keep: Tensor,
    device: Union[torch.device, Literal["cuda", "cpu", "mps"]],
    generator: Optional[torch.Generator] = None
):
    mask = generate_mask_from_fixed(
        batch_size=batch_size,
        sinogram_size=sinogram_size,
        keep=keep,
        device=device,
        generator=generator
    )
    return mask.expand(-1, channels, -1, width)


def generate_random_mask(
    batch_size: int,
    sinogram_size: int,
    keep_min: int,
    keep_max: int,
    device: Union[torch.device, Literal["cuda", "cpu", "mps"]],
    use_beta: bool = False,
    generator: Optional[torch.Generator] = None
) -> Tensor:
    if use_beta or True:
        beta_dist = Beta(torch.tensor([2.]), torch.tensor([5.]))

        samples = beta_dist.sample((batch_size, )).to(device)

        scaled_samples = (keep_max - keep_min) * samples + keep_min

        num_masked = torch.round(scaled_samples).int()

    else:
        num_masked = torch.randint(
            low=keep_min,
            high=keep_max + 1,
            size=(batch_size, 1),
            device=device,
            generator=generator,
        )

    mask = torch.ones((batch_size, 1, sinogram_size, 1), device=device)

    row_indices = torch.arange(sinogram_size, device=device).expand(batch_size, sinogram_size)
    rand_perm = torch.argsort(torch.rand(batch_size, sinogram_size, device=device, generator=generator), dim=1)

    batch_mask = row_indices < num_masked

    mask.scatter_(2, rand_perm.unsqueeze(1).unsqueeze(3), batch_mask.unsqueeze(1).unsqueeze(3).float())

    return mask

def generate_random_mask_expanded(
    batch_size: int,
    channels: int,
    sinogram_size: int,
    width: int,
    keep_min: int,
    keep_max: int,
    device: Union[torch.device, Literal["cuda", "cpu", "mps"]],
    generator: Optional[torch.Generator] = None
) -> Tensor:
    mask = generate_random_mask(
        batch_size=batch_size,
        sinogram_size=sinogram_size,
        keep_min=keep_min,
        keep_max=keep_max,
        device=device,
        generator=generator
    )
    return mask.expand(-1, channels, -1, width)


def generate_random_contiguous_mask(
    batch_size: int,
    sinogram_size: int,
    keep_min: int,
    keep_max: int,
    device: Union[torch.device, Literal["cuda", "cpu", "mps"]],
    generator: Optional[torch.Generator] = None
):
    num_masked = torch.randint(
        low=keep_min,
        high=keep_max + 1,
        size=(batch_size, 1),
        device=device,
        generator=generator,
    )

    start_positions = torch.rand(
        size=(batch_size, 1),
        device=device,
        generator=generator,
    )

    possible_starting_positions = sinogram_size - num_masked
    start_positions = (possible_starting_positions * start_positions).floor()

    row_indices = torch.arange(sinogram_size, device=device).expand(batch_size, sinogram_size)

    mask = (row_indices >= start_positions) & (row_indices < (start_positions + num_masked))
    mask = mask.view(batch_size, 1, sinogram_size, 1).float()

    return mask


def generate_random_contiguous_mask_expanded(
    batch_size: int,
    channels: int,
    sinogram_size: int,
    width: int,
    keep_min: int,
    keep_max: int,
    device: Union[torch.device, Literal["cuda", "cpu", "mps"]],
    generator: Optional[torch.Generator] = None
) -> Tensor:
    mask = generate_random_contiguous_mask(
        batch_size=batch_size,
        sinogram_size=sinogram_size,
        keep_min=keep_min,
        keep_max=keep_max,
        device=device,
        generator=generator
    )
    return mask.expand(-1, channels, -1, width)


def generate_contiguous_mask_expanded(
    batch_size: int,
    channels: int,
    sinogram_size: int,
    width: int,
    keep: Tensor,
    device: Union[torch.device, Literal["cuda", "cpu", "mps"]],
    generator: Optional[torch.Generator] = None
):

    start_positions = torch.rand(
        size=(batch_size, 1),
        device=device,
        generator=generator,
    )

    possible_starting_positions = sinogram_size - keep
    start_positions = (possible_starting_positions * start_positions).floor()

    row_indices = torch.arange(sinogram_size, device=device).expand(batch_size, sinogram_size)

    mask = (row_indices >= start_positions) & (row_indices < (start_positions + keep))
    mask = mask.view(batch_size, 1, sinogram_size, 1).float()
    return mask.expand(-1, channels, -1, width)


def subsample_mask(
    mask: Tensor,
    num_to_drop: int,
    device: Union[torch.device, Literal["cuda", "cpu", "mps"]],
    generator: Optional[torch.Generator] = None
):
    new_mask = mask.clone()

    slice_mask = new_mask[:, :, :, 0]
    ones_indices = torch.where(slice_mask == 1)[2]

    assert len(ones_indices) > num_to_drop

    rand_perm = torch.randperm(len(ones_indices), device=device, generator=generator)

    drop_indices = ones_indices[rand_perm[:num_to_drop]]

    new_mask[:, :, drop_indices, :] = 0

    return new_mask