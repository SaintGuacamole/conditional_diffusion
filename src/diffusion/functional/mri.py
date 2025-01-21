import logging
from typing import Optional

import numpy
import numpy as np
import torch
from torch import Tensor


def gaussian_probabilities(
    x: int,
    sigma: Optional[float] = None,
) -> Tensor:
    if sigma is None:
        sigma = x / 4

    # Create Gaussian centered at middle
    mean = x // 2
    x_coords = np.arange(x)

    g = np.exp(-0.5 * ((x_coords - mean) / sigma) ** 2)
    g = g / g.sum()

    return torch.from_numpy(g).float()


def gaussian_2d_probabilities(
    height: int,
    width: int,
    sigma_h: Optional[float] = None,
    sigma_w: Optional[float] = None,
)-> Tensor:

    if sigma_h is None:
        sigma_h = height / 4
    if sigma_w is None:
        sigma_w = width / 4

    y = np.arange(-height // 2, height // 2) if height % 2 == 0 else np.arange(-(height // 2), height // 2 + 1)
    x = np.arange(-width // 2, width // 2) if width % 2 == 0 else np.arange(-(width // 2), width // 2 + 1)
    xs, ys = np.meshgrid(x, y)

    # Create 2D Gaussian
    g = np.exp(-0.5 * ((ys / sigma_h) ** 2 + (xs / sigma_w) ** 2))

    # Normalize
    g = g / g.sum()

    return torch.from_numpy(g).float()

def cartesian_k_space_mask(
    batch: int,
    channels: int,
    height: int,
    width: int,
    generator: numpy.random.Generator,
    keep_n_cols: Tensor,
    device: torch.device,
    center_width: int = 2,
    probabilities: Optional[Tensor] = None,
):
    assert keep_n_cols.shape == (batch,)

    probabilities = probabilities or gaussian_probabilities(height)

    mid = width // 2

    probabilities[mid - center_width:mid + center_width + 1] = 0
    probabilities[:mid] = 0

    probabilities = probabilities / probabilities.sum()

    n_center_cols = 2 * center_width + 1

    available_cols = np.where(probabilities.cpu().numpy() > 0)[0]
    available_probabilities = probabilities[available_cols].cpu().numpy() / probabilities[available_cols].cpu().numpy().sum()

    mask = torch.zeros((batch, channels, height, width), device=device)
    mask[:, :, :, mid - center_width:mid + center_width + 1] = 1.
    for idx in range(batch):
        n_sample_cols = ((keep_n_cols[idx] - n_center_cols) // 2).item()  # Half the remaining columns (rounded down)

        if n_sample_cols >= 0:
            if n_sample_cols > available_cols.shape[0]:
                n_sample_cols = available_cols.shape[0]
                logging.info(
                    f"Trying to sample more columns than available. Sampling {n_sample_cols} columns."
                )
            indices = generator.choice(
                a=available_cols,
                size=n_sample_cols,
                replace=False,
                p=available_probabilities,
            )

            mask[idx, :, :, indices] = 1.

            sym_cols = (2 * mid - indices) % width
            mask[idx, :, :, sym_cols] = 1.
        else:
            logging.warning(
                f"Trying to create a mask that is smaller than the fixed center radius."
                f"Attempted to sample {n_sample_cols} columns."
            )
    return mask


def circle_pixel_count(radius: int) -> int:
    if radius <= 0:
        return 0

    count = 1
    for x in range(1, radius + 1):
        y = int((radius * radius - x * x) ** 0.5)
        count += y * 4
        count += 4

    return count

def create_circle_mask(height: int, width: int, radius: int) -> np.ndarray:

    y = np.arange(-height//2, height//2) if height % 2 == 0 else np.arange(-(height//2), height//2 + 1)
    x = np.arange(-width//2, width//2) if width % 2 == 0 else np.arange(-(width//2), width//2 + 1)
    xs, ys = np.meshgrid(x, y)
    return xs**2 + ys**2 <= radius**2


def get_unique_points(height: int, width: int) -> np.ndarray:
    y = np.arange(-height // 2, height // 2) if height % 2 == 0 else np.arange(-(height // 2), height // 2 + 1)
    x = np.arange(-width // 2, width // 2) if width % 2 == 0 else np.arange(-(width // 2), width // 2 + 1)
    xs, ys = np.meshgrid(x, y)

    unique_mask = ys > 0
    center_line = (ys == 0) & (xs >= 0)

    return unique_mask | center_line


def gaussian_k_space_mask(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    generator: numpy.random.Generator,
    keep_n_pixels: Tensor,
    device: torch.device,
    center_radius: int = 2,
    probabilities: Optional[Tensor] = None,
):
    assert keep_n_pixels.shape == (batch_size,)

    if probabilities is None:
        probabilities = gaussian_2d_probabilities(height, width)

    unique_points = torch.from_numpy(get_unique_points(height, width)).to(device)
    center_mask = torch.from_numpy(create_circle_mask(height, width, center_radius)).to(device)

    probabilities[~unique_points] = 0
    probabilities[center_mask] = 0
    probabilities = probabilities / probabilities.sum()

    mask = torch.zeros((batch_size, channels, height, width), device=device)
    mask[:, :, center_mask] = 1.

    flat_probs = probabilities.flatten()
    flat_probs_np = flat_probs.cpu().numpy()

    mask_n_pixels = torch.sum(center_mask).item()
    available_indices = np.where(flat_probs_np > 0)[0]
    flat_probs_available = flat_probs_np[available_indices] / flat_probs_np[available_indices].sum()

    for idx in range(batch_size):

        total_points = (keep_n_pixels[idx] - mask_n_pixels).item()

        if total_points >= 0:
            if total_points > available_indices.shape[0]:
                total_points = available_indices.shape[0]
                logging.info(
                    f"Trying to sample more columns than available. Sampling {total_points} columns."
                )
            chosen_indices = generator.choice(
                a=available_indices,
                size=int(total_points),
                replace=False,
                p=flat_probs_available
            )

            mask[idx, :, (chosen_indices // width), (chosen_indices % width)] = 1.

            y = (chosen_indices // width) - height // 2
            x = (chosen_indices % width) - width // 2

            sym_y = (-y + height // 2) % height
            sym_x = (-x + width // 2) % width

            mask[idx, :, sym_y, sym_x] = 1.
        else:
            logging.warning(
                f"Trying to create a mask that is smaller than the fixed center radius."
                f"Keeping {mask_n_pixels} center pixels, no additional pixels were sampled."
            )

    return mask


def random_k_space_mask(
    b: int,
    c: int,
    h: int,
    w: int,
    keep_n_pixels: Tensor,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
):
    assert keep_n_pixels.shape == (b,)
    assert c == 1
    mask = torch.zeros((b, c, h, w), device=device)

    for idx in range(b):
        pixels = torch.randperm(h * w, generator=generator)[:keep_n_pixels[idx]]
        y_indices = pixels // w
        x_indices = pixels % w
        mask[idx, :, y_indices, x_indices] = 1.

    return mask