from typing import Tuple
import torch
from torch import Tensor


def calculate_difference(x: Tensor, axis: int) -> Tensor:
    match axis:
        case 0:
            return x[:-1, :, :, :] - x[1:, :, :, :]
        case 1:
            return x[:, :-1, :, :] - x[:, 1:, :, :]
        case 2:
            return x[:, :, :-1, :] - x[:, :, 1:, :]
        case 3:
            return x[:, :, :, :-1] - x[:, :, :, 1:]
        case _:
            raise NotImplementedError


def l2_variation(x: Tensor, weight: Tuple[float, float] = (1.0, 1.0)) -> Tensor:
    h = torch.pow(calculate_difference(x, 2), 2).sum()
    w = torch.pow(calculate_difference(x, 3), 2).sum()
    return h * weight[0] + w * weight[1]


def l2_variation_scaled(x: Tensor, weight: Tuple[float, float] = (1.0, 1.0)) -> Tensor:
    tv = l2_variation(x, weight)
    return tv / (x.size(2) * x.size(3))


def l1_variation(x: Tensor, weight: Tuple[float, float] = (1.0, 1.0)) -> Tensor:
    h = torch.abs(calculate_difference(x, 2)).sum()
    w = torch.abs(calculate_difference(x, 3)).sum()
    return h * weight[0] + w * weight[1]


def l1_variation_scaled(x: Tensor, weight: Tuple[float, float] = (1.0, 1.0)) -> Tensor:
    tv = l1_variation(x, weight)
    return tv / (x.size(2) * x.size(3))
