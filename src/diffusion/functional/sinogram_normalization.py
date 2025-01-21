from typing import Optional, Tuple

from torch import Tensor


def vanilla_normalize(
    x: Tensor,
    *args,
    **kwargs
):
    return (x - x.min()) / (x.max() - x.min())


def normalize_by_mask_amount(
    x: Tensor,
    mask_amount: Tensor,
    clamp: Optional[Tuple[float, float]] = (0., 1.),
    *args,
    **kwargs
):
    assert x.shape[0] == mask_amount.shape[0]

    x = x / mask_amount[:, None, None, None]

    if clamp:
        x = x.clamp(min=clamp[0], max=clamp[1])

    return x