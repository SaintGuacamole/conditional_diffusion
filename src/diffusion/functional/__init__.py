

import torch.nn.functional as F
from torch import Tensor


def normalize(x: Tensor):

    return (x - x.min()) / (x.max() - x.min())


def complex_mse(
    y_pred: Tensor,
    y_true: Tensor,
    reduction: str = "mean",
) -> Tensor:
    real_loss = F.mse_loss(y_pred.real, y_true.real, reduction=reduction)
    imag_loss = F.mse_loss(y_pred.imag, y_true.imag, reduction=reduction)
    return real_loss + imag_loss

from .batch_slice_mask import *
from .metrics import *
from .total_variation import *
from .classifier_free_guidance import *