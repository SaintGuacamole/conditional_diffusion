from dataclasses import dataclass
from typing import Tuple

import torch.nn.functional
from torch import Tensor
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

def float_psnr(
    x: Tensor,
    y: Tensor,
    reduce: bool = False,
) -> Tensor:
    assert x.size() == y.size(), "Input tensors must have the same shape."
    assert torch.all(x >= 0) and torch.all(x <= 1), "Tensor x must be in the range [0, 1]."
    assert torch.all(y >= 0) and torch.all(y <= 1), "Tensor y must be in the range [0, 1]."
    assert len(x.size()) == len(y.size()) == 4, "Input tensors must be 4D."

    mse = torch.mean((x - y) ** 2, dim=(-3, -2, -1))

    mse = torch.where(mse == 0., torch.tensor(1e-8), mse)

    psnr = 10 * torch.log10(1. / mse)

    if reduce:
        return psnr.mean()

    return psnr

@dataclass
class MetricOutput:
    psnr: Tensor
    ssim: Tensor
    mse: Tensor
    fid: Tensor


class Metrics:

    def __init__(
        self,
    ):
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction=None)
        # self.fid = FrechetInceptionDistance(input_img_size=size, normalize=True)

    def __call__(
        self,
        x: Tensor,
        y: Tensor,
        detach: bool = True
    ):
        _psnr = float_psnr(x, y)

        _ssim = self.ssim(x, y)
        if len(_ssim.shape) == 0:
            _ssim = _ssim.unsqueeze(0)

        _mse = torch.mean((x - y) ** 2, dim=(1, 2, 3))

        # _fid = self.fid(x, y)
        # if len(_fid.shape) == 0:
        #     _fid = _fid.unsqueeze(0)
        _fid = torch.zeros(0)
        if detach:
            return MetricOutput(
                psnr=_psnr.detach().cpu(),
                ssim=_ssim.detach().cpu(),
                mse=_mse.detach().cpu(),
                fid=_fid.detach().cpu()
            )
        else:
            return MetricOutput(
                psnr=_psnr,
                ssim=_ssim,
                mse=_mse,
                fid=_fid
            )



