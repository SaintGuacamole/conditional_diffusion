import torch
from torch import nn, Tensor

class GaussianNLL(nn.Module):

    def __init__(
        self,
        eps_min: float = 1e-6,
        eps_max: float = 1e3,
    ):
        super(GaussianNLL, self).__init__()
        self.eps_min = eps_min
        self.eps_max = eps_max

    def forward(
        self,
        model_output: Tensor,
        target: Tensor,
        reduction: str = "none",
    ):

        assert model_output.shape[1] % 2 == 0
        assert target.shape[1] == model_output.shape[1] / 2

        mean, log_var = model_output.chunk(2, dim=1)

        diff = mean - target

        var = torch.exp(log_var)

        var = var.clamp(self.eps_min, self.eps_max)

        loss = torch.log(var) + (diff ** 2) / var

        match reduction:
            case "none":
                return loss
            case "mean":
                return torch.mean(loss)
            case _:
                raise NotImplementedError

