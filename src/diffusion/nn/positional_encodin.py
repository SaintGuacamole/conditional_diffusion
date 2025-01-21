import math
from typing import Callable, Optional

import torch
from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    def __init__(
            self,
            *,
            d_model,
            max_len=5000,
            index_transform: Callable[[Tensor], Tensor] = lambda x: x.long(),
            squeeze_dim: Optional[int] = 1,
    ):
        super(PositionalEncoding, self).__init__()

        self.index_transform = index_transform

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))
        self.squeeze_dim = squeeze_dim

    def forward(
            self,
            x,
            *,
            indices: Optional[torch.Tensor] = None,
    ):
        if indices is None:
            indices = torch.arange(x.size(-2))

        indices = self.index_transform(indices)

        if len(indices.size()) == 2:
            for i in range(indices.size(0)):
                x[i] = x[i] + self.pe[:, indices[i], :]
        else:
            x = x + self.pe[:, indices, :]

        if self.squeeze_dim is not None:
            return x.squeeze(self.squeeze_dim)
        else:
            return x
