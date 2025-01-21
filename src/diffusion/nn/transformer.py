import torch
from torch import nn, Tensor
from diffusion.nn.positional_encodin import PositionalEncoding

class TransformerEncoder(nn.Module):

    def __init__(
            self,
            *,
            input_dim: int,
            embedding_dim: int = 512,
            max_len: int = 6000,
            nhead: int = 8,
            dim_feedforward: int = 2048,
            dropout: float = 0.,
            num_layers: int = 1,
    ):
        super().__init__()

        self.intput_projection = nn.Linear(
            in_features=input_dim,
            out_features=embedding_dim,
        )

        self.positional_encoder = PositionalEncoding(
            d_model=embedding_dim,
            max_len=max_len+1,
            squeeze_dim=1,
        )

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)


    def forward(
            self,
            x: Tensor,
            angles: Tensor,
    ) -> Tensor:

        x = self.intput_projection(x)
        x = self.positional_encoder(x.unsqueeze(1), indices=angles)

        return self.encoder(x)


