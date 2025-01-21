import torch
from diffusers import ModelMixin
from diffusers.configuration_utils import register_to_config, ConfigMixin
from torch import nn

from diffusion.models.vae import Decoder
from diffusion.nn import TransformerEncoder


class TransformerAutoEncoder(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
            self,
            input_embedding_size: int,
            embedding_dim: int = 512,
            max_len: int = 6000,
            target_nangles: int = 180,
            target_width: int = 512,
            nhead: int = 8,
            dim_feedforward: int = 2048,
            dropout: float = 0.,
            num_layers: int = 8,
    ):
        super(TransformerAutoEncoder, self).__init__()

        self.transformer_encoder = TransformerEncoder(
            input_dim=input_embedding_size,
            embedding_dim=embedding_dim,
            max_len=max_len,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layers=num_layers,
        )
        self.out_projection = None
        if embedding_dim != target_width:
            self.out_projection = nn.Linear(embedding_dim, target_width)

        self.out_conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.target_nangles = target_nangles

    def forward(
            self,
            x,
            *,
            angles: torch.Tensor,
    ):
        x = self.transformer_encoder(x, angles=angles)
        if self.out_projection is not None:
            x = self.out_projection(x)
        x = x[:, :self.target_nangles]
        x = x.unsqueeze(1)
        return self.out_conv(x)


class TransformerAutoEncoderResnetDeconder(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
            self,
            input_length: int,
            embedding_dim: int = 512,
            max_len: int = 6000,
            nhead: int = 8,
            dim_feedforward: int = 2048,
            dropout: float = 0.,
            num_layers: int = 8,
    ):
        super(TransformerAutoEncoderResnetDeconder, self).__init__()

        self.transformer_encoder = TransformerEncoder(
            input_dim=input_length,
            embedding_dim=embedding_dim,
            max_len=max_len,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layers=num_layers,
        )

        self.decoder = Decoder(
            in_channels=embedding_dim,
            out_channels=1,
            block_types=("UpDecoderBlock", "UpDecoderBlock", "UpDecoderBlock", "UpDecoderBlock", "UpDecoderBlock"),
            channels=(embedding_dim, 256, 128, 64, 32),
            layers_per_block=2,
            norm_num_groups=32,
            mid_block_add_attention=True,
        )

    def forward(
            self,
            x,
            *,
            angles: torch.Tensor,
    ):
        x = self.transformer_encoder(x, angles=angles)

        x = x[:, 0]
        b, c = x.size()
        x = x.view(b, c, 1, 1)
        x = x.repeat(1, 1, 32, 32)
        return self.decoder(x)