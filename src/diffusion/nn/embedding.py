from typing import Optional, Tuple

import torch
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from torch import nn

from diffusion.nn import EmbeddingType


class MaskEmbedding(nn.Module):
    def __init__(
            self,
            mask_length: int,
            embedding_dim: int,
            pooling: Tuple[int, ...] = (2, 4, 8, 16, 32),
            mlp_factor: int = 2
    ):

        super().__init__()


        self.pooling_scales = [
            (i, i) for i in pooling
        ]

        self.max_pooling_layers = nn.ModuleList([
            nn.MaxPool1d(kernel_size=k, stride=s)
            for k, s in self.pooling_scales
        ])

        total_pooled_length = sum([
            mask_length // stride
            for _, stride in self.pooling_scales
        ])

        self.layers = nn.Sequential(
            nn.Linear(total_pooled_length + mask_length, embedding_dim * mlp_factor),
            nn.LayerNorm(embedding_dim * mlp_factor),
            nn.SiLU(),
            nn.Linear(embedding_dim * mlp_factor, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x):

        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) != 3:
            raise ValueError("Input should be 2d or 3d")

        max_pooled_features = []

        for max_pool in self.max_pooling_layers:

            max_features = max_pool(x).squeeze(1)
            max_pooled_features.append(max_features)

        # Combine all pooled features
        pooled_features = torch.cat(max_pooled_features, dim=1)

        combined_features = torch.cat([pooled_features, x.squeeze(1)], dim=1)

        return self.layers(combined_features)


def get_embedding(
    embedding_type: EmbeddingType,
    embedding_dim: int,
    freq_shift: float,
    flip_sin_to_cos: bool = False,
    num_embeddings: Optional[int] = None,
):

    match embedding_type:
        case "positional":
            return nn.Sequential(
                Timesteps(embedding_dim, flip_sin_to_cos, freq_shift),
                TimestepEmbedding(embedding_dim, embedding_dim),
            )
        case "learned":
            if num_embeddings is None:
                raise ValueError(f"Num embeddings can't be None for {embedding_type}")
            return nn.Sequential(
                nn.Embedding(num_embeddings, embedding_dim),
                TimestepEmbedding(embedding_dim, embedding_dim),
            )
        case "identity":
            return nn.Identity(embedding_dim, embedding_dim)

        case "vanilla":
            if num_embeddings is None:
                raise ValueError(f"Num embeddings can't be None for {embedding_type}")
            return nn.Embedding(num_embeddings, embedding_dim)

        case "mask":
            assert num_embeddings is not None, "Num embeddings can't be None for mask embedding"
            return MaskEmbedding(
                mask_length=num_embeddings,
                embedding_dim=embedding_dim
            )
        case _:
            raise NotImplementedError(f"Unknown embedding type {embedding_type}")
