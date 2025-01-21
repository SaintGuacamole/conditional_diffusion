"""
FreeSeed Mask Unet

Concatenates downsampled mask to each layer
"""

from dataclasses import dataclass
from typing import Optional, Union, Tuple

import torch
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput
from torch import nn, Tensor

from diffusion.nn import EmbeddingType, DownBlockType, UpBlockType, UpSampleMode, ResnetTimeScaleShiftType, get_embedding, \
    get_down_block, get_up_block, MidBlock


@dataclass
class UNetOutput(BaseOutput):

    sample: Tensor


class MaskUNet(ModelMixin, ConfigMixin):


    @register_to_config
    def __init__(
            self,
            *,
            sample_size: Optional[Union[int, Tuple[int, int]]] = None,
            in_channels: int = 1,
            out_channels: int = 1,
            embeddings: Tuple[EmbeddingType | str, ...] = (),
            embedding_dim: Optional[int] = None,
            num_embeddings: Union[int, Tuple[int, ...], None] = None,
            freq_shift: Union[int, Tuple[int, ...]] = 0,
            flip_sin_to_cos: bool = True,
            down_block_types: Tuple[DownBlockType | str, ...] = ("ResnetDownBlock", "ResnetDownBlock", "AttnDownBlock", "AttnDownBlock"),
            up_block_types: Tuple[UpBlockType | str, ...] = ("AttnUpBlock", "AttnUpBlock", "ResnetUpBlock", "ResnetUpBlock"),
            block_out_channels: Tuple[int | str, ...] = (64, 128, 256, 512),
            layers_per_block: int = 2,
            up_sample_mode: UpSampleMode = "nearest",
            up_sample_use_conv: bool = True,
            dropout: float = 0.0,
            act_fn: str = "silu",
            attention_head_dim: Optional[int] = 8,
            norm_num_groups: int = 32,
            attn_norm_num_groups: Optional[int] = None,
            norm_eps: float = 1e-5,
            resnet_time_scale_shift: ResnetTimeScaleShiftType = "default",
            add_attention: bool = True,
            mid_block_num_layers: int = 1,
    ):
        super().__init__()

        assert isinstance(embeddings, tuple) or embedding_dim is None

        self.sample_size = sample_size
        if embedding_dim is None and len(embeddings) > 0:
            embedding_dim = block_out_channels[0] * 4

        if isinstance(freq_shift, int):
            freq_shift = (freq_shift, ) * len(embeddings)
        if not isinstance(num_embeddings, Tuple):
            num_embeddings = (num_embeddings, ) * len(embeddings)

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # embedding
        self.embedders = nn.ModuleList([])
        for i, embedding_type in enumerate(embeddings):

            embedding = get_embedding(
                embedding_type=embedding_type,
                embedding_dim=embedding_dim,
                freq_shift=freq_shift[i],
                flip_sin_to_cos=flip_sin_to_cos,
                num_embeddings=num_embeddings[i]
            )
            self.embedders.append(embedding)

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])
        self.mask_pool = nn.ModuleList([])
        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=embedding_dim,
                do_down_sample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                down_sample_mode="conv",
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                resnet_time_scale_shift=resnet_time_scale_shift,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)
            self.mask_pool.append(nn.AvgPool2d(kernel_size=2, stride=2))

        # mid
        self.mid_block = MidBlock(
            in_channels=block_out_channels[-1],
            temb_channels=embedding_dim,
            num_layers=mid_block_num_layers,
            dropout=dropout,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_head_dim=attention_head_dim if attention_head_dim is not None else block_out_channels[-1],
            resnet_groups=norm_num_groups,
            attn_groups=attn_norm_num_groups,
            add_attention=add_attention,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=embedding_dim,
                do_up_sample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                resnet_time_scale_shift=resnet_time_scale_shift,
                up_sample_mode=up_sample_mode,
                up_sample_use_conv=up_sample_use_conv,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)

        # out
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        sample: Tensor,
        mask: Tensor,
        embeddings: Optional[Tuple[Union[Tensor, int, float]]] = (),
        return_dict: bool = True,
    ) -> Union[UNetOutput, Tuple]:

        assert isinstance(embeddings, tuple) or embeddings is None

        if not len(embeddings) == len(self.embedders):
            raise ValueError(f"This model is configered to use {len(self.embedders)} embeddings, but {len(embeddings)} were given.")

        # 1. embedding
        embedding_sum = 0
        for i, embedding in enumerate(embeddings):
            if not torch.is_tensor(embedding):
                embedding = torch.tensor([embedding], dtype=torch.long, device=sample.device)
            elif torch.is_tensor(embedding) and len(embedding.shape) == 0:
                embedding = embedding[None].to(sample.device)

            if embedding.shape[0] != sample.shape[0]:
                embedding = embedding * torch.ones(sample.shape[0], dtype=embedding.dtype, device=embedding.device)
            tmp = self.embedders[i](embedding)


            embedding_sum += tmp

        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for i, downsample_block in enumerate(self.down_blocks):
            sample[:, -1, :, :] = mask
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=embedding_sum, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=embedding_sum)

            down_block_res_samples += res_samples

            mask = self.mask_pool[i](mask)

        # 4. mid
        sample = self.mid_block(sample, embedding_sum)

        # 5. up
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, embedding_sum, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, embedding_sum)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample


        if not return_dict:
            return (sample,)

        return UNetOutput(sample=sample)


if __name__ == "__main__":

    import lovely_tensors
    lovely_tensors.monkey_patch()
    model = MaskUNet(
        down_block_types=("ResnetDownBlock", "ResnetDownBlock"),
        up_block_types=("ResnetUpBlock", "ResnetUpBlock"),
        block_out_channels=(64, 128),
    )

    inp = torch.randn((1, 1, 32, 32))

    mask = torch.randn_like(inp)

    out = model(inp, mask).sample

    print(out.shape)