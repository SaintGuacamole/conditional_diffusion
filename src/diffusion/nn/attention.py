from typing import Optional, Tuple

import torch
from diffusers.models.attention_processor import Attention
from torch import nn, Tensor

from diffusion.nn import ResnetTimeScaleShiftType
from diffusion.nn.resnet import ResnetBlock
from diffusion.nn.upsample import Upsample
from diffusion.nn.downsample import Downsample


class AttnDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        memb_channels: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: ResnetTimeScaleShiftType = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        do_down_sample: bool = True,
        do_freq_conv: bool = False,
    ):
        super().__init__()
        resnets = []
        attentions = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    memb_channels=memb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    do_freq_conv=do_freq_conv
                )
            )
            attentions.append(
                Attention(
                    out_channels,
                    heads=out_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.down_sample = None
        if do_down_sample:
            self.down_sample = Downsample(
                out_channels, out_channels=out_channels, mode="conv"
            )


    def forward(
        self,
        hidden_states: Tensor,
        temb: Optional[Tensor] = None,
        memb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:

        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb, memb)
            hidden_states = attn(hidden_states)
            output_states = output_states + (hidden_states,)

        if self.down_sample is not None:
            hidden_states = self.down_sample(hidden_states)
            output_states += (hidden_states,)

        return hidden_states, output_states


class AttnUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        memb_channels: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: ResnetTimeScaleShiftType = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        do_up_sample: bool = True,
        do_freq_conv: bool = False,
    ):
        super().__init__()
        resnets = []
        attentions = []


        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    memb_channels=memb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    do_freq_conv=do_freq_conv
                )
            )
            attentions.append(
                Attention(
                    out_channels,
                    heads=out_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.up_sample = None
        if do_up_sample:
            self.up_sample = Upsample(out_channels, out_channels=out_channels, conv=True)


    def forward(
        self,
        hidden_states: Tensor,
        res_hidden_states_tuple: Tuple[Tensor, ...],
        temb: Optional[Tensor] = None,
        memb: Optional[Tensor] = None,
    ) -> Tensor:

        for resnet, attn in zip(self.resnets, self.attentions):

            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb, memb)
            hidden_states = attn(hidden_states)

        if self.up_sample is not None:

            hidden_states = self.up_sample(hidden_states)

        return hidden_states