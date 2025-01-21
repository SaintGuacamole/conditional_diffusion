from typing import Optional, Tuple, cast

import torch
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.activations import get_activation
from torch import nn, Tensor

from diffusion.nn import ResnetTimeScaleShiftType, DownSampleMode, UpSampleMode
from diffusion.nn.upsample import Upsample
from diffusion.nn.downsample import Downsample
from diffusion.nn.frequency_convolution import FrequencyConvolution2d


class ResnetBlock(nn.Module):

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        temb_channels: int = 512,
        memb_channels: Optional[int] = None,
        groups: int = 32,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        time_embedding_norm: ResnetTimeScaleShiftType = "default",
        up: bool = False,
        down: bool = False,
        do_freq_conv: bool = False,
        output_scale_factor: float = 1.
    ):
        super().__init__()

        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.time_embedding_norm = time_embedding_norm
        self.do_freq_conv = do_freq_conv
        self.output_scale_factor = output_scale_factor
        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        if do_freq_conv:
            self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=1, padding=1)
            self.freq_conv = FrequencyConvolution2d(in_channels, out_channels // 2, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = nn.Linear(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = nn.Linear(temb_channels, 2 * out_channels)
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")
        else:
            self.time_emb_proj = None

        if memb_channels is not None:
            self.mask_emb_proj = nn.Linear(memb_channels, out_channels)
        else:
            self.mask_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.non_linearity = get_activation(non_linearity)

        self.up_sample = self.down_sample = None

        if self.up:
            self.up_sample = Upsample(in_channels)
        elif self.down:
            self.down_sample = Downsample(in_channels)

        self.conv_shortcut = None
        if self.in_channels != self.out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(
        self,
        x: Tensor,
        temb: Tensor = None,
        memb: Tensor = None,
    ) -> torch.Tensor:

        hidden_states = x

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.non_linearity(hidden_states)

        if self.up_sample is not None:
            if hidden_states.shape[0] >= 64:
                x = x.contiguous()
                hidden_states = hidden_states.contiguous()
            x = self.up_sample(x)
            hidden_states = self.up_sample(hidden_states)
        elif self.down_sample is not None:
            x = self.down_sample(x)
            hidden_states = self.down_sample(hidden_states)

        if self.do_freq_conv:
            hidden_states_image = self.conv1(hidden_states)

            freq_hidden_states = self.freq_conv(hidden_states)

            hidden_states = torch.cat([hidden_states_image, freq_hidden_states], dim=1)

        else:
            hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            temb = self.non_linearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if self.mask_emb_proj is not None:
            memb = self.non_linearity(memb)
            memb = self.mask_emb_proj(memb)[:, :, None, None]

            hidden_states = hidden_states + memb

        match self.time_embedding_norm:
            case "default":
                if temb is not None:
                    hidden_states = hidden_states + temb
                hidden_states = self.norm2(hidden_states)

            case "scale_shift":
                if temb is None:
                    raise ValueError(
                        f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
                    )
                time_scale, time_shift = torch.chunk(temb, 2, dim=1)
                hidden_states = self.norm2(hidden_states)
                hidden_states = hidden_states * (1 + time_scale) + time_shift
            case _:
                hidden_states = self.norm2(hidden_states)


        hidden_states = self.non_linearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        output_tensor = (x + hidden_states) / self.output_scale_factor

        return output_tensor


class ResnetDownBlock(ModelMixin, ConfigMixin):
    @register_to_config
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
        down_sample_mode: Optional[DownSampleMode] = "avg",
        do_down_sample: bool = True,
        do_freq_conv: bool = False,
    ):
        super().__init__()
        if down_sample_mode is None:
            down_sample_mode = cast(DownSampleMode, "avg")
        resnets = []

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

        self.resnets = nn.ModuleList(resnets)

        self.down_sample = None
        if do_down_sample:
            self.down_sample = Downsample(
                out_channels, out_channels=out_channels, mode=down_sample_mode
            )


    def forward(
        self,
        hidden_states: Tensor,
        temb: Optional[Tensor] = None,
        memb: Optional[torch.Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:

        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb, memb)
            output_states = output_states + (hidden_states,)

        if self.down_sample is not None:
            hidden_states = self.down_sample(hidden_states)
            output_states = output_states + (hidden_states,)

        return hidden_states, output_states



class ResnetUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        memb_channels: Optional[int] = None,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: ResnetTimeScaleShiftType = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        up_sample_mode: Optional[UpSampleMode] = "nearest",
        do_up_sample: bool = True,
        do_freq_conv: bool = False,
    ):
        super().__init__()
        if up_sample_mode is None:
            up_sample_mode = cast(UpSampleMode, "nearest")
        resnets = []

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

        self.resnets = nn.ModuleList(resnets)

        self.up_sample = None
        if do_up_sample:
            self.up_sample = Upsample(out_channels, out_channels=out_channels, mode=up_sample_mode)

        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...] = None,
        temb: Optional[torch.Tensor] = None,
        upsample_size: Optional[int] = None,
        memb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        for resnet in self.resnets:

            if res_hidden_states_tuple is not None:
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb, memb)

        if self.up_sample is not None:
            hidden_states = self.up_sample(hidden_states, upsample_size)

        return hidden_states


class DownEncoderBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: ResnetTimeScaleShiftType = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            do_down_sample: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    memb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        self.down_sample = None
        if do_down_sample:
            self.down_sample = Downsample(
                out_channels, out_channels=out_channels, mode="conv"
            )


    def forward(self, hidden_states: Tensor) -> Tensor:

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None, memb=None)

        if self.down_sample is not None:

            hidden_states = self.down_sample(hidden_states)

        return hidden_states



class UpDecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: ResnetTimeScaleShiftType = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            do_up_sample: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        self.up_sample = None
        if do_up_sample:
            self.up_sample = Upsample(out_channels, out_channels=out_channels, conv=True)



    def forward(self, hidden_states: Tensor) -> Tensor:

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)

        if self.up_sample is not None:

            hidden_states = self.up_sample(hidden_states)

        return hidden_states