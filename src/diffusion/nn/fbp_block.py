from typing import Optional, cast, Tuple

from diffusers.models.attention_processor import Attention
from torch import nn, Tensor

from diffusion.nn import ResnetTimeScaleShiftType, DownSampleMode
from diffusion.nn.downsample import Downsample
from diffusion.nn.resnet import ResnetBlock
from diffusion.nn.leap_projector_wrapper import Projector


class FBPDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: ResnetTimeScaleShiftType = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        down_sample_mode: Optional[DownSampleMode] = "avg",
        do_down_sample: bool = True,
        do_freq_conv: bool = False,
        project_output: bool = False,
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
            if not project_output:
                self.projected_down_sample = Downsample(
                    out_channels, out_channels=out_channels, mode=down_sample_mode
                )

        self.out_channels = out_channels
        self.projector = None
        self.project_output = project_output

    def forward(
        self,
        hidden_states: Tensor,
        temb: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:

        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            projected_hidden_states = self.projector.fbp_channel_wise_scaled(hidden_states)
            output_states = output_states + (projected_hidden_states,)

        if self.down_sample is not None:
            if self.project_output:
                hidden_states = self.projector.fbp_channel_wise_scaled(hidden_states)
                hidden_states = self.down_sample(hidden_states)
            else:
                projected_hidden_states = self.projector.fbp_channel_wise_scaled(hidden_states.clone())
                hidden_states = self.down_sample(hidden_states)
                projected_hidden_states = self.projected_down_sample(projected_hidden_states)
                output_states = output_states + (projected_hidden_states,)

        elif self.project_output:
            hidden_states = self.projector.fbp_channel_wise_scaled(hidden_states)

        return hidden_states, output_states

    def set_projector(
            self,
            batch_size: int,
            detector_width: int,
            nr_angles: int,
            device: str = "cpu"
    ):
        self.projector = Projector(
            batch_size = batch_size * self.out_channels,
            nr_angles=nr_angles,
            cols=detector_width,
            device=device,
            forward_project=False,
            use_static=False,
        )


class AttnFBPDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
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
        project_output: bool = False,
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
            if not project_output:
                self.projected_down_sample = Downsample(
                    out_channels, out_channels=out_channels, mode="conv"
                )
        self.projector = None
        self.out_channels = out_channels
        self.project_output = project_output

    def set_projector(
            self,
            batch_size: int,
            detector_width: int,
            nr_angles: int,
            device: str = "cpu"
    ):
        self.projector = Projector(
            batch_size = batch_size * self.out_channels,
            nr_angles=nr_angles,
            cols=detector_width,
            device=device,
            forward_project=False,
            use_static=False,
        )

    def forward(
        self,
        hidden_states: Tensor,
        temb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:

        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)
            projected_hidden_states = self.projector.fbp_channel_wise_scaled(hidden_states)
            output_states = output_states + (projected_hidden_states,)

        if self.down_sample is not None:
            if self.project_output:
                hidden_states = self.projector.fbp_channel_wise_scaled(hidden_states)
                hidden_states = self.down_sample(hidden_states)
            else:
                projected_hidden_states = self.projector.fbp_channel_wise_scaled(hidden_states)
                hidden_states = self.down_sample(hidden_states)
                projected_hidden_states = self.projected_down_sample(projected_hidden_states)
                output_states += (projected_hidden_states,)

        elif self.project_output:
            hidden_states = self.projector.fbp_channel_wise_scaled(hidden_states)

        return hidden_states, output_states

if __name__ == "__main__":
    import torch
    import lovely_tensors
    lovely_tensors.monkey_patch()

    block = FBPDownBlock(
        in_channels=64,
        out_channels=128,
        temb_channels=512,
        dropout=0.0,
        num_layers=4,
        resnet_eps=1e-6,
        resnet_time_scale_shift="default",
        resnet_act_fn="swish",
    )

    block.set_projector(
        batch_size=2,
        detector_width=128,
        nr_angles=180,
        device="cpu"
    )

    inp = torch.randn((2, 64, 180, 128))

    temp = torch.randn((2, 512))
    print(inp)
    hidden, res = block(inp, temp)
    print(hidden, res)
