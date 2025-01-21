from typing import Optional

from diffusers.models.attention_processor import Attention
from torch import nn, Tensor

from diffusion.nn import ResnetTimeScaleShiftType
from diffusion.nn.resnet import ResnetBlock


class MidBlock(nn.Module):


    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        memb_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: ResnetTimeScaleShiftType = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        attn_groups: Optional[int] = None,
        add_attention: bool = True,
        attention_head_dim: int = 1,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        if attn_groups is None:
            attn_groups = resnet_groups if resnet_time_scale_shift == "default" else None

        # there is always at least one resnet
        if resnet_time_scale_shift == "spatial":
            raise NotImplementedError("Spatial attention is not implemented yet.")
            # resnets = [
            #     ResnetBlockCondNorm2D(
            #         in_channels=in_channels,
            #         out_channels=in_channels,
            #         temb_channels=temb_channels,
            #         eps=resnet_eps,
            #         groups=resnet_groups,
            #         dropout=dropout,
            #         time_embedding_norm="spatial",
            #         non_linearity=resnet_act_fn,
            #         output_scale_factor=output_scale_factor,
            #     )
            # ]
        else:
            resnets = [
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
                )
            ]
        attentions = []

        if attention_head_dim is None:
            print(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
            )
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        out_channels,
                        heads=out_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=1.0,
                        eps=resnet_eps,
                        norm_num_groups=attn_groups,
                        spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            if resnet_time_scale_shift == "spatial":
                raise NotImplementedError("Spatial attention is not implemented yet.")
                # resnets.append(
                #     ResnetBlockCondNorm2D(
                #         in_channels=in_channels,
                #         out_channels=in_channels,
                #         temb_channels=temb_channels,
                #         eps=resnet_eps,
                #         groups=resnet_groups,
                #         dropout=dropout,
                #         time_embedding_norm="spatial",
                #         non_linearity=resnet_act_fn,
                #         output_scale_factor=output_scale_factor,
                #     )
                # )
            else:
                resnets.append(
                    ResnetBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        memb_channels=memb_channels,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                    )
                )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self,
        hidden_states: Tensor,
        temb: Optional[Tensor] = None,
        memb: Optional[Tensor] = None,
    ) -> Tensor:

        hidden_states = self.resnets[0](hidden_states, temb, memb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states, temb=temb)
            hidden_states = resnet(hidden_states, temb, memb)

        return hidden_states