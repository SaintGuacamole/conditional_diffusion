
"""
Majority of this code is again from diffusers / hugginface

including the resnet blocks, up and down sample blocks, and embeddings
"""



from enum import Enum
from typing import Literal, Optional

class NormalizationType(Enum):
    NONE = 0
    ZERO_TO_ONE = 1
    MINUS_ONE_TO_ONE = 2

UpBlockType = Literal["ResnetUpBlock", "AttnUpBlock", "FourierResnetUpBlock", "FourierAttnUpBlock", "UpDecoderBlock"]
DownBlockType = Literal["ResnetDownBlock", "AttnDownBlock", "FourierResnetDownBlock", "FourierAttnDownBlock", "DownEncoderBlock", "ResnetFBPDownBlock", "AttnFBPDownBlock"]
UpSampleMode = Literal["nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"]
DownSampleMode = Literal["avg", "max", "conv"]
ResnetTimeScaleShiftType = Literal["default", "scale_shift"]
EmbeddingType = Literal["positional", "fourier", "learned", "identity", "vanilla", "mask"]



def get_up_block(
    block_type: UpBlockType,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    prev_output_channel: int,
    temb_channels: int,
    do_up_sample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    memb_channels: Optional[int] = None,
    up_sample_mode: Optional[UpSampleMode] = None,
    up_sample_use_conv: bool = True,
    resolution_idx: Optional[int] = None,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: ResnetTimeScaleShiftType = "default",
    attention_type: str = "default",
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = None,
    up_sample_type: Optional[str] = None,
    dropout: float = 0.0,
):

    match block_type:
        case "ResnetUpBlock":

            return ResnetUpBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                prev_output_channel=prev_output_channel,
                temb_channels=temb_channels,
                memb_channels=memb_channels,
                resolution_idx=resolution_idx,
                dropout=dropout,
                num_layers=num_layers,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                resnet_groups=resnet_groups,
                resnet_time_scale_shift=resnet_time_scale_shift,
                up_sample_mode=up_sample_mode,
                do_up_sample=do_up_sample,
                do_freq_conv=False,
            )

        case "AttnUpBlock":
            return AttnUpBlock(
                in_channels=in_channels,
                prev_output_channel=prev_output_channel,
                out_channels=out_channels,
                temb_channels=temb_channels,
                memb_channels=memb_channels,
                dropout=dropout,
                num_layers=num_layers,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                resnet_groups=resnet_groups,
                attention_head_dim=attention_head_dim,
                resnet_time_scale_shift=resnet_time_scale_shift,
                do_up_sample=do_up_sample,
                do_freq_conv=False,
            )

        case "FourierAttnUpBlock":
            return AttnUpBlock(
                in_channels=in_channels,
                prev_output_channel=prev_output_channel,
                out_channels=out_channels,
                temb_channels=temb_channels,
                memb_channels=memb_channels,
                dropout=dropout,
                num_layers=num_layers,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                resnet_groups=resnet_groups,
                attention_head_dim=attention_head_dim,
                resnet_time_scale_shift=resnet_time_scale_shift,
                do_up_sample=do_up_sample,
                do_freq_conv=True,
            )

        case "FourierResnetUpBlock":

            return ResnetUpBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                prev_output_channel=prev_output_channel,
                temb_channels=temb_channels,
                memb_channels=memb_channels,
                resolution_idx=resolution_idx,
                dropout=dropout,
                num_layers=num_layers,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                resnet_groups=resnet_groups,
                resnet_time_scale_shift=resnet_time_scale_shift,
                up_sample_mode=up_sample_mode,
                do_up_sample=do_up_sample,
                do_freq_conv=True,
            )

        case "UpDecoderBlock":

            return UpDecoderBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout=dropout,
                num_layers=num_layers,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                resnet_groups=resnet_groups,
                resnet_time_scale_shift=resnet_time_scale_shift,
                do_up_sample=do_up_sample,
            )


        case _:
            raise NotImplementedError(f"Unsupported up block type: {block_type}")


def get_down_block(
    block_type: DownBlockType,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    do_down_sample: bool,
    resnet_eps: float,
    resnet_act_fn: str,
    memb_channels: Optional[int] = None,
    down_sample_mode: Optional[DownSampleMode] = None,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    down_sample_padding: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: ResnetTimeScaleShiftType = "default",
    attention_type: str = "default",
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = None,
    down_sample_type: Optional[str] = None,
    dropout: float = 0.0,
    project_output: bool = False,
):

    match block_type:
        case "ResnetDownBlock":

            return ResnetDownBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                memb_channels=memb_channels,
                dropout=dropout,
                num_layers=num_layers,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                resnet_groups=resnet_groups,
                resnet_time_scale_shift=resnet_time_scale_shift,
                down_sample_mode=down_sample_mode,
                do_down_sample=do_down_sample,
                do_freq_conv=False,
            )

        case "AttnDownBlock":

            return AttnDownBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                memb_channels=memb_channels,
                dropout=dropout,
                num_layers=num_layers,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                resnet_groups=resnet_groups,
                attention_head_dim=attention_head_dim,
                resnet_time_scale_shift=resnet_time_scale_shift,
                do_down_sample=do_down_sample,
                do_freq_conv=False,
            )

        case "FourierAttnDownBlock":

            return AttnDownBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                memb_channels=memb_channels,
                dropout=dropout,
                num_layers=num_layers,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                resnet_groups=resnet_groups,
                attention_head_dim=attention_head_dim,
                resnet_time_scale_shift=resnet_time_scale_shift,
                do_down_sample=do_down_sample,
                do_freq_conv=True,
            )

        case "FourierResnetDownBlock":

            return ResnetDownBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                memb_channels=memb_channels,
                dropout=dropout,
                num_layers=num_layers,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                resnet_groups=resnet_groups,
                resnet_time_scale_shift=resnet_time_scale_shift,
                down_sample_mode=down_sample_mode,
                do_down_sample=do_down_sample,
                do_freq_conv=True,
            )

        case "DownEncoderBlock":
            return DownEncoderBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout=dropout,
                num_layers=num_layers,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                resnet_groups=resnet_groups,
                resnet_time_scale_shift=resnet_time_scale_shift,
                do_down_sample=do_down_sample,
            )

        case "ResnetFBPDownBlock":
            return FBPDownBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                # memb_channels=memb_channels,
                dropout=dropout,
                num_layers=num_layers,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                resnet_groups=resnet_groups,
                resnet_time_scale_shift=resnet_time_scale_shift,
                down_sample_mode=down_sample_mode,
                do_down_sample=do_down_sample,
                do_freq_conv=False,
                project_output=project_output
            )
        case "AttnFBPDownBlock":
            return AttnFBPDownBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                dropout=dropout,
                num_layers=num_layers,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                resnet_groups=resnet_groups,
                attention_head_dim=attention_head_dim,
                resnet_time_scale_shift=resnet_time_scale_shift,
                do_down_sample=do_down_sample,
                project_output=project_output
            )
        case _:
            raise NotImplementedError(f"Unsupported down block type: {block_type}")

from .attention import AttnUpBlock, AttnDownBlock
from .circle_mask import CircleMask
from .downsample import Downsample
from .embedding import get_embedding
from .fbp_block import FBPDownBlock, AttnFBPDownBlock
from .frequency_convolution import FrequencyConvolution
from .leap_projector_wrapper import Projector
from .mid import MidBlock
from .resnet import ResnetDownBlock, ResnetUpBlock, DownEncoderBlock, UpDecoderBlock
from .upsample import Upsample
from .positional_encodin import PositionalEncoding
from .transformer import TransformerEncoder