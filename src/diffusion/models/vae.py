from typing import Tuple, Optional, Union

import torch
from diffusers import ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.utils import BaseOutput
from torch import nn, Tensor

from diffusion.nn import DownBlockType, get_down_block, ResnetTimeScaleShiftType, MidBlock, get_up_block, UpBlockType


class EncoderOutput(BaseOutput):

    latent_dist: DiagonalGaussianDistribution

class DecoderOutput(BaseOutput):

    sample: Tensor


class AutoencoderKL(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            *,
            latent_channels: int = 4,
            layers_per_block: int = 1,
            down_block_types: Tuple[DownBlockType, ...] = ("DownEncoderBlock",),
            up_block_types: Tuple[UpBlockType, ...] = ("UpDecoderBlock",),
            channels: Tuple[int, ...] = (64,),
            norm_num_groups: int = 32,
            act_fn = "silu",
            mid_block_add_attention: bool = True,
            use_quant_conv: bool = True,
            use_post_quant_conv: bool = True,
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_types=down_block_types,
            channels=channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            mid_block_add_attention=mid_block_add_attention,
        )

        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_types=up_block_types,
            channels=list(reversed(channels)),
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1) if use_quant_conv else None
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1) if use_post_quant_conv else None

    def _encode(self, x: Tensor) -> Tensor:

        enc = self.encoder(x)
        if self.quant_conv is not None:
            enc = self.quant_conv(enc)

        return enc

    def encode(
            self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[EncoderOutput, Tuple[DiagonalGaussianDistribution]]:

        h = self._encode(x)

        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return EncoderOutput(latent_dist=posterior)


    def _decode(self, z: Tensor, return_dict: bool = True) -> Union[DecoderOutput, Tensor]:

        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)

        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def decode(
            self, z: Tensor, return_dict: bool = True
    ) -> Union[DecoderOutput, Tensor]:

        decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)


    def forward(
            self,
            x: Tensor,
            sample_posterior: bool = False,
            return_dict: bool = True,
            generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:

        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

class Encoder(nn.Module):

    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            *,
            channels: Tuple[int, ...] = (64, ),
            layers_per_block: int = 3,
            block_types: Tuple[DownBlockType | str, ...] = ("DownEncoderBlock",),
            act_fn: str = "silu",
            norm_eps: float = 1e-6,
            norm_num_groups: int = 32,
            attention_head_dim: Optional[int] = 8,
            resnet_time_scale_shift: ResnetTimeScaleShiftType = "default",
            dropout: float = 0.0,
            double_z: bool = False,
            mid_block_add_attention: bool = True,

    ):
        super(Encoder, self).__init__()

        self.conv_in = nn.Conv2d(
            in_channels,
            channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.blocks = nn.ModuleList([])
        output_channel = channels[0]

        for i, down_block_type in enumerate(block_types):
            input_channel = output_channel
            output_channel = channels[i]
            is_final_block = i == len(channels) - 1

            block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=None,
                do_down_sample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                down_sample_mode="conv",
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                resnet_time_scale_shift=resnet_time_scale_shift,
                dropout=dropout,
            )
            self.blocks.append(block)


        self.mid_block = MidBlock(
            in_channels=channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            resnet_time_scale_shift="default",
            attention_head_dim=channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
        )

        self.conv_norm_out = nn.GroupNorm(num_channels=channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(channels[-1], conv_out_channels, 3, padding=1)


    def forward(self, x: Tensor) -> Tensor:

        x = self.conv_in(x)

        for block in self.blocks:
            x = block(hidden_states=x)

        x = self.mid_block(x)

        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        return x


class Decoder(nn.Module):

    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            block_types: Tuple[UpBlockType, str, ...] = ("UpDecoderBlock",),
            channels: Tuple[int, ...] = (64,),
            layers_per_block: int = 3,
            norm_num_groups: int = 32,
            act_fn: str = "silu",
            resnet_time_scale_shift: ResnetTimeScaleShiftType = "default",
            mid_block_add_attention=True,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(
            in_channels,
            channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.blocks = nn.ModuleList([])

        # mid
        self.mid_block = MidBlock(
            in_channels=channels[0],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_head_dim=channels[0],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
        )


        output_channel = channels[0]
        for i, up_block_type in enumerate(block_types):
            prev_output_channel = output_channel
            output_channel = channels[i]

            is_final_block = i == len(channels) - 1

            block = get_up_block(
                up_block_type,
                num_layers=layers_per_block,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=0,
                do_up_sample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.blocks.append(block)
            prev_output_channel = output_channel


        self.conv_norm_out = nn.GroupNorm(num_channels=channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(channels[-1], out_channels, 3, padding=1)


    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.conv_in(x)

        x = self.mid_block(x)

        # up
        for block in self.blocks:
            x = block(hidden_states=x)

        x = self.conv_norm_out(x)

        x = self.conv_act(x)
        x = self.conv_out(x)

        return x

if __name__ == "__main__":

    import lovely_tensors
    lovely_tensors.monkey_patch()

    kl_aa = AutoencoderKL()

    img = torch.randn((1, 1, 64, 64))

    print(kl_aa(img))