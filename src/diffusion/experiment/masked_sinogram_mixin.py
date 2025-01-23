from typing import Optional

import torch
from dataclasses import dataclass
from torch import Tensor

from diffusion import Normalization
from diffusion.nn.leap_projector_wrapper import SimpleProjector
from diffusion.nn.slice_mask import SliceRandomMask, SliceMaskOutput


@dataclass
class MaskedSinogramOutput:
    """
        A data class for storing the outputs of a masked sinogram generation process.

        Attributes:
            sample (Tensor): The tensor representing the processed sinogram with applied masks.
            unmasked_sample (Tensor): The tensor representing the original sinogram before any masks are applied.
            mask (Tensor): The tensor representing the applied masks on the sinogram.
            mask_amount (Tensor): The tensor representing the amount of masking applied to the sinogram.
    """
    sample: Tensor
    unmasked_sample: Tensor
    mask: Tensor
    mask_amount: Tensor


class MaskedSinogramMixin:
    """
    MaskedSinogramMixin class
    A mixin class providing methods to create masked sinograms.

    Attributes
    ----------
    projector : SimpleProjector
        An instance of SimpleProjector for projecting the input tensor.
    slice_random_mask : SliceRandomMask
        An instance of SliceRandomMask for generating random masks on the input tensor.

    Methods
    -------
    create_masked_sinogram(x: Tensor, normalization: Normalization, return_dict: bool = True):
        Creates a masked sinogram from the input tensor with specified normalization and the option to return additional information.

        Parameters
        ----------
        x : Tensor
            Input tensor to be projected and masked.
        normalization : Normalization
            Specifies the normalization to apply to the input tensor before projection.
        return_dict : bool, optional
            Specifies whether to return additional information about the mask (default is True).

        Returns
        -------
        MaskedSinogramOutput or Tensor
            Depending on return_dict, returns a MaskedSinogramOutput object containing the masked sinogram and additional data, or just the masked tensor.
    """
    projector: SimpleProjector
    slice_random_mask: SliceRandomMask
    sinogram_normalization: Optional[Normalization]


    def compute_sinogram(
        self,
        x: Tensor,
        *,
        normalization: Optional[Normalization] = None,
    ):
        normalization = normalization or self.sinogram_normalization or Normalization.NONE
        match normalization:
            case Normalization.NONE:
                sinogram = x.clone()
            case Normalization.MINUS_ONE_ONE_TO_ZERO_ONE:
                sinogram = x.clone().add(1.).mul(.5).clamp(0., 1.)
            case Normalization.ZERO_ONE_TO_MINUS_ONE_ONE:
                sinogram = x.clone().mul(2.).sub(1.).clamp(-1., 1.)
            case _:
                raise NotImplementedError(f"Normalization {normalization} not implemented.")
        return self.projector(sinogram)

    @torch.no_grad()
    def rand_masked_sinogram(
        self,
        x: Tensor,
        normalization: Optional[Normalization] = None,
        return_dict: bool = True
    ):
        sinogram = self.compute_sinogram(x, normalization=normalization)

        if not return_dict:
            return self.slice_random_mask(sinogram, return_dict=False)
        else:
            mask_output = self.slice_random_mask(sinogram, return_dict=True)
            return MaskedSinogramOutput(
                sample=mask_output.sample,
                unmasked_sample=sinogram,
                mask=mask_output.mask,
                mask_amount=mask_output.mask_amount,
            )

    @torch.no_grad()
    def fixed_sparsity_sinogram(
        self,
        *,
        x: Optional[Tensor] = None,
        sinogram: Optional[Tensor] = None,
        keep_n_angles: int,
        normalization: Optional[Normalization] = None,
        return_dict: bool = True
    ):
        assert (x is not None) or (sinogram is not None), "Either x or sinogram must be provided."

        if sinogram is None:
            sinogram = self.compute_sinogram(x, normalization=normalization)

        if not return_dict:
            return self.slice_random_mask.fixed_sparsity_mask(
                x=sinogram,
                keep_n_angles=keep_n_angles,
                return_dict=False
            )
        else:
            mask_output: SliceMaskOutput = self.slice_random_mask.fixed_sparsity_mask(
                x=sinogram,
                keep_n_angles=keep_n_angles,
                return_dict=True
            )
            return MaskedSinogramOutput(
                sample=mask_output.sample,
                unmasked_sample=sinogram,
                mask=mask_output.mask,
                mask_amount=mask_output.mask_amount,
            )

    @torch.no_grad()
    def rand_contiguous_sinogram(
        self,
        x: Tensor,
        normalization: Optional[Normalization] = None,
        return_dict: bool = True
    ):
        sinogram = self.compute_sinogram(x, normalization=normalization)

        if not return_dict:
            return self.slice_random_mask.contiguous(sinogram, return_dict=False)
        else:
            mask_output = self.slice_random_mask.contiguous(sinogram, return_dict=True)
            return MaskedSinogramOutput(
                sample=mask_output.sample,
                unmasked_sample=sinogram,
                mask=mask_output.mask,
                mask_amount=mask_output.mask_amount,
            )

    @torch.no_grad()
    def fixed_contiguous_sinogram(
        self,
        *,
        x: Optional[Tensor] = None,
        sinogram: Optional[Tensor] = None,
        keep_n_angles: int,
        normalization: Optional[Normalization] = None,
        return_dict: bool = True
    ):
        if sinogram is None:
            sinogram = self.compute_sinogram(x, normalization=normalization)

        if not return_dict:
            return self.slice_random_mask.contiguous_mask(
                x=sinogram,
                keep_n_angles=keep_n_angles,
                return_dict=False
            )
        else:
            mask_output = self.slice_random_mask.contiguous_mask(
                x=sinogram,
                keep_n_angles=keep_n_angles,
                return_dict=True
            )
            return MaskedSinogramOutput(
                sample=mask_output.sample,
                unmasked_sample=sinogram,
                mask=mask_output.mask,
                mask_amount=mask_output.mask_amount,
            )