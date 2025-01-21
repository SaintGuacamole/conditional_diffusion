from typing import Union, Callable, Optional

from torch import Tensor, chunk, cat, zeros_like


def basic_classifier_free_guidance(
    conditional_input: Tensor,
    unconditional_input: Tensor,
    guidance_scale: Union[Tensor, float],
):

    return unconditional_input + guidance_scale * (conditional_input - unconditional_input)


def tradeoff_classifier_free_guidance(
    conditional_input: Tensor,
    unconditional_input: Tensor,
    guidance_scale: Union[Tensor, float],
):

    return (1-guidance_scale) * unconditional_input + guidance_scale * conditional_input


def compute_guidance(
    model_output: Tensor,
    guidance_function: Callable,
    guidance_scale: Union[Tensor, float],
    conditional_first: bool = True
):
    assert model_output.shape[0] % 2 == 0, f"The model output must have a batch size multiple of two, found {model_output.shape[0]}"
    if conditional_first:
        conditional, unconditional = chunk(model_output, 2, dim=0)
    else:
        unconditional, conditional = chunk(model_output, 2, dim=0)
    return guidance_function(conditional, unconditional, guidance_scale)


def create_classifier_free_guidance_input(
    conditional_input: Tensor,
    unconditional_input: Optional[Tensor],
    conditional_first: bool = True,
):

    if unconditional_input is None:
        unconditional_input = zeros_like(conditional_input, device=conditional_input.device)
    else:
        assert conditional_input.shape == unconditional_input.shape, f"Both inputs must have same dimensions, found {conditional_input.shape} and {unconditional_input.shape}"

    if conditional_first:
        return cat([conditional_input, unconditional_input], dim=0)
    else:
        return cat([unconditional_input, conditional_input], dim=0)
