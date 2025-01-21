import pytest
import torch
from torch import Tensor

from diffusion.nn.slice_mask import SliceRandomMask, SliceMaskOutput

KEEP_MIN = 12
KEEP_MAX = 48
MASK_VALUE = 0.

B = 3
C = 1
H = KEEP_MAX
W = 64

DEVICE = torch.device("cpu")

@pytest.fixture
def slice_random_mask():
    return SliceRandomMask(
        keep_min=KEEP_MIN,
        keep_max=KEEP_MAX,
        device=DEVICE,
        mask_value=MASK_VALUE,
    )

def test_slice_random_mask_returns_dict(slice_random_mask):
    tensor_a = torch.randn(B, C, H, W, device=DEVICE)

    slice_mask = slice_random_mask(tensor_a)
    assert isinstance(slice_mask, SliceMaskOutput)


def test_slice_random_mask_returns_tensor(slice_random_mask):

    tensor_a = torch.randn(B, C, H, W, device=DEVICE)

    slice_mask = slice_random_mask(tensor_a, return_dict=False)
    assert isinstance(slice_mask, Tensor)


def test_slice_random_mask_shape(slice_random_mask):
    tensor_a = torch.randn(B, C, H, W, device=DEVICE)

    slice_mask = slice_random_mask(tensor_a, return_dict=False)

    assert slice_mask.shape == (B, C, H, W)


def test_slice_random_mask_unique(slice_random_mask):

    tensor_a = torch.randn(B, C, H, W, device=DEVICE)

    mask_a = slice_random_mask(tensor_a, return_dict=False)
    mask_b = slice_random_mask(tensor_a, return_dict=False)

    assert not torch.equal(mask_a, mask_b)


def test_slice_random_mask_not_equal_without_reset_rng(slice_random_mask):

    tensor_a = torch.randn(B, C, H, W, device=DEVICE)

    slice_random_mask.reset_rng()

    mask_a = slice_random_mask.fixed_sparsity_mask(
        x=tensor_a,
        keep_n_angles=KEEP_MAX-1,
        return_dict=False
    )

    mask_b = slice_random_mask.fixed_sparsity_mask(
        x=tensor_a,
        keep_n_angles=KEEP_MAX-1,
        return_dict=False
    )

    assert not torch.equal(mask_a, mask_b)

def test_slice_random_mask_equal_with_reset_rng(slice_random_mask):

    tensor_a = torch.randn(B, C, H, W, device=DEVICE)

    slice_random_mask.reset_rng()

    mask_a = slice_random_mask.fixed_sparsity_mask(
        x=tensor_a,
        keep_n_angles=KEEP_MAX-1,
        return_dict=False
    )

    slice_random_mask.reset_rng()

    mask_b = slice_random_mask.fixed_sparsity_mask(
        x=tensor_a,
        keep_n_angles=KEEP_MAX-1,
        return_dict=False
    )

    assert torch.equal(mask_a, mask_b)