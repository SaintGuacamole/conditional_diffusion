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

def test_slice_random_mask_fixed_mask_amount(slice_random_mask):

    tensor_a = torch.randn(B, C, H, W, device=DEVICE)

    mask_output = slice_random_mask.fixed_sparsity_mask(
        x=tensor_a,
        keep_n_angles=20,
        return_dict=True
    )
    assert isinstance(mask_output, SliceMaskOutput)

    assert (mask_output.mask_amount == 20. / H).all()

    for i in range(B):
        assert mask_output.mask[i, 0, :, 0].sum().item() == 20

def test_slice_random_mask_applies(slice_random_mask):

    tensor_a = torch.randn(B, C, H, W, device=DEVICE)

    mask_output = slice_random_mask(
        x=tensor_a,
        return_dict=True
    )

    assert isinstance(mask_output, SliceMaskOutput)
    assert (mask_output.sample == tensor_a * mask_output.mask + (1 - mask_output.mask) * MASK_VALUE).all()