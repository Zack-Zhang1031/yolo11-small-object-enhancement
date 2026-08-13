"""Behavior tests for the MobileViT-style block."""

import pytest
import torch
from torch import nn

from yolo11_small_object_enhancement.modules import MobileViTBlock


def test_mobilevit_preserves_declared_shape_and_supports_backward() -> None:
    block = MobileViTBlock(32, 48, transformer_dim=48, depth=1, patch_size=2, num_heads=4)
    input_tensor = torch.randn(2, 32, 15, 17, requires_grad=True)
    output = block(input_tensor)
    assert output.shape == (2, 48, 15, 17)
    output.mean().backward()
    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()


def test_mobilevit_avoids_fused_transformer_encoder_layer() -> None:
    block = MobileViTBlock(16, 16, transformer_dim=16, depth=1, patch_size=2, num_heads=4)
    assert not any(isinstance(module, nn.TransformerEncoderLayer) for module in block.modules())
    assert not any(isinstance(module, nn.MultiheadAttention) for module in block.modules())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"c1": 0, "c2": 16}, "channel dimensions"),
        ({"c1": 16, "c2": 16, "transformer_dim": 30, "num_heads": 4}, "num_heads"),
        ({"c1": 16, "c2": 16, "depth": 0}, "depth"),
        ({"c1": 16, "c2": 16, "patch_size": 0}, "patch_size"),
        ({"c1": 16, "c2": 16, "mlp_ratio": 0}, "mlp_ratio"),
    ],
)
def test_mobilevit_rejects_invalid_configuration(kwargs: dict[str, int | float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MobileViTBlock(**kwargs)
