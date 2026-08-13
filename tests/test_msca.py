"""Behavior tests for multi-scale convolutional attention."""

import pytest
import torch

from yolo11_small_object_enhancement.modules import MSCA


def test_msca_preserves_declared_shape_and_supports_backward() -> None:
    block = MSCA(32, 48)
    input_tensor = torch.randn(2, 32, 16, 20, requires_grad=True)
    output = block(input_tensor)
    assert output.shape == (2, 48, 16, 20)
    output.square().mean().backward()
    assert input_tensor.grad is not None
    assert torch.isfinite(input_tensor.grad).all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"c1": 0, "c2": 16}, "channel dimensions"),
        ({"c1": 16, "c2": 16, "kernels": []}, "kernels"),
        ({"c1": 16, "c2": 16, "kernels": [3, 4]}, "kernels"),
        ({"c1": 16, "c2": 16, "reduction": 0}, "reduction"),
    ],
)
def test_msca_rejects_invalid_configuration(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MSCA(**kwargs)
