"""Tests for configuration resources and tensor preprocessing."""

import pytest
import torch

from yolo11_small_object_enhancement import build_model, pad_to_stride, predict_tensor
from yolo11_small_object_enhancement.builder import config_path


def test_bundled_configuration_is_available() -> None:
    with config_path("mobilevit-msca-p2") as path:
        assert path.is_file()
        assert "MobileViTBlock" in path.read_text(encoding="utf-8")


def test_edge_configuration_is_bundled() -> None:
    with config_path("mobilevit-msca-p2-edge") as path:
        assert path.is_file()
        assert "scale: n" in path.read_text(encoding="utf-8")


def test_padding_aligns_odd_rectangular_input() -> None:
    padded, padding = pad_to_stride(torch.zeros(1, 3, 321, 511))
    assert padded.shape == (1, 3, 352, 512)
    assert padding == (31, 1)


def test_predict_tensor_can_reject_unaligned_input() -> None:
    with pytest.raises(ValueError, match="divisible"):
        predict_tensor(build_model("baseline"), torch.zeros(1, 3, 321, 511), auto_pad=False)


def test_pad_to_stride_rejects_non_bchw_tensor() -> None:
    with pytest.raises(ValueError, match="BCHW"):
        pad_to_stride(torch.zeros(3, 320, 320))


def test_pad_to_stride_rejects_integer_tensor() -> None:
    with pytest.raises(ValueError, match="floating-point"):
        pad_to_stride(torch.zeros(1, 3, 320, 320, dtype=torch.uint8))


def test_predict_tensor_restores_training_mode() -> None:
    model = build_model("baseline").train()
    predict_tensor(model, torch.zeros(1, 3, 128, 128))
    assert model.training
