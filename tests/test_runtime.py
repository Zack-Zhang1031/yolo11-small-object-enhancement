"""Tests for explicit compute-device validation."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from yolo11_small_object_enhancement.runtime import validate_device


@pytest.mark.parametrize("device", ["", "cpu", "mps"])
def test_non_cuda_devices_are_accepted(device: str) -> None:
    validate_device(device)


@pytest.mark.parametrize("device", ["0", "0,1", "cuda", "cuda:0"])
def test_cuda_request_requires_cuda_build(device: str) -> None:
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="no CUDA support"):
            validate_device(device)
