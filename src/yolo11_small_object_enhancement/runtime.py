"""Runtime preflight checks for controlled training and deployment."""

from __future__ import annotations

import re

import torch


def validate_device(device: str) -> None:
    """Reject explicit CUDA requests when the active PyTorch build cannot use CUDA."""
    normalized = device.strip().lower()
    if normalized in {"", "cpu", "mps"}:
        return
    requests_cuda = normalized == "cuda" or normalized.startswith("cuda:") or bool(
        re.fullmatch(r"\d+(,\d+)*", normalized)
    )
    if requests_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "a CUDA device was requested, but the active PyTorch build has no CUDA support; "
            "install a CUDA-enabled PyTorch wheel in this environment or pass --device=cpu"
        )
