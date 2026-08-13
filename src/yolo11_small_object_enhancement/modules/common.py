"""Shared convolution building blocks."""

from __future__ import annotations

from torch import nn


class ConvNormAct(nn.Sequential):
    """Convolution followed by batch normalization and SiLU activation."""

    def __init__(self, c1: int, c2: int, kernel_size: int = 1, groups: int = 1) -> None:
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        super().__init__(
            nn.Conv2d(c1, c2, kernel_size, padding=kernel_size // 2, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparableConv(nn.Sequential):
    """Depthwise spatial convolution followed by pointwise projection."""

    def __init__(self, c1: int, c2: int, kernel_size: int = 3) -> None:
        super().__init__(
            ConvNormAct(c1, c1, kernel_size, groups=c1),
            ConvNormAct(c1, c2, 1),
        )
