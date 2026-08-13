"""Lightweight multi-scale convolutional attention for YOLO feature maps."""

from __future__ import annotations

import torch
from torch import nn

from .common import ConvNormAct


class MSCA(nn.Module):
    """Combine depthwise multi-scale context with channel and spatial weighting."""

    def __init__(
        self,
        c1: int,
        c2: int,
        kernels: tuple[int, ...] | list[int] = (3, 5, 7),
        reduction: int = 4,
    ) -> None:
        super().__init__()
        kernels = tuple(kernels)
        if c1 < 1 or c2 < 1:
            raise ValueError("channel dimensions must be positive")
        if not kernels or any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
            raise ValueError("kernels must contain positive odd values")
        if reduction < 1:
            raise ValueError("reduction must be positive")

        self.input_projection = ConvNormAct(c1, c2, 1)
        self.branches = nn.ModuleList(
            ConvNormAct(c2, c2, kernel, groups=c2) for kernel in kernels
        )
        self.fusion = ConvNormAct(c2 * len(kernels), c2, 1)
        hidden = max(c2 // reduction, 8)
        self.channel_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c2, 1),
            nn.Sigmoid(),
        )
        self.spatial_weight = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
        self.shortcut = nn.Identity() if c1 == c2 else ConvNormAct(c1, c2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhance a feature map while preserving its declared output shape."""
        projected = self.input_projection(x)
        multi_scale = self.fusion(torch.cat([branch(projected) for branch in self.branches], dim=1))
        channel_refined = multi_scale * self.channel_weight(multi_scale)
        statistics = torch.cat(
            (channel_refined.mean(dim=1, keepdim=True), channel_refined.amax(dim=1, keepdim=True)), dim=1
        )
        return channel_refined * self.spatial_weight(statistics) + self.shortcut(x)
