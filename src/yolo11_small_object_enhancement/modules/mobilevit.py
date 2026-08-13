"""MobileViT-style feature block for Ultralytics detection backbones."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .common import ConvNormAct, DepthwiseSeparableConv


class MobileViTBlock(nn.Module):
    """Combine local convolutions and global token attention at constant resolution."""

    def __init__(
        self,
        c1: int,
        c2: int,
        transformer_dim: int = 128,
        depth: int = 2,
        patch_size: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        if c1 < 1 or c2 < 1 or transformer_dim < 1:
            raise ValueError("channel dimensions must be positive")
        if num_heads < 1 or transformer_dim % num_heads:
            raise ValueError("num_heads must divide transformer_dim")
        if depth < 1:
            raise ValueError("depth must be positive")
        if patch_size < 1:
            raise ValueError("patch_size must be positive")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")

        self.patch_size = patch_size
        self.local_representation = nn.Sequential(
            DepthwiseSeparableConv(c1, c1, 3),
            ConvNormAct(c1, transformer_dim, 1),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=int(transformer_dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            enable_nested_tensor=False,
        )
        self.global_projection = ConvNormAct(transformer_dim, c2, 1)
        self.fusion = DepthwiseSeparableConv(c1 + c2, c2, 3)
        self.shortcut = nn.Identity() if c1 == c2 else ConvNormAct(c1, c2, 1)

    def _to_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        """Unfold a feature map into patch-position token sequences."""
        batch, channels, height, width = x.shape
        patch = self.patch_size
        padded_h = math.ceil(height / patch) * patch
        padded_w = math.ceil(width / patch) * patch
        x = F.pad(x, (0, padded_w - width, 0, padded_h - height))
        grid_h, grid_w = padded_h // patch, padded_w // patch
        tokens = (
            x.reshape(batch, channels, grid_h, patch, grid_w, patch)
            .permute(0, 3, 5, 2, 4, 1)
            .reshape(batch * patch * patch, grid_h * grid_w, channels)
        )
        return tokens, (height, width, grid_h, grid_w)

    def _to_feature_map(
        self,
        tokens: torch.Tensor,
        metadata: tuple[int, int, int, int],
        batch: int,
    ) -> torch.Tensor:
        """Fold patch-position token sequences back into a feature map."""
        height, width, grid_h, grid_w = metadata
        patch = self.patch_size
        channels = tokens.shape[-1]
        feature = (
            tokens.reshape(batch, patch, patch, grid_h, grid_w, channels)
            .permute(0, 5, 3, 1, 4, 2)
            .reshape(batch, channels, grid_h * patch, grid_w * patch)
        )
        return feature[:, :, :height, :width]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a locally and globally enhanced feature map."""
        local = self.local_representation(x)
        tokens, metadata = self._to_tokens(local)
        global_features = self._to_feature_map(self.transformer(tokens), metadata, x.shape[0])
        global_features = self.global_projection(global_features)
        return self.fusion(torch.cat((x, global_features), dim=1)) + self.shortcut(x)
