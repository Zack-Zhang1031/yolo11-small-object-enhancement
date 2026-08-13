"""MobileViT-style feature block for Ultralytics detection backbones."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .common import ConvNormAct, DepthwiseSeparableConv


class ExportableTransformerBlock(nn.Module):
    """Pre-norm attention block expressed with ONNX-supported tensor operations."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.projection = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        """Compute multi-head self-attention with portable matrix operations."""
        batch, tokens, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch, tokens, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv.unbind(0)
        weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attended = torch.matmul(weights.softmax(dim=-1), value)
        return self.projection(attended.transpose(1, 2).reshape(batch, tokens, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention and feed-forward residual updates."""
        normalized = self.norm1(x)
        x = x + self._attention(normalized)
        return x + self.mlp(self.norm2(x))


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
        self.transformer = nn.Sequential(
            *(ExportableTransformerBlock(transformer_dim, num_heads, mlp_ratio) for _ in range(depth))
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
