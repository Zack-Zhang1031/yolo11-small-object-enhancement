# Small-Object Enhancement

## Overview

The feature adds global context, multi-scale convolutional attention, and a higher-resolution detection branch to YOLO11s.

## Design Decisions

- Place MobileViT at P5 to limit token count.
- Use depthwise-separable spatial and fusion convolutions to control parameter growth.
- Refine P3 after top-down fusion with depthwise MSCA.
- Build P2 from the backbone's stride-4 feature and retain a four-level PAN path.
- Bundle YAML resources inside a uniquely named src-layout Python package.
- Register custom modules before every Ultralytics construction path.
- Use a width-0.25 edge student that retains all four detection levels.
- Distill only between the full and edge project graphs because feature hooks require matching layer indices.

## Implementation Notes

Custom blocks expose `[B, C, H, W] -> [B, C_out, H, W]`. `predict_tensor` handles arbitrary rectangular image tensors by padding to the maximum model stride. Repository and installed console entry points share the same package API.

The edge workflow initializes compatible weights from YOLO11n, then uses the pinned Ultralytics score-weighted feature distillation implementation. Export validation compares deterministic raw outputs across PyTorch and the target backend before latency benchmarking.
