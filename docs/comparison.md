# Variant Comparison

| Variant | Global Context | Multi-Scale Attention | Detect Inputs | Parameters |
|---|---|---|---|---:|
| Baseline | C2PSA | Standard FPN/PAN | P3, P4, P5 | 9.432M |
| MobileViT | MobileViT at P5 | Standard FPN/PAN | P3, P4, P5 | 10.634M |
| MSCA | C2PSA | MSCA after P3 fusion | P3, P4, P5 | 9.518M |
| MobileViT-MSCA | MobileViT at P5 | MSCA after P3 fusion | P3, P4, P5 | 10.720M |
| MobileViT-MSCA-P2 | MobileViT at P5 | MSCA after P3 fusion | P2, P3, P4, P5 | 10.866M |
| MobileViT-MSCA-P2 Edge | MobileViT at P5 | MSCA after P3 fusion | P2, P3, P4, P5 | 2.999M |

Depthwise-separable local and fusion paths keep the MobileViT variant within 13% of the baseline parameter count.

## VisDrone Protocol

`scripts/compare_visdrone.py` compares the following models under one controlled protocol:

| Key | Initialization | Role |
|---|---|---|
| `yolo11s` | `yolo11s.pt` | Production-oriented control |
| `enhanced` | Project architecture plus compatible YOLO11s weights | Small-object proposal |
| `yolo12s` | `yolo12s.pt` | Attention-centric research reference |
| `rtdetr-l` | `rtdetr-l.pt` | End-to-end transformer reference |

Every run shares the same VisDrone split, image size, epoch budget, batch policy, seed, deterministic mode, and `max_det=300` validation limit. Reports record accuracy, parameter count, checkpoint size, and preprocess/inference/postprocess latency. Latency comparisons are meaningful only on the same host and backend.

Use `scripts/prepare_visdrone.py` to download and convert the official dataset definition, or point `configs/visdrone.yaml` at an existing converted dataset. Explicit CUDA requests fail fast if the current environment has a CPU-only PyTorch build.

RT-DETR's released model uses 300 queries, so scenes containing more than 300 annotated objects require a separately trained higher-query architecture for an unconstrained recall comparison.
