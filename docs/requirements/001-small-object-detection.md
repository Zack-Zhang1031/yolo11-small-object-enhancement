# YOLO11 Small-Object Detection Requirements

## Overview

Provide installable YOLO11 variants with MobileViT, MSCA, an optional P2 detection head, and an edge-scale distilled student.

## User Stories

- As an engineer, I can construct each model through a stable Python API.
- As a researcher, I can start Ultralytics training through a registered project entry point.
- As a reviewer, I can inspect the actual Detect feature maps and strides.
- As a package user, I can install a wheel that includes every model configuration.
- As a researcher, I can compare the enhanced model with YOLO11s, YOLO12s, and RT-DETR-L under one protocol.
- As a deployment engineer, I can distill, export, verify, and benchmark a compact P2 student.

## Functional Reqs

- Build baseline, MobileViT, MSCA, combined, and combined-P2 variants.
- Accept BCHW tensor input with automatic stride alignment.
- Support real image inference and Ultralytics training entry points.
- Package all model YAML files as Python resources.
- Record comparison results as JSON and CSV with accuracy, size, and latency fields.
- Reject incompatible teacher/student graphs before starting distillation.
- Require calibration data for INT8 export and numerical parity checks for exported artifacts.

## Non-Functional Reqs

- Support Python 3.10 or later on Windows and Linux.
- Keep MobileViT parameters within 25% of the baseline model.
- Preserve the installed Ultralytics package unchanged.
- Keep generated weights, datasets, and runs out of version control.
- Keep the edge student below 3.1 million parameters while retaining P2/P3/P4/P5 outputs.

## Data Model

YAML files define model graphs and dataset class mappings. Checkpoints use standard Ultralytics formats.

## UI/UX

Command-line tools print model parameters, feature scales, padding, and actionable errors.

## API

- `build_model(key)` constructs a PyTorch detection model.
- `predict_tensor(model, tensor)` performs stride-safe tensor inference.
- `create_yolo(model)` creates a registered Ultralytics wrapper.
- `train_distilled_model(student, teacher, data)` starts feature distillation.

## Testing

CI runs Ruff, Mypy, compileall, pytest, all-model construction, comparison dry-run, smoke inference, and wheel construction.

## Open Questions

- The production deployment target determines whether ONNX, OpenVINO, or TensorRT is preferred.
- Deployment licensing depends on whether the application is open source or proprietary.
