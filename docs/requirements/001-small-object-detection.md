# YOLO11 Small-Object Detection Requirements

## Overview

Provide five installable YOLO11 variants with MobileViT, MSCA, and an optional P2 detection head.

## User Stories

- As an engineer, I can construct each model through a stable Python API.
- As a researcher, I can start Ultralytics training through a registered project entry point.
- As a reviewer, I can inspect the actual Detect feature maps and strides.
- As a package user, I can install a wheel that includes every model configuration.

## Functional Reqs

- Build baseline, MobileViT, MSCA, combined, and combined-P2 variants.
- Accept BCHW tensor input with automatic stride alignment.
- Support real image inference and Ultralytics training entry points.
- Package all model YAML files as Python resources.

## Non-Functional Reqs

- Support Python 3.10 or later on Windows and Linux.
- Keep MobileViT parameters within 25% of the baseline model.
- Preserve the installed Ultralytics package unchanged.
- Keep generated weights, datasets, and runs out of version control.

## Data Model

YAML files define model graphs and dataset class mappings. Checkpoints use standard Ultralytics formats.

## UI/UX

Command-line tools print model parameters, feature scales, padding, and actionable errors.

## API

- `build_model(key)` constructs a PyTorch detection model.
- `predict_tensor(model, tensor)` performs stride-safe tensor inference.
- `create_yolo(model)` creates a registered Ultralytics wrapper.

## Testing

CI runs Ruff, Mypy, compileall, pytest, all-model construction, smoke inference, and wheel construction.

## Open Questions

- Dataset-specific hyperparameters should be selected for each experiment.
- Deployment licensing depends on whether the application is open source or proprietary.
