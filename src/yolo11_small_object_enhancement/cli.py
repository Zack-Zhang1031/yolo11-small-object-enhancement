"""Installed console entry points."""

from __future__ import annotations

import sys
from pathlib import Path

from .builder import build_model, count_parameters, detection_feature_shapes
from .training import create_yolo


def inspect_entrypoint() -> None:
    """Inspect a model from the installed package."""
    key = sys.argv[1] if len(sys.argv) > 1 else "mobilevit-msca-p2"
    model = build_model(key, verbose=True)
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Detect strides: {model.stride.tolist()}")
    print(f"Detect feature shapes: {detection_feature_shapes(model, 320)}")


def train_entrypoint() -> None:
    """Start training from the installed package with Ultralytics-style key=value arguments."""
    arguments = dict(argument.split("=", 1) for argument in sys.argv[1:] if "=" in argument)
    model_key = arguments.pop("model", "mobilevit-msca-p2")
    data = arguments.pop("data", None)
    if data is None or not Path(data).is_file():
        raise SystemExit("data=<dataset.yaml> is required")
    model = create_yolo(model_key)
    model.train(data=data, **arguments)
