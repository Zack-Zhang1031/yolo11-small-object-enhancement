"""Ultralytics training integration with project layer registration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO

from .builder import config_path, register_custom_modules


def create_yolo(model: str = "mobilevit-msca-p2", *, verbose: bool = False) -> YOLO:
    """Create an Ultralytics YOLO wrapper for a project variant or external weights."""
    register_custom_modules()
    candidate = Path(model)
    if candidate.suffix.lower() in {".pt", ".yaml", ".yml"}:
        return YOLO(candidate, task="detect", verbose=verbose)
    with config_path(model) as path:
        return YOLO(path, task="detect", verbose=verbose)


def train_model(model: str, data: str | Path, **kwargs: Any) -> Any:
    """Train a registered model with Ultralytics keyword arguments."""
    return create_yolo(model).train(data=str(data), **kwargs)
