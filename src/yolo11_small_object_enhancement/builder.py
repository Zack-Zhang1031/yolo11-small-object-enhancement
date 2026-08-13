"""Model construction, registration, and tensor preprocessing."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from ultralytics.nn import tasks
from ultralytics.nn.modules import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML

from .modules import MSCA, MobileViTBlock


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Metadata for an available model variant."""

    key: str
    display_name: str
    config_name: str


MODEL_SPECS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        ModelSpec("baseline", "YOLO11 Baseline", "yolo11_baseline.yaml"),
        ModelSpec("mobilevit", "YOLO11-MobileViT", "yolo11_mobilevit.yaml"),
        ModelSpec("msca", "YOLO11-MSCA", "yolo11_msca.yaml"),
        ModelSpec("mobilevit-msca", "YOLO11-MobileViT-MSCA", "yolo11_mobilevit_msca.yaml"),
        ModelSpec("mobilevit-msca-p2", "YOLO11-MobileViT-MSCA-P2", "yolo11_mobilevit_msca_p2.yaml"),
        ModelSpec(
            "mobilevit-msca-p2-edge",
            "YOLO11-MobileViT-MSCA-P2 Edge",
            "yolo11_mobilevit_msca_p2_edge.yaml",
        ),
    )
}


def register_custom_modules() -> None:
    """Register project layers in the active Ultralytics parser namespace."""
    tasks.MobileViTBlock = MobileViTBlock
    tasks.MSCA = MSCA


def get_model_spec(key: str) -> ModelSpec:
    """Return metadata for a model key."""
    try:
        return MODEL_SPECS[key]
    except KeyError as error:
        raise KeyError(f"Unknown model '{key}'. Available: {', '.join(MODEL_SPECS)}") from error


@contextmanager
def config_path(key: str) -> Iterator[Path]:
    """Yield a filesystem path for a bundled YAML configuration."""
    resource = files("yolo11_small_object_enhancement.configs").joinpath(get_model_spec(key).config_name)
    with as_file(resource) as path:
        yield path


def build_model(key: str, *, verbose: bool = False) -> DetectionModel:
    """Build a model variant with the upstream Ultralytics parser."""
    register_custom_modules()
    with config_path(key) as path:
        config = YAML.load(path)
    model = DetectionModel(config, ch=3, nc=int(config["nc"]), verbose=verbose)
    model.eval()
    return model


def count_parameters(model: torch.nn.Module) -> int:
    """Count all model parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def describe_output(output: Any) -> Any:
    """Convert nested inference output into tensor-shape metadata."""
    if isinstance(output, torch.Tensor):
        return list(output.shape)
    if isinstance(output, dict):
        return {key: describe_output(value) for key, value in output.items()}
    if isinstance(output, (tuple, list)):
        return [describe_output(value) for value in output]
    return type(output).__name__


def pad_to_stride(
    tensor: torch.Tensor, stride: int = 32, value: float = 114 / 255
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad a BCHW image tensor on the bottom and right to a stride multiple."""
    if tensor.ndim != 4:
        raise ValueError(f"expected BCHW tensor, received shape {tuple(tensor.shape)}")
    if not tensor.is_floating_point():
        raise ValueError("expected a floating-point image tensor")
    if stride < 1:
        raise ValueError("stride must be positive")
    height, width = tensor.shape[-2:]
    pad_h = (-height) % stride
    pad_w = (-width) % stride
    return F.pad(tensor, (0, pad_w, 0, pad_h), value=value), (pad_h, pad_w)


def predict_tensor(
    model: DetectionModel, tensor: torch.Tensor, *, auto_pad: bool = True
) -> tuple[Any, tuple[int, int]]:
    """Run tensor inference with optional stride-aligned padding."""
    padding = (0, 0)
    if auto_pad:
        tensor, padding = pad_to_stride(tensor, stride=int(model.stride.max().item()))
    elif any(dimension % int(model.stride.max().item()) for dimension in tensor.shape[-2:]):
        raise ValueError("input height and width must be divisible by the maximum model stride")
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            return model(tensor), padding
    finally:
        model.train(was_training)


def detection_feature_shapes(model: DetectionModel, image_size: int = 128) -> list[tuple[int, ...]]:
    """Capture the feature maps entering the Detect head."""
    head = model.model[-1]
    if not isinstance(head, Detect):
        raise TypeError(f"expected a Detect head, received {type(head).__name__}")
    captured: list[tuple[int, ...]] = []

    def capture_inputs(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        captured.extend(tuple(feature.shape) for feature in inputs[0])

    handle = head.register_forward_pre_hook(capture_inputs)
    try:
        predict_tensor(model, torch.zeros(1, 3, image_size, image_size))
    finally:
        handle.remove()
    return captured
