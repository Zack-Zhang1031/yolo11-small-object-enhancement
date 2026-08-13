"""Controlled VisDrone comparison experiment definitions and result reporting."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ultralytics import RTDETR, YOLO
from ultralytics import __version__ as ultralytics_version

from .builder import count_parameters
from .training import create_yolo


@dataclass(frozen=True, slots=True)
class ComparisonSpec:
    """A model participating in the controlled VisDrone comparison."""

    key: str
    display_name: str
    source: str
    family: str
    notes: str


COMPARISON_SPECS: dict[str, ComparisonSpec] = {
    spec.key: spec
    for spec in (
        ComparisonSpec("yolo11s", "YOLO11s", "yolo11s.pt", "yolo", "Official control model"),
        ComparisonSpec(
            "enhanced",
            "YOLO11s + MobileViT + MSCA + P2",
            "mobilevit-msca-p2",
            "project",
            "Loads compatible YOLO11s pretrained weights",
        ),
        ComparisonSpec(
            "yolo12s",
            "YOLO12s",
            "yolo12s.pt",
            "yolo",
            "Attention-centric community benchmark",
        ),
        ComparisonSpec(
            "rtdetr-l",
            "RT-DETR-L",
            "rtdetr-l.pt",
            "rtdetr",
            "End-to-end detector; pretrained checkpoint uses 300 queries",
        ),
    )
}


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Serializable validation result for one trained model."""

    key: str
    checkpoint: str
    parameters: int
    checkpoint_mb: float
    map50_95: float
    map50: float
    map75: float
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float


def get_comparison_spec(key: str) -> ComparisonSpec:
    """Return a comparison definition by key."""
    try:
        return COMPARISON_SPECS[key]
    except KeyError as error:
        available = ", ".join(COMPARISON_SPECS)
        raise KeyError(f"Unknown comparison model '{key}'. Available: {available}") from error


def create_comparison_model(key: str, *, pretrained: bool = True) -> YOLO | RTDETR:
    """Create one comparison model through its correct Ultralytics wrapper."""
    spec = get_comparison_spec(key)
    if spec.family == "project":
        return create_yolo(spec.source, pretrained="yolo11s.pt" if pretrained else None)
    if spec.family == "rtdetr":
        return RTDETR(spec.source)
    return YOLO(spec.source, task="detect")


def load_comparison_checkpoint(key: str, checkpoint: str | Path) -> YOLO | RTDETR:
    """Load a trained checkpoint through the wrapper required by its model family."""
    spec = get_comparison_spec(key)
    if spec.family == "project":
        return create_yolo(str(checkpoint))
    if spec.family == "rtdetr":
        return RTDETR(str(checkpoint))
    return YOLO(str(checkpoint), task="detect")


def result_from_metrics(key: str, checkpoint: Path, model: YOLO | RTDETR, metrics: Any) -> ComparisonResult:
    """Normalize Ultralytics detection metrics for CSV and JSON reports."""
    speed = metrics.speed
    return ComparisonResult(
        key=key,
        checkpoint=str(checkpoint),
        parameters=count_parameters(model.model),
        checkpoint_mb=round(checkpoint.stat().st_size / (1024 * 1024), 3),
        map50_95=float(metrics.box.map),
        map50=float(metrics.box.map50),
        map75=float(metrics.box.map75),
        preprocess_ms=float(speed.get("preprocess", 0.0)),
        inference_ms=float(speed.get("inference", 0.0)),
        postprocess_ms=float(speed.get("postprocess", 0.0)),
    )


def write_comparison_report(
    results: list[ComparisonResult], output_dir: str | Path, metadata: dict[str, Any]
) -> tuple[Path, Path]:
    """Write machine-readable JSON and spreadsheet-friendly CSV reports."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "comparison.json"
    csv_path = destination / "comparison.csv"
    json_path.write_text(
        json.dumps({"metadata": metadata, "results": [asdict(result) for result in results]}, indent=2),
        encoding="utf-8",
    )
    fieldnames = list(ComparisonResult.__dataclass_fields__)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    return json_path, csv_path


def environment_metadata() -> dict[str, str | None]:
    """Return library versions required to repeat a comparison."""
    import torch

    return {
        "ultralytics": ultralytics_version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
