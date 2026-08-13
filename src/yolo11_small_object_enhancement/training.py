"""Ultralytics training integration with project layer registration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO
from ultralytics.nn.modules import Detect

from .builder import config_path, register_custom_modules


def create_yolo(
    model: str = "mobilevit-msca-p2", *, pretrained: str | Path | None = None, verbose: bool = False
) -> YOLO:
    """Create an Ultralytics YOLO wrapper for a project variant or external weights."""
    register_custom_modules()
    candidate = Path(model)
    if candidate.suffix.lower() in {".pt", ".yaml", ".yml"}:
        yolo = YOLO(candidate, task="detect", verbose=verbose)
    else:
        with config_path(model) as path:
            yolo = YOLO(path, task="detect", verbose=verbose)
    if pretrained is not None:
        yolo.load(pretrained)
    return yolo


def train_model(model: str, data: str | Path, **kwargs: Any) -> Any:
    """Train a registered model with Ultralytics keyword arguments."""
    return create_yolo(model).train(data=str(data), **kwargs)


def train_distilled_model(
    student: str,
    teacher: str | Path,
    data: str | Path,
    *,
    distill_weight: float = 6.0,
    student_pretrained: str | Path | None = "yolo11n.pt",
    **kwargs: Any,
) -> Any:
    """Train a student with Ultralytics feature distillation from a teacher checkpoint."""
    teacher_path = Path(teacher)
    if teacher_path.suffix.lower() != ".pt" or not teacher_path.is_file():
        raise FileNotFoundError(f"teacher checkpoint not found: {teacher_path}")
    if distill_weight <= 0:
        raise ValueError("distill_weight must be positive")
    student_model = create_yolo(student, pretrained=student_pretrained)
    teacher_model = create_yolo(str(teacher_path))
    validate_distillation_pair(student_model, teacher_model)
    return student_model.train(
        data=str(data), distill_model=str(teacher_path), dis=distill_weight, **kwargs
    )


def validate_distillation_pair(student: YOLO, teacher: YOLO) -> None:
    """Reject teacher/student pairs that cannot share Ultralytics feature hooks."""

    def detect_head(model: YOLO) -> Detect:
        head = model.model.model[-1]
        if not isinstance(head, Detect):
            raise TypeError(f"expected a Detect head, received {type(head).__name__}")
        return head

    student_head = detect_head(student)
    teacher_head = detect_head(teacher)
    student_sources = tuple(student_head.f)
    teacher_sources = tuple(teacher_head.f)
    if student_sources != teacher_sources or len(student.model.model) != len(teacher.model.model):
        raise ValueError(
            "feature distillation requires teacher and student variants with matching layer indices; "
            "train the mobilevit-msca-p2 teacher before distilling mobilevit-msca-p2-edge"
        )
    if student_head.nc != teacher_head.nc:
        raise ValueError(
            f"teacher/student class mismatch: teacher={teacher_head.nc}, student={student_head.nc}"
        )
