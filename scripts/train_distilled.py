"""Distill a trained enhanced teacher into the edge-scale P2 student."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from yolo11_small_object_enhancement import create_yolo, train_distilled_model
from yolo11_small_object_enhancement.runtime import validate_device


def parse_args() -> argparse.Namespace:
    """Parse distillation options supported by the pinned Ultralytics release."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True, help="Trained full enhanced .pt checkpoint")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--student", default="mobilevit-msca-p2-edge")
    parser.add_argument("--student-pretrained", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distill-weight", type=float, default=6.0)
    parser.add_argument("--project", default="runs/visdrone-distillation")
    parser.add_argument("--name", default="mobilevit-msca-p2-edge-kd")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Validate inputs and optionally start feature distillation."""
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"dataset configuration not found: {args.data}")
    if not args.teacher.is_file() or args.teacher.suffix.lower() != ".pt":
        raise FileNotFoundError(f"teacher checkpoint not found: {args.teacher}")
    if args.distill_weight <= 0:
        raise ValueError("distill-weight must be positive")
    validate_device(args.device)
    create_yolo(args.student, verbose=args.dry_run)
    if args.dry_run:
        print(f"Distillation configuration: PASS ({args.student} <- {args.teacher})")
        return 0
    train_distilled_model(
        args.student,
        args.teacher,
        args.data,
        distill_weight=args.distill_weight,
        student_pretrained=args.student_pretrained,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        project=args.project,
        name=args.name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
