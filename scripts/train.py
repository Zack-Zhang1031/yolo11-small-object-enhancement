"""Train an enhanced YOLO11 variant through the registered project pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from yolo11_small_object_enhancement import create_yolo
from yolo11_small_object_enhancement.runtime import validate_device


def parse_args() -> argparse.Namespace:
    """Parse a focused set of Ultralytics training options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mobilevit-msca-p2", help="Variant key, YAML, or checkpoint")
    parser.add_argument("--pretrained", help="Compatible checkpoint used to initialize the model")
    parser.add_argument("--data", type=Path, required=True, help="Ultralytics dataset YAML")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="yolo11-mobilevit-msca-p2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Build and validate arguments without training"
    )
    return parser.parse_args()


def main() -> int:
    """Build the registered model and optionally start Ultralytics training."""
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"dataset configuration not found: {args.data}")
    validate_device(args.device)
    model = create_yolo(args.model, pretrained=args.pretrained, verbose=args.dry_run)
    if args.dry_run:
        print(f"Training configuration: PASS ({args.model}, data={args.data})")
        return 0
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
