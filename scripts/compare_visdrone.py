"""Train and validate controlled YOLO11, enhanced, YOLO12, and RT-DETR experiments."""

from __future__ import annotations

import argparse
import platform
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from yolo11_small_object_enhancement.experiments import (
    COMPARISON_SPECS,
    create_comparison_model,
    environment_metadata,
    get_comparison_spec,
    load_comparison_checkpoint,
    result_from_metrics,
    write_comparison_report,
)
from yolo11_small_object_enhancement.runtime import validate_device


def parse_args() -> argparse.Namespace:
    """Parse the shared experiment protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=COMPARISON_SPECS, default=list(COMPARISON_SPECS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=Path("runs/visdrone-comparison"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run every model under identical training and validation settings."""
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"dataset configuration not found: {args.data}")
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        raise ValueError("epochs and batch must be positive; imgsz must be at least 32")
    if not args.dry_run:
        validate_device(args.device)

    if args.dry_run:
        create_comparison_model("enhanced", pretrained=False)
        for key in args.models:
            spec = get_comparison_spec(key)
            print(f"{key}: {spec.display_name} <- {spec.source} ({spec.notes})")
        print("Comparison configuration: PASS")
        return 0

    results = []
    for key in args.models:
        model = create_comparison_model(key)
        model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            seed=args.seed,
            deterministic=True,
            project=str(args.project),
            name=key,
            exist_ok=False,
        )
        checkpoint = Path(model.trainer.best)
        trained = load_comparison_checkpoint(key, checkpoint)
        metrics = trained.val(
            data=str(args.data),
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            max_det=300,
            plots=False,
        )
        results.append(result_from_metrics(key, checkpoint, trained, metrics))

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "device": args.device,
        "platform": platform.platform(),
        "max_det": 300,
        **environment_metadata(),
    }
    json_path, csv_path = write_comparison_report(results, args.project, metadata)
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
