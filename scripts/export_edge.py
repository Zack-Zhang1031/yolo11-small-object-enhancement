"""Export a distilled edge checkpoint to an Ultralytics deployment format."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from yolo11_small_object_enhancement import create_yolo
from yolo11_small_object_enhancement.runtime import validate_device


def parse_args() -> argparse.Namespace:
    """Parse conservative edge-export options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--format", choices=("onnx", "openvino", "engine"), default="onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--data", type=Path, help="Calibration dataset YAML required for INT8")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Validate export constraints and create the deployment artifact."""
    args = parse_args()
    if not args.weights.is_file() or args.weights.suffix.lower() != ".pt":
        raise FileNotFoundError(f"checkpoint not found: {args.weights}")
    if args.half and args.int8:
        raise ValueError("choose either FP16 or INT8")
    if args.int8 and (args.data is None or not args.data.is_file()):
        raise ValueError("--data=<dataset.yaml> is required for INT8 calibration")
    if args.format == "engine" and args.device == "cpu":
        raise ValueError("TensorRT export requires a CUDA device such as --device=0")
    validate_device(args.device)
    model = create_yolo(str(args.weights))
    if args.dry_run:
        print(f"Export configuration: PASS ({args.format}, imgsz={args.imgsz})")
        return 0
    output = model.export(
        format=args.format,
        imgsz=args.imgsz,
        device=args.device,
        quantize=8 if args.int8 else 16 if args.half else 32,
        data=str(args.data) if args.data else None,
        dynamic=args.dynamic,
        simplify=True,
    )
    print(f"Exported artifact: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
