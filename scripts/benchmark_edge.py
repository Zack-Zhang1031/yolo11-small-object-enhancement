"""Benchmark a deployment checkpoint or export on VisDrone with Ultralytics."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from ultralytics.utils.benchmarks import benchmark

from yolo11_small_object_enhancement.runtime import validate_device


def parse_args() -> argparse.Namespace:
    """Parse benchmark options shared across deployment formats."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--format", choices=("onnx", "openvino", "engine"), default="onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quantize", choices=(8, 16, 32), type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Validate inputs and run accuracy/latency benchmarking."""
    args = parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"model artifact not found: {args.weights}")
    if not args.data.is_file():
        raise FileNotFoundError(f"dataset configuration not found: {args.data}")
    if args.format == "engine" and args.device == "cpu":
        raise ValueError("TensorRT benchmarking requires a CUDA device")
    validate_device(args.device)
    if args.dry_run:
        print(f"Benchmark configuration: PASS ({args.format}, quantize={args.quantize})")
        return 0
    result = benchmark(
        model=str(args.weights),
        data=str(args.data),
        imgsz=args.imgsz,
        device=args.device,
        format=args.format,
        quantize=args.quantize,
        verbose=True,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
