"""Check numerical parity between a PyTorch checkpoint and an exported model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import torch
from ultralytics.nn.autobackend import AutoBackend

from yolo11_small_object_enhancement import create_yolo


def first_tensor(value: Any) -> torch.Tensor:
    """Return the first prediction tensor from a nested backend result."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for item in value.values():
            try:
                return first_tensor(item)
            except TypeError:
                continue
    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                return first_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"no tensor found in output type {type(value).__name__}")


def parse_args() -> argparse.Namespace:
    """Parse parity-check inputs and tolerances."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser.parse_args()


def main() -> int:
    """Run identical deterministic input through both backends."""
    args = parse_args()
    for path in (args.weights, args.export):
        if not path.is_file():
            raise FileNotFoundError(f"model artifact not found: {path}")
    device = torch.device(args.device)
    generator = torch.Generator(device="cpu").manual_seed(42)
    image = torch.rand(1, 3, args.imgsz, args.imgsz, generator=generator).to(device)

    source = create_yolo(str(args.weights)).model.to(device).eval()
    target = AutoBackend(str(args.export), device=device, fp16=False, verbose=False)
    with torch.inference_mode():
        source_output = first_tensor(source(image))
        target_output = first_tensor(target(image))
    if source_output.shape != target_output.shape:
        raise RuntimeError(
            f"output shape mismatch: source={source_output.shape}, export={target_output.shape}"
        )
    max_abs_error = float((source_output.float() - target_output.float()).abs().max())
    if not torch.allclose(source_output.float(), target_output.float(), atol=args.atol, rtol=args.rtol):
        raise RuntimeError(f"export parity failed: max_abs_error={max_abs_error:.6g}")
    print(f"Export parity: PASS (shape={tuple(source_output.shape)}, max_abs_error={max_abs_error:.6g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
