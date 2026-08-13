"""Run enhanced YOLO11 inference on an image or a generated input tensor."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import torch

from yolo11_small_object_enhancement import build_model, create_yolo, predict_tensor
from yolo11_small_object_enhancement.builder import describe_output


def parse_args() -> argparse.Namespace:
    """Parse inference options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="Image, directory, video, URL, or camera accepted by Ultralytics")
    parser.add_argument("--weights", help="Optional .pt checkpoint used with --source")
    parser.add_argument("--model", default="mobilevit-msca-p2", help="Project model variant")
    parser.add_argument("--device", default="cpu", help="Ultralytics device string")
    parser.add_argument("--save", action="store_true", help="Save annotated predictions")
    return parser.parse_args()


def main() -> int:
    """Run image inference or the built-in tensor path."""
    args = parse_args()
    if args.source:
        model = create_yolo(args.weights or args.model)
        results = model.predict(source=args.source, device=args.device, save=args.save)
        print(f"Inference complete: {len(results)} result(s)")
        return 0

    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, size=(321, 511, 3), dtype=np.uint8)
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    output, padding = predict_tensor(build_model(args.model), tensor)
    print(f"Tensor inference: PASS (padding={padding})")
    print(f"Output: {describe_output(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
