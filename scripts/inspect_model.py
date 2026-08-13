"""Print the parsed layer graph and runtime metadata for one variant."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from yolo11_small_object_enhancement import build_model, count_parameters, detection_feature_shapes


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default="mobilevit-msca-p2", help="Model variant key")
    parser.add_argument("--image-size", type=int, default=320, help="Square inspection input size")
    return parser.parse_args()


def main() -> int:
    """Build and inspect a selected model."""
    args = parse_args()
    model = build_model(args.model, verbose=True)
    print(model)
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Detect strides: {model.stride.tolist()}")
    print(f"Detect feature shapes: {detection_feature_shapes(model, args.image_size)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
