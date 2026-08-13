"""Download, convert, and verify the official VisDrone detection dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from ultralytics.data.utils import check_det_dataset


def parse_args() -> argparse.Namespace:
    """Parse the dataset definition and download policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="VisDrone.yaml", help="Ultralytics YAML name or local YAML path")
    parser.add_argument("--check-only", action="store_true", help="Verify existing files without downloading")
    return parser.parse_args()


def main() -> int:
    """Resolve all splits and report the usable dataset location."""
    args = parse_args()
    definition = str(Path(args.data).resolve()) if Path(args.data).is_file() else args.data
    data = check_det_dataset(definition, autodownload=not args.check_only)
    train = Path(data["train"])
    val = Path(data["val"])
    if not train.exists() or not val.exists():
        raise FileNotFoundError("VisDrone train/val images were not resolved")
    print(f"VisDrone dataset: PASS (root={data['path']}, classes={data['nc']})")
    print(f"Train images: {train}")
    print(f"Validation images: {val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
