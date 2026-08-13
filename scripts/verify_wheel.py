"""Install a built wheel and verify bundled resources outside the repository."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse the wheel path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    return parser.parse_args()


def main() -> int:
    """Verify wheel contents, install it, and build the final model from a temp directory."""
    wheel = parse_args().wheel.resolve()
    if not wheel.is_file():
        raise FileNotFoundError(wheel)
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    configs = [name for name in names if name.endswith(".yaml")]
    if len(configs) != 6:
        raise RuntimeError(f"expected 6 bundled model configs, found {len(configs)}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheel)],
        check=True,
    )
    code = (
        "from yolo11_small_object_enhancement import build_model, create_yolo; "
        "model=build_model('mobilevit-msca-p2'); "
        "assert model.stride.int().tolist()==[4,8,16,32]; "
        "edge=build_model('mobilevit-msca-p2-edge'); "
        "assert sum(p.numel() for p in edge.parameters())<3100000; "
        "assert create_yolo('mobilevit-msca-p2').model is not None; "
        "print('Wheel install: PASS')"
    )
    with tempfile.TemporaryDirectory() as directory:
        subprocess.run([sys.executable, "-c", code], cwd=directory, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
