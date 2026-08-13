"""Run fast CPU forward checks for every model variant."""

from __future__ import annotations

import _bootstrap  # noqa: F401
import torch

from yolo11_small_object_enhancement import MODEL_SPECS, build_model, predict_tensor
from yolo11_small_object_enhancement.builder import describe_output


def main() -> int:
    """Execute deterministic 320-square inference without model weights."""
    torch.manual_seed(42)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    sample = torch.randn(1, 3, 321, 511)
    failures = 0
    for spec in MODEL_SPECS.values():
        try:
            model = build_model(spec.key)
            with torch.inference_mode():
                output, padding = predict_tensor(model, sample)
            print(f"[PASS] {spec.display_name}: padding={padding}, output={describe_output(output)}")
        except (RuntimeError, ValueError, KeyError) as error:
            failures += 1
            print(f"[FAIL] {spec.display_name}: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
