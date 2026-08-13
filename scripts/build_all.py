"""Build every model variant and report parameters and detection scales."""

from __future__ import annotations

import _bootstrap  # noqa: F401

from yolo11_small_object_enhancement import MODEL_SPECS, build_model, count_parameters


def main() -> int:
    """Build all variants and return a process status code."""
    failures = 0
    print("=" * 64)
    for spec in MODEL_SPECS.values():
        print(spec.display_name)
        try:
            model = build_model(spec.key)
            strides = [int(value) for value in model.stride.tolist()]
            print("Build: PASS")
            print(f"Params: {count_parameters(model) / 1_000_000:.3f} M")
            scales = ", ".join(f"P{stride.bit_length() - 1}/{stride}" for stride in strides)
            print(f"Detection scales: {scales}")
        except (RuntimeError, ValueError, KeyError) as error:
            failures += 1
            print(f"Build: FAIL ({error})")
        print("-" * 64)
    print("=" * 64)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
