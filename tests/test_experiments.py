"""Tests for controlled comparison definitions and report serialization."""

from __future__ import annotations

import csv
import json

import pytest

from yolo11_small_object_enhancement.experiments import (
    COMPARISON_SPECS,
    ComparisonResult,
    create_comparison_model,
    get_comparison_spec,
    write_comparison_report,
)


def test_comparison_matrix_contains_all_required_families() -> None:
    assert set(COMPARISON_SPECS) == {"yolo11s", "enhanced", "yolo12s", "rtdetr-l"}


def test_unknown_comparison_model_is_rejected() -> None:
    with pytest.raises(KeyError, match="Unknown comparison model"):
        get_comparison_spec("missing")


def test_enhanced_comparison_model_builds_without_download() -> None:
    model = create_comparison_model("enhanced", pretrained=False)
    assert model.model.stride.tolist() == [4.0, 8.0, 16.0, 32.0]


def test_comparison_report_writes_matching_json_and_csv(tmp_path) -> None:
    result = ComparisonResult("enhanced", "best.pt", 3_000_000, 6.0, 0.4, 0.7, 0.45, 0.1, 2.0, 0.3)
    json_path, csv_path = write_comparison_report([result], tmp_path, {"seed": 42})

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert payload["metadata"]["seed"] == 42
    assert payload["results"][0]["map50_95"] == 0.4
    assert rows[0]["key"] == "enhanced"
