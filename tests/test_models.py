"""Integration tests for all bundled model variants."""

import pytest
import torch

from yolo11_small_object_enhancement import MODEL_SPECS, build_model, predict_tensor
from yolo11_small_object_enhancement.modules import MSCA, MobileViTBlock


@pytest.mark.parametrize("model_key", MODEL_SPECS)
def test_each_model_builds_and_runs_inference(model_key: str) -> None:
    torch.manual_seed(42)
    model = build_model(model_key)
    output, padding = predict_tensor(model, torch.randn(1, 3, 129, 191))
    prediction = output[0] if isinstance(output, tuple) else output
    assert prediction.shape[:2] == (1, 14)
    assert padding == (31, 1)


def test_final_model_runs_training_mode_forward() -> None:
    model = build_model("mobilevit-msca-p2").train()
    output = model(torch.randn(2, 3, 64, 64))
    assert set(output) == {"boxes", "scores", "feats"}
    assert len(output["feats"]) == 4


def test_custom_variants_contain_expected_modules() -> None:
    assert any(isinstance(module, MobileViTBlock) for module in build_model("mobilevit").modules())
    assert any(isinstance(module, MSCA) for module in build_model("msca").modules())
    combined = build_model("mobilevit-msca")
    assert any(isinstance(module, MobileViTBlock) for module in combined.modules())
    assert any(isinstance(module, MSCA) for module in combined.modules())


def test_mobilevit_variant_stays_close_to_baseline_parameter_count() -> None:
    baseline = sum(parameter.numel() for parameter in build_model("baseline").parameters())
    mobilevit = sum(parameter.numel() for parameter in build_model("mobilevit").parameters())
    assert mobilevit < baseline * 1.25


def test_edge_student_parameter_budget() -> None:
    edge = sum(parameter.numel() for parameter in build_model("mobilevit-msca-p2-edge").parameters())
    full = sum(parameter.numel() for parameter in build_model("mobilevit-msca-p2").parameters())
    assert edge < 3_100_000
    assert edge < full * 0.3
