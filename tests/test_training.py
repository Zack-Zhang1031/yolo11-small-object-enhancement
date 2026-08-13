"""Contract tests for Ultralytics training and distillation construction."""

import pytest
from ultralytics import YOLO

from yolo11_small_object_enhancement import create_yolo, validate_distillation_pair
from yolo11_small_object_enhancement.modules import MobileViTBlock


def test_create_yolo_registers_custom_layers() -> None:
    model = create_yolo("mobilevit-msca-p2")
    assert isinstance(model, YOLO)
    assert any(isinstance(module, MobileViTBlock) for module in model.model.modules())


def test_edge_student_uses_four_detection_scales() -> None:
    model = create_yolo("mobilevit-msca-p2-edge")
    assert model.model.stride.tolist() == [4.0, 8.0, 16.0, 32.0]


def test_full_and_edge_variants_are_distillation_compatible() -> None:
    validate_distillation_pair(
        create_yolo("mobilevit-msca-p2-edge"), create_yolo("mobilevit-msca-p2")
    )


def test_incompatible_teacher_is_rejected() -> None:
    with pytest.raises(ValueError, match="matching layer indices"):
        validate_distillation_pair(create_yolo("mobilevit-msca-p2-edge"), create_yolo("baseline"))
