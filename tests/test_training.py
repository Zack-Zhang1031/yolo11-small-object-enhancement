"""Contract tests for Ultralytics wrapper construction."""

from ultralytics import YOLO

from yolo11_small_object_enhancement import create_yolo
from yolo11_small_object_enhancement.modules import MobileViTBlock


def test_create_yolo_registers_custom_layers() -> None:
    model = create_yolo("mobilevit-msca-p2")
    assert isinstance(model, YOLO)
    assert any(isinstance(module, MobileViTBlock) for module in model.model.modules())
