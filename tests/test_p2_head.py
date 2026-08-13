"""Runtime verification for the final model's four detection scales."""

from yolo11_small_object_enhancement import build_model, detection_feature_shapes


def test_final_model_detects_from_real_p2_through_p5_features() -> None:
    """Assert actual Detect inputs and calculated strides are P2/4-P5/32."""
    image_size = 128
    model = build_model("mobilevit-msca-p2")
    shapes = detection_feature_shapes(model, image_size)
    runtime_strides = [image_size // shape[-1] for shape in shapes]
    assert model.model[-1].nl == 4
    assert model.stride.int().tolist() == [4, 8, 16, 32]
    assert runtime_strides == [4, 8, 16, 32]
