import importlib.util
from pathlib import Path

from PIL import Image


def load_script_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


EXPORTER = load_script_module(
    "export_detection_review_set",
    "scripts/export_detection_review_set.py",
)
BUILDER = load_script_module(
    "build_verifier_dataset",
    "scripts/build_verifier_dataset.py",
)


def test_slugify_removes_path_unsafe_characters():
    assert EXPORTER.slugify("austin:camera/id 123") == "austin_camera_id_123"


def test_infer_domain_marks_highway_locations():
    row = {"market": "houston", "camera_id": "123", "camera_key": "k", "source": "txdot"}
    camera = {"location_name": "IH-10 @ Shepherd"}
    assert EXPORTER.infer_domain(row, camera) == "highway"


def test_crop_box_scales_large_model_coordinates_to_saved_image():
    image = Image.new("RGB", (800, 450))
    bbox = [800, 300, 1000, 420]
    assert BUILDER.crop_box(image, bbox, padding=0.0) == (640, 240, 800, 336)


def test_assign_splits_keeps_rows_grouped_by_market_domain_label():
    rows = [
        {
            "market": "austin",
            "domain": "urban",
            "review_label": "waymo",
            "candidate_id": str(index),
        }
        for index in range(10)
    ]
    splits = BUILDER.assign_splits(rows, seed=1, val_frac=0.2, test_frac=0.2)
    assert len(splits["test"]) == 2
    assert len(splits["val"]) == 2
    assert len(splits["train"]) == 6
