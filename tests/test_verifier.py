from pathlib import Path

from PIL import Image

from src.detector import Detection, DetectionResult
from src.verifier import WaymoVerifier, crop_box, preprocess_crop


class StubVerifier(WaymoVerifier):
    def __init__(self, scores):
        super().__init__(
            model_path=Path("unused.pt"),
            model_url="https://example.com/unused.pt",
            austin_threshold=0.475,
            non_austin_threshold=0.90,
        )
        self.scores = list(scores)

    def score_detection(self, image, detection):
        return self.scores.pop(0)


def test_threshold_for_market_uses_stricter_non_austin_policy():
    verifier = StubVerifier([])

    assert verifier.threshold_for_market("austin") == 0.475
    assert verifier.threshold_for_market("houston") == 0.90


def test_crop_box_pads_and_clamps_to_image_bounds():
    image = Image.new("RGB", (100, 60))

    assert crop_box(image, [5, 4, 25, 24], padding=0.5) == (0, 0, 35, 34)


def test_preprocess_crop_outputs_normalized_batch_tensor():
    image = Image.new("RGB", (20, 10), color=(127, 127, 127))

    tensor = preprocess_crop(image, image_size=32)

    assert tuple(tensor.shape) == (1, 3, 32, 32)


def test_verify_result_filters_rejected_proposals_and_preserves_scores():
    image = Image.new("RGB", (100, 80))
    result = DetectionResult(
        camera_key="austin:austin_cctv:1",
        camera_id="1",
        market="austin",
        source="austin_cctv",
        waymo_count=2,
        detections=[
            Detection(confidence=0.8, bbox=[10, 10, 20, 20]),
            Detection(confidence=0.9, bbox=[30, 30, 40, 40]),
        ],
        avg_confidence=0.85,
    )
    verifier = StubVerifier([0.3, 0.8])

    filtered = verifier.verify_result(image, result)

    assert filtered.waymo_count == 1
    assert filtered.avg_confidence == 0.9
    assert filtered.detections[0].bbox == [30, 30, 40, 40]
    assert filtered.detections[0].verifier_confidence == 0.8
