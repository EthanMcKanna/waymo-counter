"""
Second-stage Waymo crop verifier.

The YOLO model produces candidate vehicle boxes. This verifier scores each crop
and filters likely false positives before images or counts are persisted.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .detector import Detection, DetectionResult


IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)

CALIBRATOR_MARKETS = (
    "atlanta",
    "austin",
    "dallas",
    "houston",
    "miami",
    "orlando",
    "phoenix",
    "san_antonio",
)
CALIBRATOR_MEAN = np.array(
    (
        -0.04827887937426567,
        0.39634987711906433,
        0.777036190032959,
        4.330371379852295,
        4.1257734298706055,
        8.456143379211426,
        1.256852626800537,
        0.13983051478862762,
        0.4053672254085541,
        0.09604519605636597,
        0.19491524994373322,
        0.007062146905809641,
        0.005649717524647713,
        0.11864406615495682,
        0.032485876232385635,
    ),
    dtype=np.float32,
)
CALIBRATOR_STD = np.array(
    (
        5.3117218017578125,
        0.456235408782959,
        0.14048679172992706,
        0.9266569018363953,
        0.8842604160308838,
        1.7969250679016113,
        0.2679489850997925,
        0.3468121886253357,
        0.490964412689209,
        0.2946540117263794,
        0.39613741636276245,
        0.08374043554067612,
        0.07495298981666565,
        0.3233698904514313,
        0.17728760838508606,
    ),
    dtype=np.float32,
)
CALIBRATOR_WEIGHTS = np.array(
    (
        0.5404242873191833,
        0.5796090960502625,
        0.258226603269577,
        0.20560230314731598,
        0.2039840817451477,
        0.20640695095062256,
        0.03208533301949501,
        -0.06381717324256897,
        0.25102540850639343,
        -0.08912207186222076,
        -0.11937563866376877,
        -0.031208360567688942,
        0.05146217346191406,
        -0.05951080471277237,
        -0.05393761023879051,
    ),
    dtype=np.float32,
)
CALIBRATOR_BIAS = -1.0096935033798218


def crop_box(
    image: Image.Image,
    bbox: list[float],
    padding: float = 0.35,
) -> tuple[int, int, int, int]:
    """Return a padded, clamped crop box for an xyxy detection."""

    width, height = image.size
    x1, y1, x2, y2 = [float(value) for value in bbox]
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    pad_x = box_width * padding
    pad_y = box_height * padding

    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(width, int(round(x2 + pad_x)))
    bottom = min(height, int(round(y2 + pad_y)))

    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)
    return left, top, right, bottom


def preprocess_crop(crop: Image.Image, image_size: int) -> torch.Tensor:
    """Convert a PIL crop to the normalized verifier tensor."""

    resized = crop.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


class WaymoVerifier:
    """TorchScript crop verifier for filtering YOLO proposals."""

    def __init__(
        self,
        model_path: Path,
        model_url: str,
        image_size: int = 224,
        crop_padding: float = 0.35,
        austin_threshold: float = 0.475,
        non_austin_threshold: float = 0.90,
        calibration_enabled: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.model_path = model_path
        self.model_url = model_url
        self.image_size = image_size
        self.crop_padding = crop_padding
        self.austin_threshold = austin_threshold
        self.non_austin_threshold = non_austin_threshold
        self.calibration_enabled = calibration_enabled
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.jit.ScriptModule] = None

    def ensure_model(self) -> None:
        if self.model_path.exists() and self.model_path.stat().st_size > 0:
            return

        print(
            f"Verifier weights not found at {self.model_path}. "
            "Falling back to runtime download."
        )
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        response = httpx.get(self.model_url, follow_redirects=True, timeout=120.0)
        response.raise_for_status()
        self.model_path.write_bytes(response.content)
        print(f"Verifier downloaded to {self.model_path}")

    def load_model(self) -> None:
        if self.model is not None:
            return

        self.ensure_model()
        print(f"Loading verifier from {self.model_path}")
        self.model = torch.jit.load(str(self.model_path), map_location=self.device)
        self.model.eval()

    def threshold_for_market(self, market: str) -> float:
        if market.lower() == "austin":
            return self.austin_threshold
        return self.non_austin_threshold

    @torch.inference_mode()
    def calibrated_score(
        self,
        raw_probability: float,
        detection: Detection,
        market: str,
    ) -> float:
        x1, y1, x2, y2 = [float(value) for value in detection.bbox]
        width = max(1e-3, x2 - x1)
        height = max(1e-3, y2 - y1)
        area = width * height
        probability = min(1.0 - 1e-5, max(1e-5, raw_probability))
        logit = math.log(probability / (1.0 - probability))

        features = np.array(
            (
                logit,
                raw_probability,
                detection.confidence,
                math.log(width),
                math.log(height),
                math.log(area),
                width / height,
                *(1.0 if market.lower() == item else 0.0 for item in CALIBRATOR_MARKETS),
            ),
            dtype=np.float32,
        )
        normalized = (features - CALIBRATOR_MEAN) / CALIBRATOR_STD
        calibrated_logit = float(np.dot(normalized, CALIBRATOR_WEIGHTS) + CALIBRATOR_BIAS)
        return 1.0 / (1.0 + math.exp(-calibrated_logit))

    @torch.inference_mode()
    def score_detection(self, image: Image.Image, detection: Detection) -> float:
        self.load_model()
        assert self.model is not None

        crop = image.crop(crop_box(image, detection.bbox, self.crop_padding))
        tensor = preprocess_crop(crop, self.image_size).to(self.device)
        logits = self.model(tensor)
        probability = F.softmax(logits, dim=1)[0, 1].detach().cpu().item()
        crop.close()
        return float(probability)

    def score_detection_for_market(
        self,
        image: Image.Image,
        detection: Detection,
        market: str,
    ) -> float:
        raw_probability = self.score_detection(image, detection)
        if not self.calibration_enabled:
            return raw_probability
        return self.calibrated_score(raw_probability, detection, market)

    def verify_result(self, image: Image.Image, result: DetectionResult) -> DetectionResult:
        if not result.detections:
            return result

        threshold = self.threshold_for_market(result.market)
        accepted: list[Detection] = []
        for detection in result.detections:
            verifier_confidence = self.score_detection_for_market(image, detection, result.market)
            if verifier_confidence >= threshold:
                accepted.append(
                    Detection(
                        confidence=detection.confidence,
                        bbox=list(detection.bbox),
                        verifier_confidence=verifier_confidence,
                    )
                )

        avg_confidence = None
        if accepted:
            avg_confidence = sum(detection.confidence for detection in accepted) / len(accepted)

        return DetectionResult(
            camera_key=result.camera_key,
            camera_id=result.camera_id,
            market=result.market,
            source=result.source,
            waymo_count=len(accepted),
            detections=accepted,
            avg_confidence=avg_confidence,
        )
