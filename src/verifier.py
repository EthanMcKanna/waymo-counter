"""
Second-stage Waymo crop verifier.

The YOLO model produces candidate vehicle boxes. This verifier scores each crop
and filters likely false positives before images or counts are persisted.
"""

from __future__ import annotations

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
        device: Optional[torch.device] = None,
    ):
        self.model_path = model_path
        self.model_url = model_url
        self.image_size = image_size
        self.crop_padding = crop_padding
        self.austin_threshold = austin_threshold
        self.non_austin_threshold = non_austin_threshold
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
    def score_detection(self, image: Image.Image, detection: Detection) -> float:
        self.load_model()
        assert self.model is not None

        crop = image.crop(crop_box(image, detection.bbox, self.crop_padding))
        tensor = preprocess_crop(crop, self.image_size).to(self.device)
        logits = self.model(tensor)
        probability = F.softmax(logits, dim=1)[0, 1].detach().cpu().item()
        crop.close()
        return float(probability)

    def verify_result(self, image: Image.Image, result: DetectionResult) -> DetectionResult:
        if not result.detections:
            return result

        threshold = self.threshold_for_market(result.market)
        accepted: list[Detection] = []
        for detection in result.detections:
            verifier_confidence = self.score_detection(image, detection)
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
