"""
YOLO Detection Wrapper

Handles model loading (with download if needed) and running inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import httpx
from PIL import Image
from ultralytics import YOLO


@dataclass
class Detection:
    """A single Waymo detection result."""

    confidence: float
    bbox: list[float]


@dataclass
class DetectionResult:
    """Detection results for a single camera image."""

    camera_key: str
    camera_id: str
    market: str
    source: str
    waymo_count: int
    detections: list[Detection]
    avg_confidence: Optional[float]


class WaymoDetector:
    """YOLO-based Waymo vehicle detector."""

    def __init__(
        self,
        model_path: Path,
        model_url: str,
        confidence_threshold: float = 0.50,
    ):
        self.model_path = model_path
        self.model_url = model_url
        self.confidence_threshold = confidence_threshold
        self.model: Optional[YOLO] = None

    def ensure_model(self):
        if self.model_path.exists():
            return

        print(
            f"Model weights not found at {self.model_path}. "
            "Falling back to runtime download."
        )
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        response = httpx.get(self.model_url, follow_redirects=True, timeout=120.0)
        response.raise_for_status()
        self.model_path.write_bytes(response.content)
        print(f"Model downloaded to {self.model_path}")

    def load_model(self):
        if self.model is not None:
            return

        self.ensure_model()
        print(f"Loading model from {self.model_path}")
        self.model = YOLO(str(self.model_path))

    def _build_result(
        self,
        results,
        camera_key: str,
        camera_id: str,
        market: str,
        source: str,
    ) -> DetectionResult:
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                detections.append(
                    Detection(
                        confidence=float(box.conf[0]),
                        bbox=box.xyxy[0].tolist(),
                    )
                )

        avg_conf = None
        if detections:
            avg_conf = sum(d.confidence for d in detections) / len(detections)

        return DetectionResult(
            camera_key=camera_key,
            camera_id=camera_id,
            market=market,
            source=source,
            waymo_count=len(detections),
            detections=detections,
            avg_confidence=avg_conf,
        )

    def detect_from_bytes(
        self,
        image_bytes: bytes,
        camera_key: str,
        camera_id: str,
        market: str,
        source: str,
    ) -> DetectionResult:
        self.load_model()
        image = Image.open(BytesIO(image_bytes))
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            verbose=False,
        )
        image.close()
        return self._build_result(results, camera_key, camera_id, market, source)

    def detect_from_pil(
        self,
        image: Image.Image,
        camera_key: str,
        camera_id: str,
        market: str,
        source: str,
    ) -> DetectionResult:
        self.load_model()
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            verbose=False,
        )
        return self._build_result(results, camera_key, camera_id, market, source)
