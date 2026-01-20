"""
YOLO Detection Wrapper

Handles model loading (with download if needed) and running inference.
"""

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
    bbox: list[float]  # [x1, y1, x2, y2]


@dataclass
class DetectionResult:
    """Detection results for a single camera image."""

    camera_id: str
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
        """Download the model if it doesn't exist locally."""
        if self.model_path.exists():
            return

        print(f"Downloading model from {self.model_url}...")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        response = httpx.get(self.model_url, follow_redirects=True, timeout=120.0)
        response.raise_for_status()

        with open(self.model_path, "wb") as f:
            f.write(response.content)

        print(f"Model downloaded to {self.model_path}")

    def load_model(self):
        """Load the YOLO model."""
        if self.model is not None:
            return

        self.ensure_model()
        print(f"Loading model from {self.model_path}")
        self.model = YOLO(str(self.model_path))

    def detect_from_bytes(self, image_bytes: bytes, camera_id: str) -> DetectionResult:
        """
        Run detection on image bytes.

        Args:
            image_bytes: Raw image bytes
            camera_id: Camera identifier for the result

        Returns:
            DetectionResult with counts and detection details
        """
        self.load_model()

        # Load image from bytes
        image = Image.open(BytesIO(image_bytes))

        # Run inference
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            verbose=False,
        )

        # Close image to free memory
        image.close()

        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    detection = Detection(
                        confidence=float(box.conf[0]),
                        bbox=box.xyxy[0].tolist(),
                    )
                    detections.append(detection)

        # Calculate average confidence
        avg_conf = None
        if detections:
            avg_conf = sum(d.confidence for d in detections) / len(detections)

        return DetectionResult(
            camera_id=camera_id,
            waymo_count=len(detections),
            detections=detections,
            avg_confidence=avg_conf,
        )

    def detect_from_pil(self, image: Image.Image, camera_id: str) -> DetectionResult:
        """
        Run detection on a PIL Image.

        Args:
            image: PIL Image object
            camera_id: Camera identifier for the result

        Returns:
            DetectionResult with counts and detection details
        """
        self.load_model()

        # Run inference
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            verbose=False,
        )

        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    detection = Detection(
                        confidence=float(box.conf[0]),
                        bbox=box.xyxy[0].tolist(),
                    )
                    detections.append(detection)

        # Calculate average confidence
        avg_conf = None
        if detections:
            avg_conf = sum(d.confidence for d in detections) / len(detections)

        return DetectionResult(
            camera_id=camera_id,
            waymo_count=len(detections),
            detections=detections,
            avg_confidence=avg_conf,
        )
