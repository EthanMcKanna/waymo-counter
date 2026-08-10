"""Private R2-backed detection image uploads through the Worker API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from .cameras import Camera
from .cloudflare import CloudflareClient


class ImageStorage:
    def __init__(self, client: CloudflareClient):
        self.client = client

    def upload_image(self, image_bytes: bytes, camera: Camera,
                     timestamp: Optional[datetime] = None) -> Optional[str]:
        timestamp = timestamp or datetime.now(timezone.utc)
        storage_id = f"{camera.market}-{camera.source}-{camera.storage_slug}-{timestamp.strftime('%Y%m%d%H%M%S')}"
        try:
            return self.client.upload_media(image_bytes, storage_id)
        except Exception as exc:
            print(f"Failed to upload image for camera {camera.camera_key}: {type(exc).__name__}")
            return None
