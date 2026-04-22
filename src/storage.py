"""
Supabase Storage Module

Handles uploading detection images to Supabase Storage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from supabase import Client

from .cameras import Camera


class ImageStorage:
    """Handles uploading images to Supabase Storage."""

    BUCKET_NAME = "detection-images"

    def __init__(self, client: Client):
        self.client = client

    def upload_image(
        self,
        image_bytes: bytes,
        camera: Camera,
        timestamp: Optional[datetime] = None,
    ) -> Optional[str]:
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        date_str = timestamp.strftime("%Y-%m-%d")
        time_str = timestamp.strftime("%H%M%S")
        file_path = (
            f"detections/{camera.market}/{camera.source}/{camera.area_label}/"
            f"{camera.storage_slug}/{date_str}/{time_str}.jpg"
        )

        try:
            self.client.storage.from_(self.BUCKET_NAME).upload(
                path=file_path,
                file=image_bytes,
                file_options={
                    "content-type": "image/jpeg",
                    "cache-control": "public, max-age=31536000",
                },
            )
            return self.client.storage.from_(self.BUCKET_NAME).get_public_url(file_path)
        except Exception as exc:
            print(f"Failed to upload image for camera {camera.camera_key}: {exc}")
            return None
