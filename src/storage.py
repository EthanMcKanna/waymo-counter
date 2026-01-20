"""
Supabase Storage Module

Handles uploading detection images to Supabase Storage.
"""

from datetime import datetime, timezone
from typing import Optional

from supabase import Client


class ImageStorage:
    """Handles uploading images to Supabase Storage."""

    BUCKET_NAME = "detection-images"

    def __init__(self, client: Client):
        """
        Initialize the ImageStorage with a Supabase client.

        Args:
            client: Initialized Supabase client
        """
        self.client = client

    def upload_image(
        self,
        image_bytes: bytes,
        camera_id: str,
        timestamp: Optional[datetime] = None,
    ) -> Optional[str]:
        """
        Upload an image to Supabase Storage.

        Args:
            image_bytes: JPEG image bytes to upload
            camera_id: Camera identifier for organizing files
            timestamp: Timestamp for the image (defaults to now)

        Returns:
            Public URL of the uploaded image, or None if upload fails
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Build path: detections/{camera_id}/{YYYY-MM-DD}/{timestamp}.jpg
        date_str = timestamp.strftime("%Y-%m-%d")
        time_str = timestamp.strftime("%H%M%S")
        file_path = f"detections/{camera_id}/{date_str}/{time_str}.jpg"

        try:
            # Upload to Supabase Storage
            result = self.client.storage.from_(self.BUCKET_NAME).upload(
                path=file_path,
                file=image_bytes,
                file_options={
                    "content-type": "image/jpeg",
                    "cache-control": "public, max-age=31536000",  # 1 year cache
                },
            )

            # Get public URL
            public_url = self.client.storage.from_(self.BUCKET_NAME).get_public_url(file_path)
            return public_url

        except Exception as e:
            print(f"Failed to upload image for camera {camera_id}: {e}")
            return None
