"""
Austin CCTV Camera Management

Fetches active cameras from the Austin traffic camera API and filters
them to the Waymo service area.
"""

from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import httpx
from PIL import Image

from .service_area import point_in_polygon


@dataclass
class Camera:
    """Represents an Austin traffic camera."""

    camera_id: str
    location_name: str
    longitude: Optional[float]
    latitude: Optional[float]
    council_district: Optional[int]


class CameraFetcher:
    """Fetches and filters Austin traffic cameras."""

    API_BASE = "https://data.austintexas.gov/resource/b4k4-adkb.json"
    IMAGE_BASE = "https://cctv.austinmobility.io/image"

    def __init__(self, timeout: float = 30.0):
        self.client = httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(max_connections=20),
            follow_redirects=True,
        )

    def fetch_active_cameras(self, filter_to_service_area: bool = True) -> list[Camera]:
        """
        Fetch list of active cameras from Austin API.

        Args:
            filter_to_service_area: If True, only return cameras within Waymo service area

        Returns:
            List of Camera objects
        """
        params = {
            "$limit": 5000,
            "$where": "camera_status='TURNED_ON'",
        }

        response = self.client.get(self.API_BASE, params=params)
        response.raise_for_status()
        all_cameras = response.json()

        cameras = []
        for camera_data in all_cameras:
            location = camera_data.get("location", {})
            coords = location.get("coordinates", [])

            lon = coords[0] if coords and len(coords) >= 2 else None
            lat = coords[1] if coords and len(coords) >= 2 else None

            # Skip if filtering and camera is outside service area
            if filter_to_service_area:
                if lon is None or lat is None:
                    continue
                if not point_in_polygon(lon, lat):
                    continue

            # Parse council_district - may be "4, 7" for multiple districts
            council_district = None
            raw_district = camera_data.get("council_district")
            if raw_district is not None:
                try:
                    # Handle "4, 7" by taking first value
                    district_str = str(raw_district).split(",")[0].strip()
                    council_district = int(district_str)
                except (ValueError, TypeError):
                    pass

            camera = Camera(
                camera_id=camera_data.get("camera_id", ""),
                location_name=camera_data.get("location_name", ""),
                longitude=lon,
                latitude=lat,
                council_district=council_district,
            )
            cameras.append(camera)

        return cameras

    def fetch_image(self, camera_id: str) -> Optional[bytes]:
        """
        Fetch the current image from a camera.

        Args:
            camera_id: The camera ID

        Returns:
            Image bytes or None if failed
        """
        url = f"{self.IMAGE_BASE}/{camera_id}.jpg"

        try:
            response = self.client.get(url)
            response.raise_for_status()
            return response.content
        except httpx.HTTPError:
            return None

    def fetch_image_as_pil(self, camera_id: str) -> Optional[Image.Image]:
        """
        Fetch the current image from a camera as a PIL Image.

        Args:
            camera_id: The camera ID

        Returns:
            PIL Image or None if failed
        """
        image_bytes = self.fetch_image(camera_id)
        if image_bytes is None:
            return None

        try:
            return Image.open(BytesIO(image_bytes))
        except Exception:
            return None

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
