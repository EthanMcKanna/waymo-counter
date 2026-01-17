"""
Supabase Database Client

Handles all database operations for storing scan results and detections.
"""

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from supabase import create_client, Client

from .cameras import Camera
from .detector import DetectionResult


class Database:
    """Supabase database client for Waymo counter."""

    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)

    def create_scan(
        self,
        total_cameras: int,
        cameras_scanned: int,
        cameras_failed: int = 0,
        total_waymo_count: int = 0,
        cameras_with_waymos: int = 0,
        duration_seconds: Optional[float] = None,
    ) -> str:
        """
        Create a new scan record.

        Returns:
            The UUID of the created scan
        """
        data = {
            "total_cameras": total_cameras,
            "cameras_scanned": cameras_scanned,
            "cameras_failed": cameras_failed,
            "total_waymo_count": total_waymo_count,
            "cameras_with_waymos": cameras_with_waymos,
        }

        if duration_seconds is not None:
            data["duration_seconds"] = round(duration_seconds, 2)

        result = self.client.table("scans").insert(data).execute()
        return result.data[0]["id"]

    def update_scan(
        self,
        scan_id: str,
        cameras_scanned: Optional[int] = None,
        cameras_failed: Optional[int] = None,
        total_waymo_count: Optional[int] = None,
        cameras_with_waymos: Optional[int] = None,
        duration_seconds: Optional[float] = None,
    ):
        """Update an existing scan record."""
        data = {}

        if cameras_scanned is not None:
            data["cameras_scanned"] = cameras_scanned
        if cameras_failed is not None:
            data["cameras_failed"] = cameras_failed
        if total_waymo_count is not None:
            data["total_waymo_count"] = total_waymo_count
        if cameras_with_waymos is not None:
            data["cameras_with_waymos"] = cameras_with_waymos
        if duration_seconds is not None:
            data["duration_seconds"] = round(duration_seconds, 2)

        if data:
            self.client.table("scans").update(data).eq("id", scan_id).execute()

    def insert_detection(
        self,
        scan_id: str,
        result: DetectionResult,
    ):
        """
        Insert a detection record.

        Args:
            scan_id: The parent scan UUID
            result: Detection result from the detector
        """
        # Only insert if there were detections
        if result.waymo_count == 0:
            return

        # Convert detections to JSON-serializable format
        detections_json = [
            {"confidence": d.confidence, "bbox": d.bbox}
            for d in result.detections
        ]

        data = {
            "scan_id": scan_id,
            "camera_id": result.camera_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "waymo_count": result.waymo_count,
            "avg_confidence": round(result.avg_confidence, 4) if result.avg_confidence else None,
            "detections_json": detections_json,
        }

        self.client.table("detections").insert(data).execute()

    def upsert_camera(self, camera: Camera):
        """
        Upsert camera metadata.

        Args:
            camera: Camera object with metadata
        """
        data = {
            "camera_id": camera.camera_id,
            "location_name": camera.location_name,
            "longitude": camera.longitude,
            "latitude": camera.latitude,
            "council_district": camera.council_district,
            "last_scanned": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        self.client.table("cameras").upsert(data, on_conflict="camera_id").execute()

    def bulk_upsert_cameras(self, cameras: list[Camera]):
        """
        Bulk upsert camera metadata.

        Args:
            cameras: List of Camera objects
        """
        now = datetime.now(timezone.utc).isoformat()

        data = [
            {
                "camera_id": c.camera_id,
                "location_name": c.location_name,
                "longitude": c.longitude,
                "latitude": c.latitude,
                "council_district": c.council_district,
                "last_scanned": now,
                "updated_at": now,
            }
            for c in cameras
        ]

        # Supabase has a limit on bulk inserts, batch if needed
        batch_size = 500
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            self.client.table("cameras").upsert(batch, on_conflict="camera_id").execute()
