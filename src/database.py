"""
Supabase Database Client

Handles database operations for scans, detections, and camera metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from supabase import Client, create_client

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

    def list_recent_incomplete_scans(
        self,
        since: datetime,
        limit: int = 5,
    ) -> list[dict]:
        result = (
            self.client.table("scans")
            .select(
                "id,timestamp,total_cameras,cameras_scanned,cameras_failed,"
                "duration_seconds"
            )
            .filter("duration_seconds", "is", "null")
            .gte("timestamp", since.isoformat())
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []

    def insert_detection(
        self,
        scan_id: str,
        result: DetectionResult,
        image_url: Optional[str] = None,
    ):
        if result.waymo_count == 0:
            return
        self.insert_detections(scan_id, [(result, image_url)])

    def insert_detections(
        self,
        scan_id: str,
        detections: list[tuple[DetectionResult, Optional[str]]],
    ):
        rows = [
            self._serialize_detection(scan_id, result, image_url)
            for result, image_url in detections
            if result.waymo_count > 0
        ]
        if rows:
            self.client.table("detections").insert(rows).execute()

    def insert_market_stats(self, scan_id: str, stats_rows: list[dict]):
        if not stats_rows:
            return

        rows = []
        for row in stats_rows:
            rows.append(
                {
                    "scan_id": scan_id,
                    "market": row["market"],
                    "total_cameras": row["total_cameras"],
                    "cameras_scanned": row["cameras_scanned"],
                    "cameras_failed": row["cameras_failed"],
                    "total_waymo_count": row["total_waymo_count"],
                    "cameras_with_waymos": row["cameras_with_waymos"],
                    "duration_seconds": round(row["duration_seconds"], 2),
                }
            )

        self.client.table("scan_market_stats").upsert(
            rows,
            on_conflict="scan_id,market",
        ).execute()

    def upsert_camera(self, camera: Camera):
        now = datetime.now(timezone.utc).isoformat()
        self.client.table("cameras").upsert(
            {
                "camera_key": camera.camera_key,
                "camera_id": camera.camera_id,
                "market": camera.market,
                "source": camera.source,
                "location_name": camera.location_name,
                "longitude": camera.longitude,
                "latitude": camera.latitude,
                "council_district": camera.council_district,
                "image_url": camera.image_url,
                "is_in_service_area": camera.is_in_service_area,
                "last_scanned": now,
                "updated_at": now,
            },
            on_conflict="camera_key",
        ).execute()

    def bulk_upsert_cameras(self, cameras: list[Camera]):
        if not cameras:
            return

        now = datetime.now(timezone.utc).isoformat()
        data = [
            {
                "camera_key": camera.camera_key,
                "camera_id": camera.camera_id,
                "market": camera.market,
                "source": camera.source,
                "location_name": camera.location_name,
                "longitude": camera.longitude,
                "latitude": camera.latitude,
                "council_district": camera.council_district,
                "image_url": camera.image_url,
                "is_in_service_area": camera.is_in_service_area,
                "last_scanned": now,
                "updated_at": now,
            }
            for camera in cameras
        ]

        batch_size = 500
        for start in range(0, len(data), batch_size):
            batch = data[start:start + batch_size]
            self.client.table("cameras").upsert(
                batch,
                on_conflict="camera_key",
            ).execute()

    @staticmethod
    def _serialize_detection(
        scan_id: str,
        result: DetectionResult,
        image_url: Optional[str],
    ) -> dict:
        detections_json = [
            {"confidence": detection.confidence, "bbox": detection.bbox}
            for detection in result.detections
        ]

        return {
            "scan_id": scan_id,
            "camera_key": result.camera_key,
            "camera_id": result.camera_id,
            "market": result.market,
            "source": result.source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "waymo_count": result.waymo_count,
            "avg_confidence": round(result.avg_confidence, 4) if result.avg_confidence else None,
            "detections_json": detections_json,
            "image_url": image_url,
        }
