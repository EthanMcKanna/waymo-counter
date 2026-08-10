"""Authenticated Cloudflare Worker persistence for CCTV scan output."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from .cameras import Camera
from .cloudflare import CloudflareClient
from .detector import DetectionResult


class Database:
    def __init__(self, url: str, secret: str):
        self.client = CloudflareClient(url, secret)
        self._scans: dict[str, dict] = {}

    def create_scan(self, total_cameras: int, cameras_scanned: int, cameras_failed: int = 0,
                    total_waymo_count: int = 0, cameras_with_waymos: int = 0,
                    duration_seconds: Optional[float] = None) -> str:
        scan_id = str(uuid4())
        row = {"id": scan_id, "timestamp": datetime.now(timezone.utc).isoformat(),
               "total_cameras": total_cameras, "cameras_scanned": cameras_scanned,
               "cameras_failed": cameras_failed, "total_waymo_count": total_waymo_count,
               "cameras_with_waymos": cameras_with_waymos, "duration_seconds": duration_seconds}
        self._scans[scan_id] = row
        self.client.send_records("scans", [row])
        return scan_id

    def update_scan(self, scan_id: str, cameras_scanned: Optional[int] = None,
                    cameras_failed: Optional[int] = None, total_waymo_count: Optional[int] = None,
                    cameras_with_waymos: Optional[int] = None,
                    duration_seconds: Optional[float] = None):
        row = self._scans.get(scan_id, {"id": scan_id, "timestamp": datetime.now(timezone.utc).isoformat()})
        values = {"cameras_scanned": cameras_scanned, "cameras_failed": cameras_failed,
                  "total_waymo_count": total_waymo_count, "cameras_with_waymos": cameras_with_waymos,
                  "duration_seconds": round(duration_seconds, 2) if duration_seconds is not None else None}
        row.update({key: value for key, value in values.items() if value is not None})
        self._scans[scan_id] = row
        self.client.send_records("scans", [row])

    def list_recent_incomplete_scans(self, since: datetime, limit: int = 5) -> list[dict]:
        return []

    def insert_detection(self, scan_id: str, result: DetectionResult, image_url: Optional[str] = None):
        self.insert_detections(scan_id, [(result, image_url)])

    def insert_detections(self, scan_id: str, detections: list[tuple[DetectionResult, Optional[str]]]):
        rows = [self._serialize_detection(scan_id, result, image_url) for result, image_url in detections if result.waymo_count > 0]
        if rows:
            self.client.send_records("detections", rows)

    def insert_market_stats(self, scan_id: str, stats_rows: list[dict]):
        rows = [{"id": f"{scan_id}:{row['market']}", "scan_id": scan_id, **row,
                 "duration_seconds": round(row["duration_seconds"], 2)} for row in stats_rows]
        if rows:
            self.client.send_records("scan_market_stats", rows)

    def upsert_camera(self, camera: Camera):
        self.bulk_upsert_cameras([camera])

    def bulk_upsert_cameras(self, cameras: list[Camera]):
        now = datetime.now(timezone.utc).isoformat()
        rows = [{"id": camera.camera_key, "camera_key": camera.camera_key, "camera_id": camera.camera_id,
                 "market": camera.market, "source": camera.source, "location_name": camera.location_name,
                 "longitude": camera.longitude, "latitude": camera.latitude,
                 "council_district": camera.council_district, "image_url": camera.image_url,
                 "is_in_service_area": camera.is_in_service_area, "last_scanned": now, "updated_at": now}
                for camera in cameras]
        if rows:
            self.client.send_records("cameras", rows)

    @staticmethod
    def _serialize_detection(scan_id: str, result: DetectionResult, image_url: Optional[str]) -> dict:
        detected_at = datetime.now(timezone.utc).isoformat()
        return {"id": f"{scan_id}:{result.camera_key}", "scan_id": scan_id,
                "camera_key": result.camera_key, "camera_id": result.camera_id,
                "market": result.market, "source": result.source, "timestamp": detected_at,
                "waymo_count": result.waymo_count,
                "avg_confidence": round(result.avg_confidence, 4) if result.avg_confidence else None,
                "detections_json": [{"confidence": d.confidence, "bbox": d.bbox,
                                     **({"verifier_confidence": d.verifier_confidence} if d.verifier_confidence is not None else {})}
                                    for d in result.detections], "image_url": image_url}
