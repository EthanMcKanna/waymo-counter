"""
Waymo Counter - Main Orchestration Script

This is the entry point for the cron job. It:
1. Fetches active cameras across enabled markets
2. Downloads images and runs YOLO detection
3. Uploads results to Supabase
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

from .cameras import Camera, CameraFetcher
from .config import load_config
from .database import Database
from .detector import DetectionResult, WaymoDetector
from .image_annotator import annotate_image, compress_image
from .storage import ImageStorage
from .verifier import WaymoVerifier


def fetch_camera_image(
    camera: Camera,
    camera_fetcher: CameraFetcher,
) -> tuple[Camera, bytes | None, str | None]:
    image_bytes = camera_fetcher.fetch_image(camera)
    if image_bytes is None:
        return (camera, None, "Failed to fetch image")
    return (camera, image_bytes, None)


def process_fetched_camera(
    camera: Camera,
    image_bytes: bytes,
    detector: WaymoDetector,
    verifier: WaymoVerifier | None = None,
    image_storage: ImageStorage | None = None,
) -> tuple[Camera, DetectionResult | None, str | None, str | None]:
    try:
        result = detector.detect_from_bytes(
            image_bytes,
            camera_key=camera.camera_key,
            camera_id=camera.camera_id,
            market=camera.market,
            source=camera.source,
        )

        image_url = None
        if result.waymo_count > 0 and (verifier or image_storage):
            from io import BytesIO
            from PIL import Image

            timestamp = datetime.now(timezone.utc)
            image = Image.open(BytesIO(image_bytes))
            if verifier:
                result = verifier.verify_result(image, result)
            if result.waymo_count > 0 and image_storage:
                annotated = annotate_image(image, result.detections)
                compressed = compress_image(annotated)
                image_url = image_storage.upload_image(compressed, camera, timestamp)
                annotated.close()
                del annotated
                del compressed
            image.close()
            del image

        del image_bytes
        return (camera, result, None, image_url)
    except Exception as exc:
        return (camera, None, str(exc), None)


def run_scan():
    start_time = time.time()

    print("=" * 60)
    print(f"Waymo Counter Scan - {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    print("\nLoading configuration...")
    config = load_config()
    print(f"Enabled markets: {', '.join(config.enabled_markets)}")

    print("Initializing components...")
    db = Database(config.supabase_url, config.supabase_key)
    if config.scan_lock_minutes > 0:
        lock_cutoff = datetime.now(timezone.utc) - timedelta(
            minutes=config.scan_lock_minutes
        )
        active_scans = db.list_recent_incomplete_scans(lock_cutoff)
        if active_scans:
            print(
                "Another scan appears to be running or recently stalled. "
                f"Found {len(active_scans)} incomplete scan(s) since "
                f"{lock_cutoff.isoformat()}."
            )
            for scan in active_scans:
                print(
                    "  "
                    f"{scan.get('id')} at {scan.get('timestamp')}: "
                    f"{scan.get('cameras_scanned')}/{scan.get('total_cameras')} "
                    f"scanned, {scan.get('cameras_failed')} failed"
                )
            print("Exiting without starting a duplicate scan.")
            return

    image_storage = ImageStorage(db.client)
    detector = WaymoDetector(
        model_path=config.model_path,
        model_url=config.model_url,
        confidence_threshold=config.confidence_threshold,
        image_size=config.model_image_size,
    )
    verifier = None
    if config.verifier_enabled:
        verifier = WaymoVerifier(
            model_path=config.verifier_model_path,
            model_url=config.verifier_model_url,
            image_size=config.verifier_image_size,
            crop_padding=config.verifier_crop_padding,
            austin_threshold=config.verifier_threshold,
            non_austin_threshold=config.verifier_non_austin_threshold,
            calibration_enabled=config.verifier_calibration_enabled,
        )

    print("Loading detection model...")
    detector.load_model()
    if verifier:
        print("Loading second-stage verifier...")
        verifier.load_model()

    print("\nFetching active cameras...")
    with CameraFetcher(config=config) as camera_fetcher:
        cameras = camera_fetcher.fetch_active_cameras()
        if not cameras:
            print("No cameras found. Exiting.")
            return

        total_cameras_by_market: dict[str, int] = defaultdict(int)
        for camera in cameras:
            total_cameras_by_market[camera.market] += 1

        austin_cameras = [camera for camera in cameras if camera.market == "austin"]
        austin_inside = sum(1 for camera in austin_cameras if camera.is_in_service_area)
        austin_outside = len(austin_cameras) - austin_inside

        print(f"Found {len(cameras)} active cameras across {len(total_cameras_by_market)} markets")
        for market, total in sorted(total_cameras_by_market.items()):
            print(f"  {market}: {total} camera(s)")
        if austin_cameras:
            print(f"  austin inside service area: {austin_inside}")
            print(f"  austin outside service area: {austin_outside}")

        scan_id = db.create_scan(
            total_cameras=len(cameras),
            cameras_scanned=0,
            cameras_failed=0,
            total_waymo_count=0,
            cameras_with_waymos=0,
        )
        print(f"Created scan record: {scan_id}")

        market_stats: dict[str, dict] = {
            market: {
                "market": market,
                "total_cameras": total,
                "cameras_scanned": 0,
                "cameras_failed": 0,
                "total_waymo_count": 0,
                "cameras_with_waymos": 0,
                "duration_seconds": 0.0,
            }
            for market, total in total_cameras_by_market.items()
        }
        market_start_times = {market: time.time() for market in total_cameras_by_market}

        cameras_scanned = 0
        cameras_failed = 0
        total_waymo_count = 0
        cameras_with_waymos = 0
        inside_area_waymo_count = 0
        outside_area_waymo_count = 0
        inside_area_cameras_with_waymos = 0
        outside_area_cameras_with_waymos = 0
        batched_detections: list[tuple[DetectionResult, str | None]] = []

        print(f"\nFetching with {config.fetch_workers} workers and running inference inline...")
        with ThreadPoolExecutor(max_workers=config.fetch_workers) as executor:
            futures = {
                executor.submit(fetch_camera_image, camera, camera_fetcher): camera
                for camera in cameras
            }

            for future in as_completed(futures):
                camera, image_bytes, fetch_error = future.result()
                market_row = market_stats[camera.market]
                market_row["duration_seconds"] = time.time() - market_start_times[camera.market]
                progress = cameras_scanned + cameras_failed

                if fetch_error or image_bytes is None:
                    cameras_failed += 1
                    market_row["cameras_failed"] += 1
                    print(
                        f"  [{progress + 1}/{len(cameras)}] "
                        f"{camera.market}:{camera.camera_id}: ERROR - {fetch_error}"
                    )
                    continue

                camera, result, error, image_url = process_fetched_camera(
                    camera,
                    image_bytes,
                    detector,
                    verifier,
                    image_storage,
                )
                del image_bytes

                market_row["duration_seconds"] = time.time() - market_start_times[camera.market]
                if error or result is None:
                    cameras_failed += 1
                    market_row["cameras_failed"] += 1
                    print(
                        f"  [{progress + 1}/{len(cameras)}] "
                        f"{camera.market}:{camera.camera_id}: ERROR - {error}"
                    )
                    continue

                cameras_scanned += 1
                market_row["cameras_scanned"] += 1

                if result.waymo_count > 0:
                    total_waymo_count += result.waymo_count
                    cameras_with_waymos += 1
                    market_row["total_waymo_count"] += result.waymo_count
                    market_row["cameras_with_waymos"] += 1
                    batched_detections.append((result, image_url))

                    if camera.market == "austin":
                        if camera.is_in_service_area:
                            inside_area_waymo_count += result.waymo_count
                            inside_area_cameras_with_waymos += 1
                        else:
                            outside_area_waymo_count += result.waymo_count
                            outside_area_cameras_with_waymos += 1

                    image_status = " [img saved]" if image_url else ""
                    print(
                        f"  [{cameras_scanned + cameras_failed}/{len(cameras)}] "
                        f"{camera.market}:{camera.camera_id} [{camera.area_label}]: "
                        f"{result.waymo_count} Waymo(s) detected "
                        f"(avg conf: {result.avg_confidence:.2f}){image_status}"
                    )
                else:
                    print(
                        f"  [{cameras_scanned + cameras_failed}/{len(cameras)}] "
                        f"{camera.market}:{camera.camera_id} [{camera.area_label}]: No Waymos"
                    )

        if batched_detections:
            print(f"\nWriting {len(batched_detections)} detection records...")
            db.insert_detections(scan_id, batched_detections)

        duration = time.time() - start_time
        for market in market_stats.values():
            if market["duration_seconds"] == 0.0:
                market["duration_seconds"] = duration

        db.update_scan(
            scan_id=scan_id,
            cameras_scanned=cameras_scanned,
            cameras_failed=cameras_failed,
            total_waymo_count=total_waymo_count,
            cameras_with_waymos=cameras_with_waymos,
            duration_seconds=duration,
        )
        db.insert_market_stats(scan_id, list(market_stats.values()))

        print("\nUpdating camera metadata...")
        db.bulk_upsert_cameras(cameras)

    print("\n" + "=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)
    print(f"Scan ID: {scan_id}")
    print(f"Total cameras: {len(cameras)}")
    print(f"Cameras scanned: {cameras_scanned}")
    print(f"Cameras failed: {cameras_failed}")
    print(f"Total Waymos detected: {total_waymo_count}")
    print(f"Cameras with Waymos: {cameras_with_waymos}")
    if austin_cameras:
        print(f"Austin cameras inside current area: {austin_inside}")
        print(f"Austin cameras outside current area: {austin_outside}")
        print(
            f"Austin inside-area detections: {inside_area_waymo_count} across "
            f"{inside_area_cameras_with_waymos} cameras"
        )
        print(
            f"Austin outside-area detections: {outside_area_waymo_count} across "
            f"{outside_area_cameras_with_waymos} cameras"
        )
    print("Per-market summary:")
    for market in sorted(market_stats):
        row = market_stats[market]
        print(
            f"  {market}: {row['cameras_scanned']} scanned, "
            f"{row['cameras_failed']} failed, "
            f"{row['total_waymo_count']} Waymo(s) across "
            f"{row['cameras_with_waymos']} camera(s)"
        )
    print(f"Duration: {duration:.2f} seconds")
    print("=" * 60)


def main():
    try:
        run_scan()
    except Exception as exc:
        print(f"\nFATAL ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
