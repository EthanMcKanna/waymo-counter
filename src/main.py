"""
Waymo Counter - Main Orchestration Script

This is the entry point for the cron job. It:
1. Fetches active cameras and tags them relative to the Waymo service area
2. Downloads images and runs YOLO detection
3. Uploads results to Supabase
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from .cameras import Camera, CameraFetcher
from .config import load_config
from .database import Database
from .detector import DetectionResult, WaymoDetector
from .image_annotator import annotate_image, compress_image
from .storage import ImageStorage


def fetch_camera_image(
    camera: Camera,
    camera_fetcher: CameraFetcher,
) -> tuple[Camera, bytes | None, str | None]:
    """Fetch a single camera image."""
    image_bytes = camera_fetcher.fetch_image(camera.camera_id)
    if image_bytes is None:
        return (camera, None, "Failed to fetch image")

    return (camera, image_bytes, None)


def process_fetched_camera(
    camera: Camera,
    image_bytes: bytes,
    detector: WaymoDetector,
    image_storage: ImageStorage | None = None,
) -> tuple[Camera, DetectionResult | None, str | None, str | None]:
    """
    Process a single camera: fetch image, run detection, and optionally upload annotated image.

    Returns:
        Tuple of (camera, detection_result, error_message, image_url)
    """
    try:
        # Run detection
        result = detector.detect_from_bytes(image_bytes, camera.camera_id)

        # If detections found and image storage is available, upload annotated image
        image_url = None
        if result and result.waymo_count > 0 and image_storage:
            from io import BytesIO
            from PIL import Image

            timestamp = datetime.now(timezone.utc)
            # Load image from bytes only when we need to annotate
            image = Image.open(BytesIO(image_bytes))
            annotated = annotate_image(image, result.detections)
            compressed = compress_image(annotated)
            image_url = image_storage.upload_image(
                compressed,
                camera.camera_id,
                camera.area_label,
                timestamp,
            )
            # Explicitly close and delete image objects to free memory
            image.close()
            annotated.close()
            del image
            del annotated
            del compressed

        # Clear image_bytes to free memory
        del image_bytes

        return (camera, result, None, image_url)

    except Exception as e:
        return (camera, None, str(e), None)


def run_scan():
    """Run a complete scan of the configured camera scope."""
    start_time = time.time()

    print("=" * 60)
    print(f"Waymo Counter Scan - {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    # Load configuration
    print("\nLoading configuration...")
    config = load_config()

    # Initialize components
    print("Initializing components...")
    db = Database(config.supabase_url, config.supabase_key)
    image_storage = ImageStorage(db.client)
    detector = WaymoDetector(
        model_path=config.model_path,
        model_url=config.model_url,
        confidence_threshold=config.confidence_threshold,
    )

    # Pre-load the model
    print("Loading detection model...")
    detector.load_model()

    # Fetch cameras
    print("\nFetching active cameras...")
    with CameraFetcher() as camera_fetcher:
        filter_to_service_area = config.scan_scope == "service_area"
        cameras = camera_fetcher.fetch_active_cameras(
            filter_to_service_area=filter_to_service_area
        )
        inside_cameras = sum(1 for camera in cameras if camera.is_in_service_area)
        outside_cameras = len(cameras) - inside_cameras

        if filter_to_service_area:
            print(f"Found {len(cameras)} cameras in the current Waymo service area")
        else:
            print(
                f"Found {len(cameras)} active cameras "
                f"({inside_cameras} inside area, {outside_cameras} outside area)"
            )

        if not cameras:
            print("No cameras found. Exiting.")
            return

        # Create initial scan record
        scan_id = db.create_scan(
            total_cameras=len(cameras),
            cameras_scanned=0,
            cameras_failed=0,
            total_waymo_count=0,
            cameras_with_waymos=0,
        )
        print(f"Created scan record: {scan_id}")

        # Process cameras with thread pool
        print(f"\nFetching with {config.fetch_workers} workers and running inference inline...")
        cameras_scanned = 0
        cameras_failed = 0
        total_waymo_count = 0
        cameras_with_waymos = 0
        inside_area_waymo_count = 0
        outside_area_waymo_count = 0
        inside_area_cameras_with_waymos = 0
        outside_area_cameras_with_waymos = 0
        processed_cameras: list[Camera] = []
        batched_detections: list[tuple[DetectionResult, str | None]] = []

        with ThreadPoolExecutor(max_workers=config.fetch_workers) as executor:
            futures = {
                executor.submit(
                    fetch_camera_image,
                    camera,
                    camera_fetcher,
                ): camera
                for camera in cameras
            }

            for future in as_completed(futures):
                camera, image_bytes, fetch_error = future.result()

                if fetch_error or image_bytes is None:
                    cameras_failed += 1
                    print(f"  [{cameras_scanned + cameras_failed}/{len(cameras)}] "
                          f"Camera {camera.camera_id}: ERROR - {fetch_error}")
                else:
                    camera, result, error, image_url = process_fetched_camera(
                        camera,
                        image_bytes,
                        detector,
                        image_storage,
                    )
                    del image_bytes

                    if error:
                        cameras_failed += 1
                        print(f"  [{cameras_scanned + cameras_failed}/{len(cameras)}] "
                              f"Camera {camera.camera_id}: ERROR - {error}")
                    else:
                        cameras_scanned += 1
                        processed_cameras.append(camera)

                        if result and result.waymo_count > 0:
                            total_waymo_count += result.waymo_count
                            cameras_with_waymos += 1
                            batched_detections.append((result, image_url))
                            if camera.is_in_service_area:
                                inside_area_waymo_count += result.waymo_count
                                inside_area_cameras_with_waymos += 1
                            else:
                                outside_area_waymo_count += result.waymo_count
                                outside_area_cameras_with_waymos += 1
                            image_status = " [img saved]" if image_url else ""
                            print(f"  [{cameras_scanned + cameras_failed}/{len(cameras)}] "
                                  f"Camera {camera.camera_id} [{camera.area_label}]: "
                                  f"{result.waymo_count} Waymo(s) detected "
                                  f"(avg conf: {result.avg_confidence:.2f}){image_status}")
                        else:
                            print(f"  [{cameras_scanned + cameras_failed}/{len(cameras)}] "
                                  f"Camera {camera.camera_id} [{camera.area_label}]: No Waymos")

        if batched_detections:
            print(f"\nWriting {len(batched_detections)} detection records...")
            db.insert_detections(scan_id, batched_detections)

        # Calculate duration
        duration = time.time() - start_time

        # Update scan record with final results
        db.update_scan(
            scan_id=scan_id,
            cameras_scanned=cameras_scanned,
            cameras_failed=cameras_failed,
            total_waymo_count=total_waymo_count,
            cameras_with_waymos=cameras_with_waymos,
            duration_seconds=duration,
        )

        # Bulk upsert camera metadata
        print("\nUpdating camera metadata...")
        db.bulk_upsert_cameras(processed_cameras)

    # Print summary
    print("\n" + "=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)
    print(f"Scan ID: {scan_id}")
    print(f"Total cameras: {len(cameras)}")
    print(f"Cameras inside current area: {inside_cameras}")
    print(f"Cameras outside current area: {outside_cameras}")
    print(f"Cameras scanned: {cameras_scanned}")
    print(f"Cameras failed: {cameras_failed}")
    print(f"Total Waymos detected: {total_waymo_count}")
    print(f"Cameras with Waymos: {cameras_with_waymos}")
    print(
        f"Inside-area detections: {inside_area_waymo_count} across "
        f"{inside_area_cameras_with_waymos} cameras"
    )
    print(
        f"Outside-area detections: {outside_area_waymo_count} across "
        f"{outside_area_cameras_with_waymos} cameras"
    )
    print(f"Duration: {duration:.2f} seconds")
    print("=" * 60)


def main():
    """Entry point."""
    try:
        run_scan()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
