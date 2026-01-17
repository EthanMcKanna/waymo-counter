"""
Waymo Counter - Main Orchestration Script

This is the entry point for the cron job. It:
1. Fetches active cameras within the Waymo service area
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


def process_camera(
    camera: Camera,
    camera_fetcher: CameraFetcher,
    detector: WaymoDetector,
) -> tuple[Camera, DetectionResult | None, str | None]:
    """
    Process a single camera: fetch image and run detection.

    Returns:
        Tuple of (camera, detection_result, error_message)
    """
    try:
        # Fetch image
        image_bytes = camera_fetcher.fetch_image(camera.camera_id)
        if image_bytes is None:
            return (camera, None, "Failed to fetch image")

        # Run detection
        result = detector.detect_from_bytes(image_bytes, camera.camera_id)
        return (camera, result, None)

    except Exception as e:
        return (camera, None, str(e))


def run_scan():
    """Run a complete scan of all cameras in the Waymo service area."""
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
        cameras = camera_fetcher.fetch_active_cameras(filter_to_service_area=True)
        print(f"Found {len(cameras)} cameras in Waymo service area")

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
        print(f"\nProcessing cameras with {config.max_workers} workers...")
        cameras_scanned = 0
        cameras_failed = 0
        total_waymo_count = 0
        cameras_with_waymos = 0
        processed_cameras: list[Camera] = []

        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    process_camera,
                    camera,
                    camera_fetcher,
                    detector,
                ): camera
                for camera in cameras
            }

            # Process results as they complete
            for future in as_completed(futures):
                camera, result, error = future.result()

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
                        print(f"  [{cameras_scanned + cameras_failed}/{len(cameras)}] "
                              f"Camera {camera.camera_id}: {result.waymo_count} Waymo(s) detected "
                              f"(avg conf: {result.avg_confidence:.2f})")

                        # Insert detection record
                        db.insert_detection(scan_id, result)
                    else:
                        print(f"  [{cameras_scanned + cameras_failed}/{len(cameras)}] "
                              f"Camera {camera.camera_id}: No Waymos")

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
    print(f"Cameras scanned: {cameras_scanned}")
    print(f"Cameras failed: {cameras_failed}")
    print(f"Total Waymos detected: {total_waymo_count}")
    print(f"Cameras with Waymos: {cameras_with_waymos}")
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
