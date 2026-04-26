#!/usr/bin/env python3
"""
Export recent production detections into a human-review pack.

The output is intentionally simple: annotated images plus JSONL manifests that
can be reviewed by hand or imported into a labeling tool. Each detection box is
emitted as one candidate so false positives can become hard negatives for the
second-stage verifier.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        return False


HIGHWAY_PATTERN = re.compile(
    r"\b("
    r"IH|I-|US-|SH|SR|FM|Loop|Fwy|Freeway|Expwy|Expressway|"
    r"Highway|Hwy|Turnpike|Parkway|Pkwy|Beltway|Toll"
    r")\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export recent detections for cross-market review."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/review_sets/latest"),
        help="Directory for images, manifests, and summary files.",
    )
    parser.add_argument(
        "--since-hours",
        type=float,
        default=24.0,
        help="Only include detections newer than this many hours.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum detection rows to export.",
    )
    parser.add_argument(
        "--markets",
        default="",
        help="Comma-separated market filter. Empty means all markets.",
    )
    parser.add_argument(
        "--include-missing-images",
        action="store_true",
        help="Keep manifest rows even when the annotated image cannot be downloaded.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "item"


def market_filter(raw: str) -> list[str]:
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def infer_domain(row: dict[str, Any], camera: dict[str, Any] | None) -> str:
    haystack = " ".join(
        str(value or "")
        for value in (
            row.get("camera_id"),
            row.get("camera_key"),
            row.get("source"),
            camera.get("location_name") if camera else None,
            camera.get("image_url") if camera else None,
        )
    )
    if HIGHWAY_PATTERN.search(haystack):
        return "highway"
    if row.get("market") == "austin" and camera and camera.get("is_in_service_area"):
        return "austin_service_area"
    return "urban_or_unknown"


def fetch_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    try:
        from supabase import create_client
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The supabase package is required to export review sets. "
            "Install project requirements first."
        ) from exc

    load_dotenv()
    load_dotenv(".env.local")
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise SystemExit("SUPABASE_URL and SUPABASE_KEY are required")

    client = create_client(supabase_url, supabase_key)
    since = datetime.now(timezone.utc) - timedelta(hours=args.since_hours)
    query = (
        client.table("detections")
        .select(
            "id,scan_id,camera_key,camera_id,market,source,timestamp,"
            "waymo_count,avg_confidence,detections_json,image_url"
        )
        .gte("timestamp", since.isoformat())
        .order("timestamp", desc=True)
        .limit(args.limit)
    )

    markets = market_filter(args.markets)
    if markets:
        query = query.in_("market", markets)

    rows = query.execute().data or []
    camera_keys = sorted({row["camera_key"] for row in rows if row.get("camera_key")})
    cameras: dict[str, dict[str, Any]] = {}
    for start in range(0, len(camera_keys), 500):
        batch = camera_keys[start : start + 500]
        if not batch:
            continue
        camera_rows = (
            client.table("cameras")
            .select(
                "camera_key,location_name,longitude,latitude,image_url,"
                "is_in_service_area,market,source"
            )
            .in_("camera_key", batch)
            .execute()
            .data
            or []
        )
        cameras.update({row["camera_key"]: row for row in camera_rows})

    return rows, cameras


def download_image(url: str, path: Path) -> bool:
    try:
        with urlopen(url, timeout=30.0) as response:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as handle:
                handle.write(response.read())
        return True
    except Exception as exc:
        print(f"Failed to download {url}: {exc}")
        return False


def build_candidates(
    row: dict[str, Any],
    camera: dict[str, Any] | None,
    image_path: Path | None,
) -> list[dict[str, Any]]:
    detections = row.get("detections_json") or []
    if not isinstance(detections, list):
        detections = []

    candidates: list[dict[str, Any]] = []
    for index, detection in enumerate(detections):
        candidate_id = f"{row.get('id') or slugify(row['camera_key'])}-{index}"
        candidates.append(
            {
                "candidate_id": candidate_id,
                "review_label": None,
                "allowed_labels": ["waymo", "not_waymo", "ignore"],
                "market": row.get("market"),
                "domain": infer_domain(row, camera),
                "camera_key": row.get("camera_key"),
                "camera_id": row.get("camera_id"),
                "source": row.get("source"),
                "scan_id": row.get("scan_id"),
                "timestamp": row.get("timestamp"),
                "avg_confidence": row.get("avg_confidence"),
                "detection_confidence": detection.get("confidence"),
                "bbox_xyxy": detection.get("bbox"),
                "image_url": row.get("image_url"),
                "local_image": str(image_path) if image_path else None,
                "camera": camera or {},
                "notes": "",
            }
        )
    return candidates


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    images_dir = args.output / "images"

    rows, cameras = fetch_rows(args)
    manifest_path = args.output / "review_candidates.jsonl"
    rows_path = args.output / "detection_rows.jsonl"
    summary_path = args.output / "summary.json"

    counts = Counter()
    candidates: list[dict[str, Any]] = []
    with rows_path.open("w", encoding="utf-8") as rows_file:
        for row in rows:
            rows_file.write(json.dumps(row, sort_keys=True) + "\n")
            counts["detection_rows"] += 1

            image_path = None
            image_url = row.get("image_url")
            if image_url:
                timestamp = slugify(str(row.get("timestamp") or "unknown"))
                filename = f"{timestamp}_{slugify(row.get('camera_key') or row.get('camera_id') or 'camera')}.jpg"
                image_path = images_dir / str(row.get("market") or "unknown") / filename
                if image_path.exists() or download_image(image_url, image_path):
                    counts["images_downloaded"] += 1
                else:
                    counts["image_download_failures"] += 1
                    image_path = None

            if not image_path and not args.include_missing_images:
                counts["rows_skipped_missing_image"] += 1
                continue

            camera = cameras.get(row.get("camera_key"))
            row_candidates = build_candidates(row, camera, image_path)
            candidates.extend(row_candidates)
            counts["candidates"] += len(row_candidates)
            counts[f"market:{row.get('market')}"] += len(row_candidates)

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for candidate in candidates:
            manifest_file.write(json.dumps(candidate, sort_keys=True) + "\n")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output": str(args.output),
        "since_hours": args.since_hours,
        "limit": args.limit,
        "markets": market_filter(args.markets) or "all",
        "counts": dict(counts),
        "next_step": (
            "Fill review_label for each row in review_candidates.jsonl, then run "
            "scripts/build_verifier_dataset.py."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
