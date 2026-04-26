#!/usr/bin/env python3
"""
Build a market-balanced crop-classification dataset from reviewed candidates.

Input rows come from scripts/export_detection_review_set.py after a reviewer has
set review_label to one of: waymo, not_waymo, ignore.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VALID_LABELS = {"waymo", "not_waymo"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create train/val/test verifier crops from reviewed detections."
    )
    parser.add_argument("manifest", type=Path, help="Reviewed JSONL manifest.")
    parser.add_argument("--output", type=Path, default=Path("data/verifier_dataset"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--padding", type=float, default=0.35)
    parser.add_argument(
        "--copy-full-images",
        action="store_true",
        help="Also copy full annotated images into output/full_images.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            label = row.get("review_label")
            if label in VALID_LABELS:
                rows.append(row)
            elif label not in (None, "", "ignore"):
                raise ValueError(f"Invalid review_label on line {line_number}: {label}")
    return rows


def split_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("market") or "unknown"),
        str(row.get("domain") or "unknown"),
        str(row.get("review_label") or "unknown"),
    )


def assign_splits(
    rows: list[dict[str, Any]],
    seed: int,
    val_frac: float,
    test_frac: float,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[split_key(row)].append(row)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for bucket_rows in buckets.values():
        rng.shuffle(bucket_rows)
        total = len(bucket_rows)
        test_count = round(total * test_frac)
        val_count = round(total * val_frac)
        for index, row in enumerate(bucket_rows):
            if index < test_count:
                split = "test"
            elif index < test_count + val_count:
                split = "val"
            else:
                split = "train"
            splits[split].append(row)
    return splits


def crop_box(
    image: Any,
    bbox: list[float],
    padding: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    max_x = max(x1, x2, 1.0)
    max_y = max(y1, y2, 1.0)
    scale = min(1.0, image.width / max_x, image.height / max_y)
    x1, y1, x2, y2 = [value * scale for value in (x1, y1, x2, y2)]

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    pad_x = width * padding
    pad_y = height * padding
    return (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(image.width, int(x2 + pad_x)),
        min(image.height, int(y2 + pad_y)),
    )


def output_name(row: dict[str, Any]) -> str:
    raw = "|".join(
        str(row.get(key) or "")
        for key in ("candidate_id", "market", "camera_key", "timestamp")
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{row.get('market') or 'unknown'}_{row.get('review_label')}_{digest}.jpg"


def write_dataset(args: argparse.Namespace, splits: dict[str, list[dict[str, Any]]]) -> None:
    from PIL import Image

    args.output.mkdir(parents=True, exist_ok=True)
    written_rows: list[dict[str, Any]] = []
    counts = Counter()

    for split, rows in splits.items():
        for row in rows:
            image_path = Path(str(row.get("local_image") or ""))
            bbox = row.get("bbox_xyxy")
            if not image_path.exists() or not bbox:
                counts["skipped_missing_image_or_bbox"] += 1
                continue

            label = row["review_label"]
            out_dir = args.output / split / label
            out_path = out_dir / output_name(row)
            out_dir.mkdir(parents=True, exist_ok=True)

            with Image.open(image_path) as image:
                crop = image.crop(crop_box(image, bbox, args.padding)).convert("RGB")
                crop.save(out_path, format="JPEG", quality=92)

            if args.copy_full_images:
                full_dir = args.output / "full_images" / split / label
                full_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, full_dir / image_path.name)

            record = dict(row)
            record["split"] = split
            record["crop_path"] = str(out_path)
            written_rows.append(record)
            counts[f"{split}:{label}"] += 1
            counts[f"market:{row.get('market')}:{label}"] += 1

    manifest_path = args.output / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in written_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "input_manifest": str(args.manifest),
        "output": str(args.output),
        "counts": dict(counts),
        "classes": sorted(VALID_LABELS),
        "note": (
            "Crops are intended for a second-stage verifier. For production-grade "
            "training, prefer raw unannotated detection images once raw capture is enabled."
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.manifest)
    splits = assign_splits(rows, args.seed, args.val_frac, args.test_frac)
    write_dataset(args, splits)


if __name__ == "__main__":
    main()
