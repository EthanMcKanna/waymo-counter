#!/usr/bin/env python3
"""
Download model weights into the repo's models/ directory.

Render cron jobs do not expose a persistent disk, so the model needs to be part
of the build artifact instead of fetched on every run.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.request import urlopen


DEFAULT_MODEL_URL = (
    "https://github.com/EthanMcKanna/waymo-counter/releases/download/v1.1/best.pt"
)
DEFAULT_VERIFIER_MODEL_URL = (
    "https://github.com/EthanMcKanna/waymo-counter/releases/download/v1.2/verifier.torchscript.pt"
)


def parse_bool(raw_value: str) -> bool:
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def download_if_missing(url: str, path: Path, label: str) -> None:
    if path.exists() and path.stat().st_size > 0:
        print(f"{label} already present at {path}")
        return

    print(f"Downloading {label} from {url} to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())

    print(f"Saved {label} to {path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    model_path = repo_root / "models" / "best.pt"
    model_url = os.environ.get("MODEL_URL", DEFAULT_MODEL_URL)
    verifier_model_path = repo_root / "models" / "verifier.torchscript.pt"
    verifier_model_url = os.environ.get("VERIFIER_MODEL_URL", DEFAULT_VERIFIER_MODEL_URL)

    download_if_missing(model_url, model_path, "YOLO model")
    if parse_bool(os.environ.get("VERIFIER_ENABLED", "false")):
        download_if_missing(verifier_model_url, verifier_model_path, "verifier model")


if __name__ == "__main__":
    main()
