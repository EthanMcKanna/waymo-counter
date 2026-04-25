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


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    model_path = repo_root / "models" / "best.pt"
    model_url = os.environ.get("MODEL_URL", DEFAULT_MODEL_URL)

    if model_path.exists() and model_path.stat().st_size > 0:
        print(f"Model already present at {model_path}")
        return

    print(f"Downloading model from {model_url} to {model_path}")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(model_url, timeout=120) as response:
        model_path.write_bytes(response.read())

    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
