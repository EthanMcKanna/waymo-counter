"""
Configuration Management

Loads environment variables and provides configuration for the application.
"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Application configuration."""

    # Supabase
    supabase_url: str
    supabase_key: str

    # Model
    model_url: str
    model_path: Path

    # Detection
    confidence_threshold: float
    fetch_workers: int
    scan_scope: str

    # Austin CCTV API
    cctv_api_base: str = "https://data.austintexas.gov/resource/b4k4-adkb.json"
    cctv_image_base: str = "https://cctv.austinmobility.io/image"


def load_config() -> Config:
    """Load configuration from environment variables."""

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables are required")

    model_url = os.environ.get(
        "MODEL_URL",
        "https://github.com/EthanMcKanna/waymo-counter/releases/download/v1.0/best.pt"
    )

    # Model stored in models/ directory
    model_path = Path(__file__).parent.parent / "models" / "best.pt"

    confidence_threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.50"))

    detected_cpu_count = max(
        1,
        int(float(os.environ.get("RENDER_CPU_COUNT", os.cpu_count() or 1))),
    )
    default_fetch_workers = min(32, max(8, detected_cpu_count * 8))
    fetch_workers = int(
        os.environ.get(
            "FETCH_WORKERS",
            os.environ.get("MAX_WORKERS", str(default_fetch_workers)),
        )
    )
    scan_scope = os.environ.get("SCAN_SCOPE", "all").strip().lower()
    if scan_scope not in {"all", "service_area"}:
        raise ValueError("SCAN_SCOPE must be either 'all' or 'service_area'")

    return Config(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        model_url=model_url,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        fetch_workers=fetch_workers,
        scan_scope=scan_scope,
    )
