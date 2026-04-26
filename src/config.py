"""
Configuration Management

Loads environment variables and provides configuration for the application.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .cameras import DEFAULT_ENABLED_MARKETS, MARKET_SPECS


@dataclass
class Config:
    """Application configuration."""

    supabase_url: str
    supabase_key: str
    model_url: str
    model_path: Path
    model_image_size: int
    confidence_threshold: float
    verifier_enabled: bool
    verifier_model_url: str
    verifier_model_path: Path
    verifier_image_size: int
    verifier_crop_padding: float
    verifier_threshold: float
    verifier_non_austin_threshold: float
    verifier_calibration_enabled: bool
    fetch_workers: int
    scan_scope: str
    enabled_markets: list[str]
    scan_lock_minutes: int


def _parse_bool(raw_value: str) -> bool:
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_enabled_markets(raw_value: str) -> list[str]:
    if not raw_value.strip():
        raise ValueError("ENABLED_MARKETS cannot be empty")

    markets: list[str] = []
    seen: set[str] = set()

    for part in raw_value.split(","):
        market = part.strip().lower()
        if not market:
            continue
        if market not in MARKET_SPECS:
            supported = ", ".join(DEFAULT_ENABLED_MARKETS)
            raise ValueError(
                f"Unsupported market '{market}'. Supported markets: {supported}"
            )
        if market not in seen:
            seen.add(market)
            markets.append(market)

    if not markets:
        raise ValueError("ENABLED_MARKETS did not contain any valid markets")

    return markets


def load_config() -> Config:
    """Load configuration from environment variables."""

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables are required")

    model_url = os.environ.get(
        "MODEL_URL",
        "https://github.com/EthanMcKanna/waymo-counter/releases/download/v1.1/best.pt",
    )
    model_path = Path(__file__).parent.parent / "models" / "best.pt"
    model_image_size = int(os.environ.get("MODEL_IMAGE_SIZE", "640"))
    confidence_threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.70"))
    verifier_enabled = _parse_bool(os.environ.get("VERIFIER_ENABLED", "false"))
    verifier_model_url = os.environ.get(
        "VERIFIER_MODEL_URL",
        "https://github.com/EthanMcKanna/waymo-counter/releases/download/v1.2/verifier.torchscript.pt",
    )
    verifier_model_path = Path(__file__).parent.parent / "models" / "verifier.torchscript.pt"
    verifier_image_size = int(os.environ.get("VERIFIER_IMAGE_SIZE", "224"))
    verifier_crop_padding = float(os.environ.get("VERIFIER_CROP_PADDING", "0.35"))
    verifier_calibration_enabled = _parse_bool(
        os.environ.get("VERIFIER_CALIBRATION_ENABLED", "false")
    )
    default_austin_threshold = "0.43" if verifier_calibration_enabled else "0.475"
    default_non_austin_threshold = "0.50" if verifier_calibration_enabled else "0.90"
    verifier_threshold = float(os.environ.get("VERIFIER_THRESHOLD", default_austin_threshold))
    verifier_non_austin_threshold = float(
        os.environ.get("VERIFIER_NON_AUSTIN_THRESHOLD", default_non_austin_threshold)
    )

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

    enabled_markets = _parse_enabled_markets(
        os.environ.get("ENABLED_MARKETS", ",".join(DEFAULT_ENABLED_MARKETS))
    )
    scan_lock_minutes = int(os.environ.get("SCAN_LOCK_MINUTES", "120"))

    return Config(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        model_url=model_url,
        model_path=model_path,
        model_image_size=model_image_size,
        confidence_threshold=confidence_threshold,
        verifier_enabled=verifier_enabled,
        verifier_model_url=verifier_model_url,
        verifier_model_path=verifier_model_path,
        verifier_image_size=verifier_image_size,
        verifier_crop_padding=verifier_crop_padding,
        verifier_threshold=verifier_threshold,
        verifier_non_austin_threshold=verifier_non_austin_threshold,
        verifier_calibration_enabled=verifier_calibration_enabled,
        fetch_workers=fetch_workers,
        scan_scope=scan_scope,
        enabled_markets=enabled_markets,
        scan_lock_minutes=scan_lock_minutes,
    )
