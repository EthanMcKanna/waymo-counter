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
    confidence_threshold: float
    fetch_workers: int
    scan_scope: str
    enabled_markets: list[str]


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
        "https://github.com/EthanMcKanna/waymo-counter/releases/download/v1.0/best.pt",
    )
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

    enabled_markets = _parse_enabled_markets(
        os.environ.get("ENABLED_MARKETS", ",".join(DEFAULT_ENABLED_MARKETS))
    )

    return Config(
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        model_url=model_url,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        fetch_workers=fetch_workers,
        scan_scope=scan_scope,
        enabled_markets=enabled_markets,
    )
