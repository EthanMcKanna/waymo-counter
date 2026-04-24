"""
Multi-market traffic camera fetchers.

This module provides a small registry of market-specific camera sources while
normalizing every upstream payload into a shared Camera shape.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional
from urllib.parse import quote, urljoin

import httpx
from PIL import Image

from .service_area import point_in_polygon


def build_camera_key(market: str, source: str, camera_id: str) -> str:
    """Build the canonical camera identifier used for storage and upserts."""
    return f"{market}:{source}:{camera_id}"


def _parse_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_point_wkt(value: Any) -> tuple[Optional[float], Optional[float]]:
    if not value:
        return (None, None)

    match = re.search(
        r"POINT\s*\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\)",
        str(value),
    )
    if not match:
        return (None, None)

    return (_parse_float(match.group(1)), _parse_float(match.group(2)))


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class BoundingBox:
    """Simple market-level bounding box used for statewide 511 feeds."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def contains(self, lon: Optional[float], lat: Optional[float]) -> bool:
        if lon is None or lat is None:
            return False
        return (
            self.min_lon <= lon <= self.max_lon
            and self.min_lat <= lat <= self.max_lat
        )


# Statewide 511 APIs need a market filter. These broad metro bounds keep the
# scanner focused on the requested cities until explicit polygons are added.
MARKET_BOUNDS: dict[str, BoundingBox] = {
    "atlanta": BoundingBox(min_lon=-85.35, min_lat=33.25, max_lon=-83.45, max_lat=34.35),
    "orlando": BoundingBox(min_lon=-81.85, min_lat=28.15, max_lon=-80.85, max_lat=28.95),
    "miami": BoundingBox(min_lon=-80.95, min_lat=25.25, max_lon=-79.95, max_lat=26.45),
    "phoenix": BoundingBox(min_lon=-112.85, min_lat=33.00, max_lon=-111.35, max_lat=34.05),
}


@dataclass(frozen=True)
class MarketFilter:
    """Market-level filter for statewide camera feeds."""

    bounds: BoundingBox
    counties: frozenset[str] = frozenset()
    cities: frozenset[str] = frozenset()

    @staticmethod
    def _normalize(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.strip().lower().replace(".", "")

    def matches(
        self,
        lon: Optional[float],
        lat: Optional[float],
        county: Optional[str] = None,
        city: Optional[str] = None,
    ) -> bool:
        if not self.bounds.contains(lon, lat):
            return False

        normalized_county = self._normalize(county)
        if self.counties and normalized_county and normalized_county not in self.counties:
            return False

        normalized_city = self._normalize(city)
        if self.cities and normalized_city and normalized_city not in self.cities:
            return False

        return True


MARKET_FILTERS: dict[str, MarketFilter] = {
    "atlanta": MarketFilter(bounds=MARKET_BOUNDS["atlanta"]),
    "orlando": MarketFilter(
        bounds=MARKET_BOUNDS["orlando"],
        counties=frozenset({"orange", "seminole", "osceola", "lake", "volusia"}),
    ),
    "miami": MarketFilter(
        bounds=MARKET_BOUNDS["miami"],
        counties=frozenset({"miami-dade", "broward"}),
    ),
    "phoenix": MarketFilter(
        bounds=MARKET_BOUNDS["phoenix"],
        counties=frozenset({"maricopa", "pinal"}),
    ),
}


DEFAULT_ENABLED_MARKETS = (
    "austin",
    "san_antonio",
    "dallas",
    "houston",
    "atlanta",
    "orlando",
    "miami",
    "phoenix",
)


@dataclass
class Camera:
    """Normalized camera metadata shared across every source."""

    camera_key: str
    camera_id: str
    market: str
    source: str
    location_name: str
    longitude: Optional[float]
    latitude: Optional[float]
    council_district: Optional[int] = None
    roadway: Optional[str] = None
    direction: Optional[str] = None
    jurisdiction: Optional[str] = None
    county: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    image_url: Optional[str] = None
    is_in_service_area: bool = False

    @property
    def area_label(self) -> str:
        if self.market == "austin":
            return "inside_service_area" if self.is_in_service_area else "outside_service_area"
        return "market_wide"

    @property
    def storage_slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", self.camera_key).strip("_")


class CameraSource:
    """Base class for a market camera source."""

    source_name: str

    def fetch_cameras(
        self,
        client: httpx.Client,
        config: Any,
        market: str,
        filter_to_service_area: bool = False,
    ) -> list[Camera]:
        raise NotImplementedError


class AustinCameraSource(CameraSource):
    """Austin Socrata CCTV source."""

    source_name = "austin_cctv"
    api_base = "https://data.austintexas.gov/resource/b4k4-adkb.json"
    image_base = "https://cctv.austinmobility.io/image"

    @classmethod
    def parse_camera_rows(
        cls,
        rows: list[dict[str, Any]],
        filter_to_service_area: bool = False,
    ) -> list[Camera]:
        cameras: list[Camera] = []

        for row in rows:
            camera_id = str(row.get("camera_id", "")).strip()
            if not camera_id:
                continue

            location = row.get("location", {}) or {}
            coords = location.get("coordinates", []) or []
            lon = coords[0] if len(coords) >= 2 else None
            lat = coords[1] if len(coords) >= 2 else None
            is_in_service_area = (
                lon is not None and lat is not None and point_in_polygon(lon, lat)
            )

            if filter_to_service_area and not is_in_service_area:
                continue

            raw_district = row.get("council_district")
            council_district = None
            if raw_district is not None:
                district_str = str(raw_district).split(",")[0].strip()
                council_district = _parse_int(district_str)

            cameras.append(
                Camera(
                    camera_key=build_camera_key("austin", cls.source_name, camera_id),
                    camera_id=camera_id,
                    market="austin",
                    source=cls.source_name,
                    location_name=str(row.get("location_name", "")).strip(),
                    longitude=_parse_float(lon),
                    latitude=_parse_float(lat),
                    council_district=council_district,
                    jurisdiction="Austin",
                    image_url=f"{cls.image_base}/{quote(camera_id, safe='')}.jpg",
                    is_in_service_area=is_in_service_area,
                )
            )

        return cameras

    def fetch_cameras(
        self,
        client: httpx.Client,
        config: Any,
        market: str,
        filter_to_service_area: bool = False,
    ) -> list[Camera]:
        params = {
            "$limit": 5000,
            "$where": "camera_status='TURNED_ON'",
        }
        response = client.get(self.api_base, params=params)
        response.raise_for_status()
        return self.parse_camera_rows(
            response.json(),
            filter_to_service_area=filter_to_service_area,
        )


class TxDotDistrictSource(CameraSource):
    """TxDOT district CCTV source."""

    source_name = "txdot_cctv"
    status_endpoint = "https://its.txdot.gov/its/DistrictIts/GetCctvStatusListByDistrict"
    snapshot_endpoint = "https://its.txdot.gov/its/DistrictIts/GetCctvSnapshotByIcdId"

    def __init__(self, district_code: str):
        self.district_code = district_code

    def _snapshot_url(self, camera_id: str) -> str:
        return (
            f"{self.snapshot_endpoint}?districtCode={quote(self.district_code)}"
            f"&icdId={quote(camera_id, safe='')}"
        )

    def parse_status_payload(
        self,
        payload: dict[str, Any],
        market: str,
    ) -> list[Camera]:
        if not isinstance(payload, dict):
            return []

        deduped: dict[str, Camera] = {}
        roadway_statuses = payload.get("roadwayCctvStatuses", {}) or {}
        if not isinstance(roadway_statuses, dict):
            return []

        for entries in roadway_statuses.values():
            if not isinstance(entries, list):
                continue

            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if not entry.get("hasSnapshot"):
                    continue

                camera_id = str(entry.get("icd_Id") or entry.get("name") or "").strip()
                if not camera_id or camera_id in deduped:
                    continue

                equip_loc = entry.get("equipLoc", {}) or {}
                if not isinstance(equip_loc, dict):
                    equip_loc = {}
                lon = _parse_float(entry.get("longitude") or entry.get("lonString"))
                lat = _parse_float(entry.get("latitude") or entry.get("latString"))

                deduped[camera_id] = Camera(
                    camera_key=build_camera_key(market, self.source_name, camera_id),
                    camera_id=camera_id,
                    market=market,
                    source=self.source_name,
                    location_name=str(entry.get("name") or camera_id).strip(),
                    longitude=lon,
                    latitude=lat,
                    roadway=str(equip_loc.get("roadway") or "").strip() or None,
                    direction=str(entry.get("dirDescription") or "").strip() or None,
                    jurisdiction=f"TxDOT {self.district_code}",
                    region=self.district_code,
                    image_url=self._snapshot_url(camera_id),
                    is_in_service_area=False,
                )

        return list(deduped.values())

    def fetch_cameras(
        self,
        client: httpx.Client,
        config: Any,
        market: str,
        filter_to_service_area: bool = False,
    ) -> list[Camera]:
        response = client.get(
            self.status_endpoint,
            params={"districtCode": self.district_code},
        )
        response.raise_for_status()
        return self.parse_status_payload(response.json(), market=market)


class Public511CameraSource(CameraSource):
    """Public DataTables-backed 511 camera source."""

    def __init__(
        self,
        source_name: str,
        origin: str,
        query_columns: list[dict[str, Any]],
        query_order: list[dict[str, Any]],
        market_filter: MarketFilter,
        page_size: int = 100,
        search_terms: tuple[str, ...] = (),
    ):
        self.source_name = source_name
        self.origin = origin.rstrip("/")
        self.query_columns = query_columns
        self.query_order = query_order
        self.market_filter = market_filter
        self.page_size = page_size
        self.search_terms = search_terms
        self.list_endpoint = f"{self.origin}/List/GetData/Cameras"

    def _build_query(self, start: int, search_value: str = "") -> dict[str, Any]:
        return {
            "columns": self.query_columns,
            "order": self.query_order,
            "start": start,
            "length": self.page_size,
            "search": {"value": search_value},
        }

    def _extract_image_url(self, images: list[dict[str, Any]]) -> Optional[str]:
        for image in images:
            if not isinstance(image, dict):
                continue
            if image.get("disabled") or image.get("blocked"):
                continue
            image_url = _clean_text(image.get("imageUrl"))
            if image_url:
                return urljoin(self.origin, image_url)
        return None

    def parse_camera_rows(self, rows: list[dict[str, Any]], market: str) -> list[Camera]:
        cameras: list[Camera] = []

        for row in rows:
            if not isinstance(row, dict):
                continue

            lat_lng = row.get("latLng", {}) or {}
            if not isinstance(lat_lng, dict):
                lat_lng = {}
            geography = lat_lng.get("geography", {}) or {}
            if not isinstance(geography, dict):
                geography = {}
            lon, lat = _parse_point_wkt(geography.get("wellKnownText"))

            county = _clean_text(row.get("county"))
            city = _clean_text(row.get("city"))
            if not self.market_filter.matches(lon, lat, county=county, city=city):
                continue

            images = row.get("images") or []
            if not isinstance(images, list):
                images = []
            image_url = self._extract_image_url(images)
            if not image_url:
                continue

            camera_id = str(
                row.get("id")
                or row.get("DT_RowId")
                or row.get("sourceId")
                or row.get("location")
                or ""
            ).strip()
            if not camera_id:
                continue

            location_name = str(
                row.get("location")
                or row.get("roadway")
                or camera_id
            ).strip()

            cameras.append(
                Camera(
                    camera_key=build_camera_key(market, self.source_name, camera_id),
                    camera_id=camera_id,
                    market=market,
                    source=self.source_name,
                    location_name=location_name,
                    longitude=lon,
                    latitude=lat,
                    roadway=_clean_text(row.get("roadway")),
                    direction=_clean_text(row.get("direction")),
                    jurisdiction=_clean_text(row.get("dotDistrict")) or _clean_text(row.get("source")),
                    county=county,
                    city=city,
                    region=_clean_text(row.get("region")),
                    image_url=image_url,
                    is_in_service_area=False,
                )
            )

        return cameras

    def fetch_cameras(
        self,
        client: httpx.Client,
        config: Any,
        market: str,
        filter_to_service_area: bool = False,
    ) -> list[Camera]:
        cameras_by_id: dict[str, Camera] = {}
        search_terms = self.search_terms or ("",)

        for search_term in search_terms:
            start = 0
            records_filtered: Optional[int] = None

            while records_filtered is None or start < records_filtered:
                try:
                    response = client.get(
                        self.list_endpoint,
                        params={
                            "query": json.dumps(
                                self._build_query(start, search_value=search_term),
                                separators=(",", ":"),
                            ),
                            "lang": "en",
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
                except httpx.HTTPError as exc:
                    label = search_term or "all cameras"
                    print(f"Stopping {market} 511 fetch for '{label}' at offset {start}: {exc}")
                    break

                if not isinstance(payload, dict):
                    label = search_term or "all cameras"
                    print(
                        f"Stopping {market} 511 fetch for '{label}' at offset {start}: "
                        "unexpected payload"
                    )
                    break

                rows = payload.get("data", []) or []
                if not isinstance(rows, list):
                    rows = []
                records_filtered = _parse_int(payload.get("recordsFiltered")) or len(rows)

                for camera in self.parse_camera_rows(rows, market=market):
                    cameras_by_id.setdefault(camera.camera_id, camera)

                if not rows:
                    break
                start += self.page_size

        return list(cameras_by_id.values())


@dataclass(frozen=True)
class MarketSpec:
    market: str
    label: str
    adapter: CameraSource


MARKET_SPECS: dict[str, MarketSpec] = {
    "austin": MarketSpec("austin", "Austin", AustinCameraSource()),
    "san_antonio": MarketSpec("san_antonio", "San Antonio", TxDotDistrictSource("SAT")),
    "dallas": MarketSpec("dallas", "Dallas", TxDotDistrictSource("DAL")),
    "houston": MarketSpec("houston", "Houston", TxDotDistrictSource("HOU")),
    "atlanta": MarketSpec(
        "atlanta",
        "Atlanta",
        Public511CameraSource(
            source_name="atis_511_cctv",
            origin="https://511ga.org",
            query_columns=[
                {"data": None, "name": ""},
                {"name": "sortOrder", "s": True},
                {"name": "roadway", "s": True},
                {"data": 3, "name": ""},
            ],
            query_order=[
                {"column": 1, "dir": "asc"},
                {"column": 2, "dir": "asc"},
            ],
            market_filter=MARKET_FILTERS["atlanta"],
            search_terms=("Fulton", "Cobb", "Gwinnett", "DeKalb"),
        ),
    ),
    "orlando": MarketSpec(
        "orlando",
        "Orlando",
        Public511CameraSource(
            source_name="atis_511_cctv",
            origin="https://fl511.com",
            query_columns=[
                {"data": None, "name": ""},
                {"name": "sortOrder", "s": True},
                {"name": "region", "s": True},
                {"name": "county", "s": True},
                {"name": "roadway", "s": True},
                {"name": "location"},
                {"name": "direction", "s": True},
                {"data": 7, "name": ""},
            ],
            query_order=[
                {"column": 1, "dir": "asc"},
                {"column": 2, "dir": "asc"},
            ],
            market_filter=MARKET_FILTERS["orlando"],
            search_terms=("Orange", "Seminole", "Osceola", "Lake", "Volusia"),
        ),
    ),
    "miami": MarketSpec(
        "miami",
        "Miami",
        Public511CameraSource(
            source_name="sunguide_cctv",
            origin="https://fl511.com",
            query_columns=[
                {"data": None, "name": ""},
                {"name": "sortOrder", "s": True},
                {"name": "region", "s": True},
                {"name": "county", "s": True},
                {"name": "roadway", "s": True},
                {"name": "location"},
                {"name": "direction", "s": True},
                {"data": 7, "name": ""},
            ],
            query_order=[
                {"column": 1, "dir": "asc"},
                {"column": 2, "dir": "asc"},
            ],
            market_filter=MARKET_FILTERS["miami"],
            search_terms=("Miami", "Broward"),
        ),
    ),
    "phoenix": MarketSpec(
        "phoenix",
        "Phoenix",
        Public511CameraSource(
            source_name="atis_511_cctv",
            origin="https://www.az511.gov",
            query_columns=[
                {"data": None, "name": ""},
                {"name": "sortOrder", "s": True},
                {"name": "city", "s": True},
                {"name": "roadway", "s": True},
                {"name": "location"},
                {"data": 5, "name": ""},
            ],
            query_order=[
                {"column": 1, "dir": "asc"},
                {"column": 3, "dir": "asc"},
            ],
            market_filter=MARKET_FILTERS["phoenix"],
        ),
    ),
}


class CameraFetcher:
    """Fetches and filters cameras across all configured markets."""

    def __init__(self, config: Any, timeout: float = 30.0, client: httpx.Client | None = None):
        self.config = config
        self.client = client or httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(max_connections=20),
            follow_redirects=True,
        )

    def fetch_active_cameras(self) -> list[Camera]:
        cameras: list[Camera] = []

        for market in self.config.enabled_markets:
            spec = MARKET_SPECS[market]
            filter_to_service_area = (
                market == "austin" and self.config.scan_scope == "service_area"
            )
            try:
                market_cameras = spec.adapter.fetch_cameras(
                    self.client,
                    self.config,
                    market=market,
                    filter_to_service_area=filter_to_service_area,
                )
            except httpx.HTTPError as exc:
                print(f"Skipping {spec.label}: failed to fetch cameras ({exc})")
                continue

            cameras.extend(market_cameras)

        return cameras

    @staticmethod
    def _extract_image_bytes(response: httpx.Response) -> Optional[bytes]:
        content_type = response.headers.get("content-type", "").lower()

        if "json" in content_type or response.content[:1] == b"{":
            try:
                payload = response.json()
            except json.JSONDecodeError:
                return None

            if not isinstance(payload, dict):
                return None

            snippet = payload.get("snippet")
            if snippet:
                try:
                    return base64.b64decode(snippet)
                except (ValueError, TypeError):
                    return None
            return None

        return response.content

    def fetch_image(self, camera: Camera) -> Optional[bytes]:
        if not camera.image_url:
            return None

        try:
            response = self.client.get(camera.image_url)
            response.raise_for_status()
            return self._extract_image_bytes(response)
        except Exception:
            return None

    def fetch_image_as_pil(self, camera: Camera) -> Optional[Image.Image]:
        image_bytes = self.fetch_image(camera)
        if image_bytes is None:
            return None

        try:
            return Image.open(BytesIO(image_bytes))
        except Exception:
            return None

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
