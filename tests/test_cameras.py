from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import httpx

from src.cameras import AustinCameraSource, CameraFetcher, MARKET_SPECS, TxDotDistrictSource, build_camera_key


FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text())


def test_build_camera_key():
    assert build_camera_key("austin", "austin_cctv", "123") == "austin:austin_cctv:123"


def test_market_registry_contains_all_markets():
    assert set(MARKET_SPECS) == {
        "austin",
        "san_antonio",
        "dallas",
        "houston",
        "atlanta",
        "orlando",
        "miami",
        "phoenix",
    }


def test_parse_austin_rows_and_service_area_filter():
    rows = load_fixture("austin_cameras.json")

    cameras = AustinCameraSource.parse_camera_rows(rows, filter_to_service_area=False)
    assert len(cameras) == 2
    assert cameras[0].camera_key == "austin:austin_cctv:1001"
    assert any(camera.is_in_service_area for camera in cameras)

    filtered = AustinCameraSource.parse_camera_rows(rows, filter_to_service_area=True)
    assert len(filtered) == 1
    assert filtered[0].camera_id == "1001"
    assert filtered[0].area_label == "inside_service_area"


def test_parse_txdot_payload_builds_snapshot_urls():
    payload = load_fixture("txdot_status.json")
    source = TxDotDistrictSource("DAL")

    cameras = source.parse_status_payload(payload, market="dallas")
    assert len(cameras) == 1
    camera = cameras[0]
    assert camera.camera_key == "dallas:txdot_cctv:IH20 @ Belt Line (Balch Springs)"
    assert camera.image_url is not None
    assert "districtCode=DAL" in camera.image_url
    assert "icdId=IH20%20%40%20Belt%20Line%20%28Balch%20Springs%29" in camera.image_url


def test_txdot_parser_tolerates_empty_or_malformed_payloads():
    source = TxDotDistrictSource("SAT")

    assert source.parse_status_payload(None, market="san_antonio") == []
    assert source.parse_status_payload({"roadwayCctvStatuses": None}, market="san_antonio") == []


def test_parse_public_511_payload_filters_to_market_bounds():
    atlanta_source = MARKET_SPECS["atlanta"].adapter
    atlanta_cameras = atlanta_source.parse_camera_rows(
        load_fixture("ga_511_cameras.json")["data"],
        market="atlanta",
    )
    assert len(atlanta_cameras) == 1
    assert atlanta_cameras[0].camera_id == "15035"
    assert atlanta_cameras[0].source == "atis_511_cctv"
    assert atlanta_cameras[0].location_name == "COBB-1030: Akers Mill Rd at Overton Park Dr"

    miami_source = MARKET_SPECS["miami"].adapter
    miami_cameras = miami_source.parse_camera_rows(
        load_fixture("fl511_cameras.json")["data"],
        market="miami",
    )
    assert len(miami_cameras) == 1
    assert miami_cameras[0].camera_id == "901"
    assert miami_cameras[0].source == "sunguide_cctv"
    assert miami_cameras[0].county == "Miami-Dade"

    phoenix_source = MARKET_SPECS["phoenix"].adapter
    phoenix_cameras = phoenix_source.parse_camera_rows(
        load_fixture("az511_cameras_phoenix.json")["data"],
        market="phoenix",
    )
    assert len(phoenix_cameras) == 1
    assert phoenix_cameras[0].camera_id == "1701"
    assert phoenix_cameras[0].city == "Phoenix"


def test_extract_image_bytes_decodes_txdot_snapshot_json():
    image_bytes = b"fake-image"
    response = httpx.Response(
        200,
        headers={"content-type": "application/json"},
        json={"snippet": base64.b64encode(image_bytes).decode("ascii")},
    )

    assert CameraFetcher._extract_image_bytes(response) == image_bytes


def test_mocked_source_fetches_cover_each_source_family():
    austin_rows = load_fixture("austin_cameras.json")
    txdot_payload = load_fixture("txdot_status.json")
    atlanta_payload = load_fixture("ga_511_cameras.json")

    def handler(request: httpx.Request) -> httpx.Response:
        if "austintexas.gov" in request.url.host:
            return httpx.Response(200, json=austin_rows)
        if "its.txdot.gov" in request.url.host:
            return httpx.Response(200, json=txdot_payload)
        if request.url.path.endswith("/List/GetData/Cameras"):
            return httpx.Response(200, json=atlanta_payload)
        raise AssertionError(f"Unexpected request: {request.url}")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    config = SimpleNamespace(
        scan_scope="all",
        enabled_markets=["austin", "dallas", "atlanta"],
    )
    fetcher = CameraFetcher(config=config, client=client)

    cameras = fetcher.fetch_active_cameras()
    assert any(camera.market == "austin" for camera in cameras)
    assert any(camera.market == "dallas" for camera in cameras)
    assert any(camera.market == "atlanta" for camera in cameras)
    assert {camera.source for camera in cameras} == {
        "austin_cctv",
        "txdot_cctv",
        "atis_511_cctv",
    }
