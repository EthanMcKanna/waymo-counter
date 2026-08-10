"""HMAC-authenticated ingestion and private media uploads."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any

import httpx


class CloudflareClient:
    producer_id = "waymo-counter"

    def __init__(self, base_url: str, secret: str):
        self.base_url = base_url.rstrip("/")
        self.secret = secret.encode("utf-8")
        self.http = httpx.Client(timeout=30.0)

    def _request(self, path: str, body: bytes, idempotency_key: str, content_type: str, method: str = "POST") -> dict[str, Any]:
        timestamp = str(int(time.time()))
        body_hash = hashlib.sha256(body).hexdigest()
        canonical = "\n".join((timestamp, self.producer_id, idempotency_key, body_hash)).encode("utf-8")
        signature = hmac.new(self.secret, canonical, hashlib.sha256).hexdigest()
        response = self.http.request(method, f"{self.base_url}{path}", content=body, headers={
            "content-type": content_type, "content-length": str(len(body)), "x-content-sha256": body_hash,
            "idempotency-key": idempotency_key, "x-robotaxi-producer": self.producer_id,
            "x-robotaxi-timestamp": timestamp, "x-robotaxi-signature": signature,
        })
        response.raise_for_status()
        return response.json()

    def send_records(self, dataset: str, records: list[dict[str, Any]]) -> None:
        for offset in range(0, len(records), 100):
            chunk = records[offset:offset + 100]
            occurred_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            fingerprint = hashlib.sha256(json.dumps(chunk, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:32]
            key = f"{self.producer_id}:{dataset}:{fingerprint}"
            payload = {"schemaVersion": 1, "type": "source.records", "idempotencyKey": key,
                       "occurredAt": occurred_at, "dataset": dataset, "records": chunk}
            self._request("/v1/ingest/events", json.dumps(payload, separators=(",", ":")).encode(), key, "application/json")

    def upload_media(self, image: bytes, storage_id: str) -> str:
        key = f"{self.producer_id}:media:{storage_id}"[:200]
        result = self._request("/v1/media", image, key, "image/jpeg", method="PUT")
        return f"{self.base_url}{result['url']}"
