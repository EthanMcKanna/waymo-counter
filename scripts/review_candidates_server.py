#!/usr/bin/env python3
"""
Run a tiny local labeling UI for exported detection candidates.

This intentionally uses only the Python standard library so it works anywhere
the review manifest exists. Labels are appended to review_labels.jsonl and a
fully merged review_candidates.labeled.jsonl file is rewritten after each vote.
"""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


LABELS = {"waymo", "not_waymo", "ignore"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review detection candidates locally.")
    parser.add_argument("manifest", type=Path, help="review_candidates.jsonl path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class ReviewStore:
    def __init__(self, manifest: Path):
        self.manifest = manifest
        self.root = manifest.parent
        self.labels_path = self.root / "review_labels.jsonl"
        self.labeled_manifest_path = self.root / "review_candidates.labeled.jsonl"
        self.candidates = read_jsonl(manifest)
        self.labels = {
            row["candidate_id"]: row
            for row in read_jsonl(self.labels_path)
            if row.get("candidate_id")
        }
        self.apply_labels()

    def apply_labels(self) -> None:
        for candidate in self.candidates:
            label_row = self.labels.get(candidate.get("candidate_id"))
            if label_row:
                candidate["review_label"] = label_row.get("review_label")
                candidate["notes"] = label_row.get("notes", candidate.get("notes", ""))

    def counts(self) -> dict[str, int]:
        counts = {"unlabeled": 0, "waymo": 0, "not_waymo": 0, "ignore": 0}
        for candidate in self.candidates:
            label = candidate.get("review_label")
            if label in LABELS:
                counts[label] += 1
            else:
                counts["unlabeled"] += 1
        return counts

    def next_unlabeled_index(self, start: int = 0) -> int:
        total = len(self.candidates)
        for offset in range(total):
            index = (start + offset) % total
            if self.candidates[index].get("review_label") not in LABELS:
                return index
        return min(max(start, 0), max(total - 1, 0))

    def set_label(self, index: int, label: str, notes: str = "") -> None:
        if label not in LABELS:
            raise ValueError(f"Unsupported label: {label}")
        candidate = self.candidates[index]
        label_row = {
            "candidate_id": candidate["candidate_id"],
            "review_label": label,
            "notes": notes,
        }
        self.labels[candidate["candidate_id"]] = label_row
        candidate["review_label"] = label
        candidate["notes"] = notes
        with self.labels_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(label_row, sort_keys=True) + "\n")
        self.write_labeled_manifest()

    def write_labeled_manifest(self) -> None:
        with self.labeled_manifest_path.open("w", encoding="utf-8") as handle:
            for candidate in self.candidates:
                handle.write(json.dumps(candidate, sort_keys=True) + "\n")


class ReviewHandler(BaseHTTPRequestHandler):
    store: ReviewStore

    def log_message(self, format: str, *args) -> None:
        return

    def send_text(self, text: str, status: int = HTTPStatus.OK, content_type: str = "text/html") -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/image":
            self.serve_image(parsed)
            return
        if parsed.path == "/status":
            self.send_text(json.dumps(self.store.counts(), sort_keys=True), content_type="application/json")
            return
        self.serve_review(parsed)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/label":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = parse_qs(self.rfile.read(length).decode("utf-8"))
        index = int(payload.get("index", ["0"])[0])
        label = payload.get("label", [""])[0]
        notes = payload.get("notes", [""])[0]
        self.store.set_label(index, label, notes)
        next_index = self.store.next_unlabeled_index(index + 1)
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", f"/?index={next_index}")
        self.end_headers()

    def serve_image(self, parsed) -> None:
        query = parse_qs(parsed.query)
        index = int(query.get("index", ["0"])[0])
        candidate = self.store.candidates[index]
        path = Path(candidate.get("local_image") or "")
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_review(self, parsed) -> None:
        if not self.store.candidates:
            self.send_text("<h1>No candidates found</h1>")
            return
        query = parse_qs(parsed.query)
        index = int(query.get("index", [str(self.store.next_unlabeled_index())])[0])
        index = min(max(index, 0), len(self.store.candidates) - 1)
        candidate = self.store.candidates[index]
        counts = self.store.counts()
        previous_index = max(0, index - 1)
        next_index = min(len(self.store.candidates) - 1, index + 1)
        body = render_page(candidate, index, len(self.store.candidates), counts, previous_index, next_index)
        self.send_text(body)


def render_page(
    candidate: dict,
    index: int,
    total: int,
    counts: dict[str, int],
    previous_index: int,
    next_index: int,
) -> str:
    label = candidate.get("review_label") or "unlabeled"
    metadata = {
        "market": candidate.get("market"),
        "domain": candidate.get("domain"),
        "camera": candidate.get("camera_key"),
        "confidence": candidate.get("detection_confidence"),
        "timestamp": candidate.get("timestamp"),
        "bbox": candidate.get("bbox_xyxy"),
    }
    metadata_html = "".join(
        f"<dt>{html.escape(str(key))}</dt><dd>{html.escape(str(value))}</dd>"
        for key, value in metadata.items()
    )
    counts_text = " / ".join(f"{key}: {value}" for key, value in counts.items())
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Waymo Candidate Review</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; background: #f6f6f3; color: #1f2328; }}
    main {{ max-width: 1100px; margin: 0 auto; display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 24px; }}
    img {{ width: 100%; border: 1px solid #d0d7de; background: white; }}
    aside {{ background: white; border: 1px solid #d0d7de; padding: 16px; }}
    dl {{ display: grid; grid-template-columns: 96px 1fr; gap: 8px; font-size: 14px; }}
    dt {{ font-weight: 700; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    button, a.button {{ display: block; width: 100%; box-sizing: border-box; margin: 8px 0; padding: 12px; border: 1px solid #8c959f; background: #fff; color: #1f2328; text-align: center; text-decoration: none; font: inherit; cursor: pointer; }}
    button[value="waymo"] {{ background: #d1f7c4; }}
    button[value="not_waymo"] {{ background: #ffd8d3; }}
    button[value="ignore"] {{ background: #dde7f7; }}
    textarea {{ width: 100%; height: 70px; box-sizing: border-box; }}
    .status {{ font-size: 13px; color: #57606a; }}
    @media (max-width: 860px) {{ main {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main>
    <section>
      <p class="status">Candidate {index + 1} of {total} | current label: {html.escape(str(label))} | {html.escape(counts_text)}</p>
      <img src="/image?index={index}" alt="Detection candidate">
    </section>
    <aside>
      <form method="post" action="/label">
        <input type="hidden" name="index" value="{index}">
        <textarea name="notes" placeholder="optional notes">{html.escape(str(candidate.get("notes") or ""))}</textarea>
        <button name="label" value="waymo">Waymo (W)</button>
        <button name="label" value="not_waymo">Not Waymo (N)</button>
        <button name="label" value="ignore">Ignore (I)</button>
      </form>
      <a class="button" href="/?index={previous_index}">Previous</a>
      <a class="button" href="/?index={next_index}">Next</a>
      <dl>{metadata_html}</dl>
    </aside>
  </main>
  <script>
    document.addEventListener('keydown', (event) => {{
      const map = {{'w': 'waymo', 'n': 'not_waymo', 'i': 'ignore'}};
      const label = map[event.key.toLowerCase()];
      if (!label) return;
      const button = document.querySelector(`button[value="${{label}}"]`);
      if (button) button.click();
    }});
  </script>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    ReviewHandler.store = ReviewStore(args.manifest)
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Reviewing {len(ReviewHandler.store.candidates)} candidates at {url}")
    print(f"Labels: {ReviewHandler.store.labels_path}")
    print(f"Merged manifest: {ReviewHandler.store.labeled_manifest_path}")
    server.serve_forever()


if __name__ == "__main__":
    main()
