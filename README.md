# Waymo Counter

Automated Waymo vehicle detection from Austin CCTV cameras. Runs every 5 minutes on Render.com, uploads results to Supabase, and tags detections as inside or outside the current known service area.

## Features

- Fetches active cameras from Austin's public CCTV API
- Tags cameras relative to the known Waymo service area
- Runs YOLO detection on each camera image
- Stores results in Supabase for analysis

## Project Structure

```
waymo-counter/
├── render.yaml              # Render.com cron job config
├── requirements.txt         # Python dependencies
├── runtime.txt              # Python 3.11
├── .env.example
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   ├── main.py              # Entry point - orchestrates scan
│   ├── config.py            # Environment config
│   ├── cameras.py           # Camera fetching/filtering
│   ├── detector.py          # YOLO detection wrapper
│   ├── database.py          # Supabase client
│   └── service_area.py      # Polygon + point-in-polygon
└── models/
    └── .gitkeep             # Model downloaded at runtime
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_KEY` | Yes | Service role key (not anon) |
| `MODEL_URL` | No | URL to download model weights during build |
| `CONFIDENCE_THRESHOLD` | No | Min detection confidence (default: 0.50) |
| `FETCH_WORKERS` | No | Concurrent image fetchers (default: 8 x CPU count, min 8) |
| `SCAN_SCOPE` | No | `all` to scan every active camera, `service_area` for the old boundary-only mode |

## Supabase Schema

Run these SQL commands to set up the database:

```sql
-- Scans table
CREATE TABLE scans (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    total_cameras INTEGER NOT NULL,
    cameras_scanned INTEGER NOT NULL,
    cameras_failed INTEGER DEFAULT 0,
    total_waymo_count INTEGER NOT NULL DEFAULT 0,
    cameras_with_waymos INTEGER DEFAULT 0,
    duration_seconds NUMERIC(10, 2)
);
CREATE INDEX idx_scans_timestamp ON scans(timestamp DESC);

-- Detections table
CREATE TABLE detections (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    scan_id UUID REFERENCES scans(id) ON DELETE CASCADE,
    camera_id TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    waymo_count INTEGER NOT NULL,
    avg_confidence NUMERIC(5, 4),
    detections_json JSONB
);
CREATE INDEX idx_detections_camera_id ON detections(camera_id);
CREATE INDEX idx_detections_timestamp ON detections(timestamp DESC);

-- Cameras table
CREATE TABLE cameras (
    camera_id TEXT PRIMARY KEY,
    location_name TEXT,
    longitude NUMERIC(12, 9),
    latitude NUMERIC(12, 9),
    council_district INTEGER,
    last_scanned TIMESTAMPTZ,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Local Development

1. Clone the repository
2. Create a virtual environment: `python -m venv .venv`
3. Activate: `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and fill in values
6. Download weights: `python scripts/download_model.py`
7. Run: `python -m src.main`

## Deployment

1. Push to GitHub
2. Connect repo to Render.com
3. Render will auto-detect `render.yaml` blueprint
4. Set environment variables in Render dashboard
5. Upload model weights to GitHub Releases
6. Update `MODEL_URL` in render.yaml or Render dashboard
7. Ensure the build command downloads the model into `models/best.pt`

## Model Hosting

The YOLO model weights (~18MB) should be hosted on GitHub Releases:

1. Create a release on your repo (e.g., `v1.0`)
2. Upload `best.pt` as a release asset
3. Set `MODEL_URL` to the download URL

The model is downloaded during the Render build so each cron run can start
immediately without a runtime weight fetch.

## Expansion Monitoring

By default the scanner now processes all active Austin cameras and tags each one
as `inside_service_area` or `outside_service_area` based on the hardcoded Waymo
polygon in `src/service_area.py`.
Positive images are saved under matching storage prefixes so likely expansion
hits are easy to review separately.
