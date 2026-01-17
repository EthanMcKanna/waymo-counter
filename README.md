# Waymo Counter

Automated Waymo vehicle detection from Austin CCTV cameras. Runs every 10 minutes on Render.com and uploads results to Supabase.

## Features

- Fetches active cameras from Austin's public CCTV API
- Filters to Waymo service area using point-in-polygon
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
| `MODEL_URL` | No | URL to download model weights |
| `CONFIDENCE_THRESHOLD` | No | Min detection confidence (default: 0.25) |
| `MAX_WORKERS` | No | Concurrent threads (default: 3) |

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
6. Run: `python -m src.main`

## Deployment

1. Push to GitHub
2. Connect repo to Render.com
3. Render will auto-detect `render.yaml` blueprint
4. Set environment variables in Render dashboard
5. Upload model weights to GitHub Releases
6. Update `MODEL_URL` in render.yaml or Render dashboard

## Model Hosting

The YOLO model weights (~18MB) should be hosted on GitHub Releases:

1. Create a release on your repo (e.g., `v1.0`)
2. Upload `best.pt` as a release asset
3. Set `MODEL_URL` to the download URL

The service downloads the model on first run and caches it locally.
