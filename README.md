# Waymo Counter

Automated Waymo vehicle detection from public traffic cameras across Austin, San Antonio, Dallas, Houston, Atlanta, Orlando, Miami, and Phoenix. The Render cron now runs every 30 minutes, scans all enabled markets in one job, and writes both whole-run and per-market stats to Supabase.

## Features

- Multi-market camera registry with shared source adapters
- Austin-only Waymo service-area tagging using the hardcoded polygon in [src/service_area.py](/Users/ethanmckanna/GitHub/waymo-counter/src/service_area.py)
- TxDOT district support for San Antonio, Dallas, and Houston
- Public 511 list support for Atlanta, Orlando, Phoenix, and Miami without API keys
- SunGuide-backed Miami support through the public Florida 511 camera feed
- YOLO inference on camera snapshots with annotated positive image upload
- Global scan stats plus per-market rollups in Supabase

## Project Structure

```text
waymo-counter/
├── render.yaml
├── requirements.txt
├── .env.example
├── README.md
├── src/
│   ├── main.py
│   ├── cameras.py
│   ├── config.py
│   ├── database.py
│   ├── detector.py
│   ├── image_annotator.py
│   ├── service_area.py
│   └── storage.py
└── tests/
    ├── fixtures/
    └── test_cameras.py
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SUPABASE_URL` | Yes | Supabase project URL |
| `SUPABASE_KEY` | Yes | Supabase service-role key |
| `MODEL_URL` | No | Model weights URL |
| `CONFIDENCE_THRESHOLD` | No | Minimum detection confidence. Default: `0.50` |
| `FETCH_WORKERS` | No | Concurrent image fetchers |
| `SCAN_SCOPE` | No | `all` or `service_area`. `service_area` only filters Austin |
| `ENABLED_MARKETS` | No | Comma-separated market slugs. Default: all 8 markets |

Supported `ENABLED_MARKETS` values:

```text
austin,san_antonio,dallas,houston,atlanta,orlando,miami,phoenix
```

## Data Contract

- Raw upstream camera identifier stays in `camera_id`
- Canonical identifier is `camera_key = "<market>:<source>:<camera_id>"`
- `source` is the adapter family identifier:
  - `austin_cctv`
  - `txdot_cctv`
  - `atis_511_cctv`
  - `sunguide_cctv`
- New markets are scanned market-wide until explicit service-area polygons are added
- Atlanta, Orlando, Miami, and Phoenix use the public `List/GetData/Cameras` feeds and local metro filters, so no extra API credentials are required

## Supabase Schema

Apply this migration if you are upgrading an existing Austin-only install.

```sql
-- Whole-run scans table
create table if not exists scans (
    id uuid default gen_random_uuid() primary key,
    timestamp timestamptz not null default now(),
    total_cameras integer not null,
    cameras_scanned integer not null,
    cameras_failed integer default 0,
    total_waymo_count integer not null default 0,
    cameras_with_waymos integer default 0,
    duration_seconds numeric(10, 2)
);
create index if not exists idx_scans_timestamp on scans(timestamp desc);

-- Canonical camera metadata
alter table cameras add column if not exists camera_key text;
alter table cameras add column if not exists market text;
alter table cameras add column if not exists source text;
alter table cameras add column if not exists image_url text;
alter table cameras add column if not exists is_in_service_area boolean default false;

update cameras
set
    market = coalesce(market, 'austin'),
    source = coalesce(source, 'austin_cctv'),
    camera_key = coalesce(camera_key, 'austin:austin_cctv:' || camera_id)
where camera_key is null or market is null or source is null;

alter table cameras alter column camera_key set not null;
alter table cameras alter column market set not null;
alter table cameras alter column source set not null;

create unique index if not exists idx_cameras_camera_key on cameras(camera_key);
create index if not exists idx_cameras_market on cameras(market);

-- Detection rows
alter table detections add column if not exists camera_key text;
alter table detections add column if not exists market text;
alter table detections add column if not exists source text;
alter table detections add column if not exists image_url text;

update detections
set
    market = coalesce(market, 'austin'),
    source = coalesce(source, 'austin_cctv'),
    camera_key = coalesce(camera_key, 'austin:austin_cctv:' || camera_id)
where camera_key is null or market is null or source is null;

alter table detections alter column camera_key set not null;
alter table detections alter column market set not null;
alter table detections alter column source set not null;

create index if not exists idx_detections_camera_key on detections(camera_key);
create index if not exists idx_detections_market on detections(market);
create index if not exists idx_detections_timestamp on detections(timestamp desc);

-- Per-market rollups
create table if not exists scan_market_stats (
    scan_id uuid not null references scans(id) on delete cascade,
    market text not null,
    total_cameras integer not null,
    cameras_scanned integer not null,
    cameras_failed integer not null default 0,
    total_waymo_count integer not null default 0,
    cameras_with_waymos integer not null default 0,
    duration_seconds numeric(10, 2),
    primary key (scan_id, market)
);
create index if not exists idx_scan_market_stats_market on scan_market_stats(market);
```

For a fresh install, create the `cameras` and `detections` tables with the new columns from the start:

```sql
create table if not exists cameras (
    camera_key text primary key,
    camera_id text not null,
    market text not null,
    source text not null,
    location_name text,
    longitude numeric(12, 9),
    latitude numeric(12, 9),
    council_district integer,
    image_url text,
    is_in_service_area boolean default false,
    last_scanned timestamptz,
    updated_at timestamptz default now()
);

create table if not exists detections (
    id uuid default gen_random_uuid() primary key,
    scan_id uuid references scans(id) on delete cascade,
    camera_key text not null,
    camera_id text not null,
    market text not null,
    source text not null,
    timestamp timestamptz not null,
    waymo_count integer not null,
    avg_confidence numeric(5, 4),
    detections_json jsonb,
    image_url text
);
```

## Storage Paths

Positive detections are now stored under:

```text
detections/{market}/{source}/{area_label}/{camera_storage_slug}/{YYYY-MM-DD}/{HHMMSS}.jpg
```

Austin still uses `inside_service_area` and `outside_service_area`. Other markets use `market_wide`.

## Local Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python3 scripts/download_model.py
python3 -m pytest
python3 -m src.main
```

## Deployment

- Render cron schedule: every 30 minutes
- Set `ENABLED_MARKETS` to the markets you want active
- Leave `SCAN_SCOPE=service_area` only if you want Austin filtered to the current polygon
