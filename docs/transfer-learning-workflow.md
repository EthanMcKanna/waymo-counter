# Transfer Learning Workflow

The detector should be promoted only when it holds Austin recall while reducing
false positives on non-Austin and highway-style cameras.

## 1. Export Review Candidates

Use recent production detections as the hard-negative source:

```bash
python3 scripts/export_detection_review_set.py \
  --since-hours 24 \
  --output data/review_sets/$(date -u +%Y%m%dT%H%M%SZ)
```

The exporter writes:

- `review_candidates.jsonl`: one row per predicted box
- `detection_rows.jsonl`: raw detection rows from Supabase
- `images/`: downloaded annotated detection images
- `summary.json`: market and export counts

For quick labeling, launch the local reviewer:

```bash
python3 scripts/review_candidates_server.py \
  data/review_sets/<run>/review_candidates.jsonl
```

Open `http://127.0.0.1:8765` and label each candidate. Keyboard shortcuts:
`W` for Waymo, `N` for not Waymo, and `I` for ignore. The reviewer writes
`review_labels.jsonl` and a merged `review_candidates.labeled.jsonl`.

Set each candidate's `review_label` to:

- `waymo`: true Waymo
- `not_waymo`: false positive / hard negative
- `ignore`: unusable or ambiguous

## 2. Build A Verifier Dataset

After review:

```bash
python3 scripts/build_verifier_dataset.py \
  data/review_sets/<run>/review_candidates.labeled.jsonl \
  --output data/verifier_datasets/<run>
```

This creates market-stratified `train`, `val`, and `test` crop folders for a
second-stage classifier:

```text
train/waymo
train/not_waymo
val/waymo
val/not_waymo
test/waymo
test/not_waymo
```

## 3. Promotion Gates

A new model stack must pass these gates before deployment:

- Austin inside-area recall does not regress against the locked eval set.
- Non-Austin false positives fall versus the current production model.
- Highway-domain precision is reported separately from urban cameras.
- Per-market metrics are shown for every active market.
- Future markets enter production with at least a negative-only review batch.

## 4. Next Production Capture Improvement

The current production image URL is an annotated positive image. That is enough
for human review, but verifier training should eventually use raw unannotated
crops. The next live-system change should store a raw detection image or raw crop
alongside the annotated image so the verifier cannot learn annotation artifacts.
