# VANTRA at city scale — from demo to deployment

**Today:** 18 cameras, 23k simulated detections, in-process pipeline on one laptop.
**Target:** a metro of 2,000+ cameras at ~50 detections/camera/day ≈ 100k detections
per day, every one needing inference, matching, and audit within seconds.

## 1. Ingestion — Kafka in front of everything

Each camera's video stream goes to a regional edge node running frame sampling and
plate localization (YOLOv8). Edge nodes publish *detections* (not raw video) to
Kafka topics — `detections.raw` — keyed by `camera_id`. This decouples camera
speed from processing speed: a traffic surge queues instead of dropping frames.
Object storage (S3) receives vehicle/plate crops alongside, referenced by ID.

## 2. Distributed inference workers

A consumer group of GPU workers subscribes to `detections.raw`: OCR (PaddleOCR) +
Re-ID embedding (VeRi-776-trained CNN) run per detection, enriched results
published to `detections.enriched`, keyed by **plate** — so all sightings of a
plate land on the same partition, and the trajectory matcher (a stateful Kafka
Streams / Flink job) sees them in order with no cross-node locks. The tiered
matching logic (exact → fuzzy → appearance) runs unchanged; it is pure
decision code over signals, which is exactly why it was built that way.

## 3. Storage at millions of detections

- **TimescaleDB hypertables** over the detections table, partitioned by day,
  compressed after 7 days (10× reduction), with the existing composite indexes —
  `(plate_raw, ts)` for trail lookups, `(camera_id, ts)` for per-camera queries —
  becoming covering indexes that include `ocr_conf` and `embed`.
- Hot window (30 days) in Timescale; older detections roll to S3/Parquet,
  queryable via an external table.
- Trajectories, alerts, and the camera graph stay in PostGIS (spatial joins for
  heatmaps already work there).
- Embeddings move to a vector index (pgvector / FAISS sidecar) for fast
  appearance-only candidate lookup.

## 4. What changes when the data is real

- **Travel-time distributions:** today's per-edge p5/p50/p95 become
  per-edge × hour-of-day × weekday distributions, refit nightly from the
  detection stream; the fitting code is already batch-shaped. Real data also
  brings sensor clock skew — edges will need per-camera clock-offset estimation,
  which the simulator never exercised.
- **Re-ID model:** the demo's color/shape descriptor is replaced by a CNN
  trained on real Indian vehicle images (multi-angle, dusty plates, roof
  carriers). Real appearance similarity spreads differently, so the
  `appearance_floor` and ambiguity-margin thresholds get re-tuned against a
  labeled re-acquisition set — the tiered structure stays.
- **OCR confidence calibration** must be re-measured on real cameras: the
  conf→fuzzy-threshold mapping is only as good as its calibration curve.

*Design only — nothing here is implemented in this repo.*
