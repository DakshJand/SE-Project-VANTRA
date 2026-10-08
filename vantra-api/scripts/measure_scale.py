"""Latency measurements at scale (1.2M detections, 120 cameras).

Measures: trajectory reconstruction, wildcard search, plate search, heatmap
analytics, tuning preview, and the scale-limiting queries. Run against the
vantra_scale database.

Run: python scripts/measure_scale.py
"""
from __future__ import annotations

import asyncio
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.db as dbm
import psqlpy

SCALE_URL = "postgresql://vantra:vantra@localhost:5432/vantra_scale"

RESULTS: list[tuple[str, float, str]] = []


def bench(name, coro_fn, note=""):
    async def run():
        t0 = time.perf_counter()
        out = await coro_fn()
        dt = time.perf_counter() - t0
        RESULTS.append((name, dt, note))
        print(f"  {name:44s} {dt*1000:9.1f} ms   {note}")
        return out
    return run()


async def main() -> None:
    dbm._pool = None
    dbm.pool = lambda: _pool()
    await dbm.execute("SELECT 1")

    n = await dbm.fetch_val("SELECT count(*) FROM detections")
    print(f"scale dataset: {n:,} detections, 120 cameras")

    # pick a real busy plate from the data
    busy = await dbm.fetch_one(
        """SELECT plate_raw, count(*) c FROM detections
           WHERE plate_raw IS NOT NULL GROUP BY plate_raw ORDER BY c DESC LIMIT 1""")
    plate = busy["plate_raw"]
    print(f"busiest plate: {plate} ({busy['c']} detections)")

    print("\n--- query latencies (mean of 3) ---")

    async def timed(name, fn, note=""):
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            await fn()
            times.append(time.perf_counter() - t0)
        med = sorted(times)[1]
        RESULTS.append((name, med, note))
        print(f"  {name:44s} {med*1000:9.1f} ms   {note}")

    # 1. trajectory reconstruction for a plate (the dashboard's core query)
    from app.services.reconstruct import build_matcher, fetch_detections_for_plate
    from app.services import matcher as M

    async def reconstruct():
        dets = await fetch_detections_for_plate(plate)
        m = await build_matcher()
        m.stitch(dets)
    await timed("trajectory reconstruct (busiest plate)", reconstruct)

    # 2. matcher build (stats load) — amortized cost per query today
    await timed("matcher build (edge stats load)",
                lambda: build_matcher())

    # 3. wildcard search
    prefix = plate[:5] + "*"
    await timed(f"wildcard search ({prefix})",
                lambda: dbm.fetch(
                    """SELECT DISTINCT plate_raw, count(*) FROM detections
                       WHERE plate_raw LIKE $1 GROUP BY plate_raw LIMIT 50""", [prefix.replace("*", "%")]))

    # 4. heatmap analytics (count per camera)
    await timed("heatmap (count per camera, full table)",
                lambda: dbm.fetch(
                    """SELECT camera_id, count(*) FROM detections
                       GROUP BY camera_id"""))

    # 5. heatmap with time window
    await timed("heatmap (24h window)",
                lambda: dbm.fetch(
                    """SELECT camera_id, count(*) FROM detections
                       WHERE ts > '2026-08-25' GROUP BY camera_id"""))

    # 6. segment speeds (the heaviest analytics query — self-join)
    await timed("segment speeds (self-join, 45min window)",
                lambda: dbm.fetch(
                    """SELECT a.camera_id src, b.camera_id dst, count(*)
                       FROM detections a JOIN detections b
                         ON b.plate_raw = a.plate_raw
                        AND b.ts > a.ts AND b.ts - a.ts < interval '45 minutes'
                        AND b.ts - a.ts > interval '1 minute'
                       WHERE a.plate_raw IS NOT NULL AND a.ts > '2026-08-28'
                       GROUP BY 1,2 LIMIT 20"""))

    # 7. tuning preview core: stitch 100 sessions
    sessions = await dbm.fetch(
        """SELECT session_id FROM detections GROUP BY session_id
           HAVING count(*) >= 4 ORDER BY random() LIMIT 100""")
    session_ids = [s["session_id"] for s in sessions]

    async def tuning_preview():
        m = await build_matcher()
        for sid in session_ids:
            rows = await dbm.fetch(
                "SELECT * FROM detections WHERE session_id = $1 ORDER BY ts", [sid])
            dets = [M.Detection(
                r["det_id"], r["camera_id"], r["ts"], r["plate_raw"], r["ocr_conf"],
                r["plate_correct"], r["vehicle_type"], r["vehicle_color"],
                r["crop_path"], r["plate_crop_path"],
                list(r["embed"]) if r.get("embed") else None, r["session_id"])
                for r in rows]
            m.stitch_session(dets)
    await timed("tuning preview core (100 sessions)", tuning_preview)

    # 8. live session creation (busiest-hour scan)
    async def live_create():
        rows = await dbm.fetch("SELECT ts FROM detections ORDER BY ts")
        from collections import deque
        window = deque()
        best_n = 0
        for r in rows:
            window.append(r["ts"])
            while (r["ts"] - window[0]).total_seconds() > 3600:
                window.popleft()
            best_n = max(best_n, len(window))
    await timed("live session create (busiest-hour scan)", live_create)

    print("\nsummary:")
    for name, dt, note in RESULTS:
        flag = " ⚠ SLOW" if dt > 5 else ""
        print(f"  {name:44s} {dt*1000:9.1f} ms{flag}")

    _pool_obj.close()


_pool_obj = None


def _pool():
    global _pool_obj
    if _pool_obj is None:
        _pool_obj = psqlpy.ConnectionPool(dsn=SCALE_URL, max_db_pool_size=10)
    return _pool_obj


if __name__ == "__main__":
    asyncio.run(main())
