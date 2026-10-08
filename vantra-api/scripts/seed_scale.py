"""Scale stress dataset: 120-camera grid + ~1.2M detections in a SEPARATE database
(vantra_scale). Does not touch the demo database.

Run: python scripts/seed_scale.py [--detections 1200000]
"""
from __future__ import annotations

import argparse
import asyncio
import math
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app import db
from app.config import settings

# Scale DB URL (separate from demo)
SCALE_URL = "postgresql://vantra:vantra@localhost:5432/vantra_scale"

GRID_N = 12            # 12x10 grid = 120 cameras
GRID_M = 10
LAT0, LNG0 = 12.90, 77.55
DLAT, DLNG = 0.012, 0.014   # ~1.3km x ~1.5km spacing


def build_grid():
    """Grid camera network: each node connects to its 4-neighbors (+ some diagonals)."""
    cams = []
    for i in range(GRID_N):
        for j in range(GRID_M):
            cams.append((f"S{i:02d}{j:02d}", f"Grid {i}-{j}",
                         LAT0 + i * DLAT, LNG0 + j * DLNG, "grid road"))
    adj = {c[0]: [] for c in cams}
    edges = []
    for i in range(GRID_N):
        for j in range(GRID_M):
            cid = f"S{i:02d}{j:02d}"
            for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < GRID_N and 0 <= nj < GRID_M:
                    nid = f"S{ni:02d}{nj:02d}"
                    adj[cid].append(nid)
                    adj[nid].append(cid)
    for cid, nbrs in adj.items():
        for n in nbrs:
            a = next(c for c in cams if c[0] == cid)
            b = next(c for c in cams if c[0] == n)
            d = math.hypot((a[2] - b[2]) * 111000, (a[3] - b[3]) * 101000)
            edges.append((cid, n, round(d, 1), 50))
    return cams, adj, edges


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", type=int, default=1_200_000)
    args = ap.parse_args()

    # swap the db pool to the scale database
    import app.db as dbm
    dbm._pool = None
    dbm.pool = lambda: _scale_pool()
    global _pool
    print(f"scale dataset -> {SCALE_URL}")

    cams, adj, edges = build_grid()
    print(f"grid: {len(cams)} cameras, {len(edges)} directed edges")

    # schema
    schema = (Path(__file__).parent / "schema.sql").read_text()
    await db.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    await db.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    await db.execute("CREATE EXTENSION IF NOT EXISTS fuzzystrmatch")
    await db.execute(schema)

    await db.executemany(
        """INSERT INTO cameras (camera_id, name, lat, lng, geom, road)
           VALUES ($1,$2,$3,$4, ST_SetSRID(ST_MakePoint($4,$3),4326), $5)""",
        [(c[0], c[1], c[2], c[3], c[4]) for c in cams])
    await db.executemany(
        """INSERT INTO edges (edge_id, src_cam, dst_cam, distance_m, speed_limit_kmph,
             stop_eligible, min_time_s)
           VALUES ($1,$2,$3,$4,$5,$6,$7)""",
        [(f"{a}->{b}", a, b, d, lim, (i % 17 == 0), round(d / (lim * 1.6) * 3.6, 1))
         for i, (a, b, d, lim) in enumerate(edges)])

    # synthetic travel-time stats: nominal p5/p50/p95 per edge
    stats = []
    for a, b, d, lim in edges:
        nominal = d / (lim / 3.6)
        stats.append((f"{a}->{b}", 1, nominal * 0.7, nominal * 1.2, nominal * 2.8,
                      nominal * 1.4, nominal * 0.5, 200))
    await db.executemany(
        """INSERT INTO edge_time_stats (edge_id, hop_reach, p5, p50, p95, mean, std, n_samples)
           VALUES ($1,$2,$3,$4,$5,$6,$7,$8)""", stats)

    # detections: bulk-generate directly (no image rendering at this scale)
    rng = random.Random(31337)
    n_target = args.detections
    CH = 20000
    t0 = datetime(2026, 8, 1)
    n_written = 0
    trip_plates = 40000
    plates = []
    ALPHA = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
    for _ in range(trip_plates):
        plates.append("KA" + f"{rng.randint(1, 99):02d}" +
                      "".join(rng.choice(ALPHA) for _ in range(2)) +
                      f"{rng.randint(0, 9999):04d}")

    rows = []
    while n_written < n_target:
        # a trip: 3-8 hops over the grid
        route = [rng.choice(cams)[0]]
        for _ in range(rng.randint(2, 7)):
            nbrs = adj[route[-1]]
            route.append(rng.choice(nbrs))
        t = t0 + timedelta(minutes=rng.randint(0, 60 * 24 * 30))
        plate = rng.choice(plates)
        vtype = rng.choice(["car", "suv", "truck", "bus", "auto"])
        color = rng.choice(["white", "black", "silver", "red", "blue", "grey"])
        for cam in route:
            d = next(e[2] for e in edges if e[0] == route[route.index(cam) - 1] and e[1] == cam) if route.index(cam) > 0 else 1300
            t = t + timedelta(seconds=d / (50 / 3.6) * rng.uniform(0.8, 2.5))
            # OCR model: 70% clean exact, 20% 1-char error, 6% multi-char, 4% unreadable
            r = rng.random()
            if r < 0.70:
                praw, conf = plate, rng.uniform(0.88, 0.99)
            elif r < 0.90:
                chars = list(plate)
                i = rng.randrange(len(chars))
                chars[i] = rng.choice(ALPHA)
                praw, conf = "".join(chars), rng.uniform(0.6, 0.85)
            elif r < 0.96:
                praw, conf = None, rng.uniform(0.1, 0.4)
            else:
                chars = list(plate)
                for i in rng.sample(range(len(chars)), 2):
                    chars[i] = rng.choice(ALPHA)
                praw, conf = "".join(chars), rng.uniform(0.4, 0.7)
            rows.append((cam, t, praw, conf, plate, vtype, color,
                         None, None, None, None, cam))
            n_written += 1
            if n_written >= n_target:
                break
        if len(rows) >= CH:
            await db.executemany(
                """INSERT INTO detections (camera_id, ts, plate_raw, ocr_conf,
                     plate_correct, vehicle_type, vehicle_color, crop_path,
                     plate_crop_path, embed, session_id, gt_cam)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)""", rows)
            rows = []
            print(f"  {n_written:,} / {n_target:,}", flush=True)
    if rows:
        await db.executemany(
            """INSERT INTO detections (camera_id, ts, plate_raw, ocr_conf,
                 plate_correct, vehicle_type, vehicle_color, crop_path,
                 plate_crop_path, embed, session_id, gt_cam)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)""", rows)
    print(f"detections: {n_written:,} written")

    # indexes (schema has them, but ensure ANALYZE for the planner)
    await db.execute("ANALYZE detections")
    print("analyze done")


_pool = None


def _scale_pool():
    global _pool
    if _pool is None:
        import psqlpy
        _pool = psqlpy.ConnectionPool(dsn=SCALE_URL, max_db_pool_size=10)
    return _pool


if __name__ == "__main__":
    asyncio.run(main())
