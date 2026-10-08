"""Generate the synthetic historical ANPR dataset for VANTRA.

- cameras + edges from app.services.network
- weeks of simulated trips (commuter vehicles with daily patterns + episodic traffic)
- per-detection: OCR text/conf via image synthesis + real OCR pass (sampled subset),
  embeddings from Re-ID, images on disk for a sampled subset (full set would be huge)
- travel-time stats fitted per edge (direct + 2-hop) from the simulated ground truth

Run:  python scripts/seed_history.py [--days 21] [--fast]
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app import db
from app.config import settings, DATA_DIR
from app.services import network
from app.ml import simulator, imagegen, ocr, reid


async def seed_network() -> None:
    await db.execute("DELETE FROM trajectory_hops")
    await db.execute("DELETE FROM trajectories")
    await db.execute("DELETE FROM detections")
    rows = [(cid, name, lat, lng, road) for cid, name, lat, lng, road in network.CAMERAS]
    await db.executemany(
        """INSERT INTO cameras (camera_id, name, lat, lng, geom, road)
           VALUES (%s,%s,%s,%s, ST_SetSRID(ST_MakePoint(%s,%s),4326), %s)
           ON CONFLICT (camera_id) DO UPDATE SET name=EXCLUDED.name, lat=EXCLUDED.lat,
             lng=EXCLUDED.lng, geom=EXCLUDED.geom, road=EXCLUDED.road""",
        [(r[0], r[1], r[2], r[3], r[3], r[2], r[4]) for r in rows],
    )
    edges = network.build_edges()
    await db.executemany(
        """INSERT INTO edges (edge_id, src_cam, dst_cam, distance_m, speed_limit_kmph,
             stop_eligible, min_time_s)
           VALUES (%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT (edge_id) DO UPDATE SET distance_m=EXCLUDED.distance_m,
             speed_limit_kmph=EXCLUDED.speed_limit_kmph, stop_eligible=EXCLUDED.stop_eligible,
             min_time_s=EXCLUDED.min_time_s""",
        [(e.edge_id, e.src, e.dst, e.distance_m, e.speed_limit_kmph,
          e.stop_eligible, e.min_time_s) for e in edges],
    )
    print(f"network: {len(rows)} cameras, {len(edges)} directed edges")


async def fit_travel_time_stats() -> None:
    """Fit per-edge travel-time distributions from ground-truth consecutive passes
    (session_id, gt ordering) — direct (1-hop) and 2-hop (skipped camera)."""
    rows = await db.fetch("""
        SELECT session_id, gt_cam, ts FROM detections
        WHERE gt_cam IS NOT NULL ORDER BY session_id, ts
    """)
    by_session: dict[str, list[tuple[str, datetime]]] = {}
    for r in rows:
        by_session.setdefault(r["session_id"], []).append((r["gt_cam"], r["ts"]))

    edges = {e.edge_id: e for e in network.build_edges()}
    direct: dict[str, list[float]] = {}
    twohop: dict[str, list[float]] = {}
    for sess, passes in by_session.items():
        for i in range(len(passes) - 1):
            (c1, t1), (c2, t2) = passes[i], passes[i + 1]
            dt = (t2 - t1).total_seconds()
            if dt <= 0 or dt > 6 * 3600:
                continue
            eid = f"{c1}->{c2}"
            if eid in edges:
                direct.setdefault(eid, []).append(dt)
            else:
                eid2 = f"{c1}->{c2}"
                twohop.setdefault(eid2, []).append(dt)

    # 2-hop pairs: cam -> cam2 reachable via exactly one intermediate (which was missed)
    two_hop_map = network.two_hop_pairs()
    for sess, passes in by_session.items():
        for i in range(len(passes) - 1):
            (c1, t1), (c2, t2) = passes[i], passes[i + 1]
            if c2 in two_hop_map.get(c1, []):
                dt = (t2 - t1).total_seconds()
                if 0 < dt <= 6 * 3600:
                    twohop.setdefault(f"{c1}->{c2}", []).append(dt)

    def pct(v: list[float], q: float) -> float:
        return float(np.percentile(v, q))

    stats_rows = []
    for eid, vals in direct.items():
        if len(vals) < 5:
            continue
        stats_rows.append((eid, 1, pct(vals, 5), pct(vals, 50), pct(vals, 95),
                           float(np.mean(vals)), float(np.std(vals)), len(vals)))
    for eid, vals in twohop.items():
        if len(vals) < 5:
            continue
        stats_rows.append((eid, 2, pct(vals, 5), pct(vals, 50), pct(vals, 95),
                           float(np.mean(vals)), float(np.std(vals)), len(vals)))
    await db.execute("DELETE FROM edge_time_stats")
    await db.executemany(
        """INSERT INTO edge_time_stats (edge_id, hop_reach, p5, p50, p95, mean, std, n_samples)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""", stats_rows)
    # also fill convenience columns on edges for direct stats
    for eid, hop, p5, p50, p95, mean, std, n in stats_rows:
        if hop == 1:
            await db.execute(
                """UPDATE edges SET tt_p5=%s, tt_p50=%s, tt_p95=%s, tt_mean=%s, tt_std=%s
                   WHERE edge_id=%s""", (p5, p50, p95, mean, std, eid))
    print(f"travel-time stats fitted: {sum(1 for r in stats_rows if r[1]==1)} direct, "
          f"{sum(1 for r in stats_rows if r[1]==2)} two-hop")


def generate_history(days: int, fast: bool, seed: int = 20260905) -> list[simulator.Pass]:
    """Simulate `days` of traffic. Returns all non-missed passes."""
    rng = random.Random(seed)
    t0 = datetime(2026, 8, 1, 0, 0, 0)
    passes: list[simulator.Pass] = []

    # commuter vehicles: daily trips over fixed-ish routes at commute hours
    commuters = []
    for i in range(simulator.REPEAT_VEHICLES):
        v = simulator.make_vehicle(rng, i)
        v["plate"] = imagegen.random_plate(rng)
        home, work = rng.sample([c[0] for c in network.CAMERAS], 2)
        v["route_to"] = simulator.random_walk_route(rng, 4, 6)
        v["route_back"] = simulator.random_walk_route(rng, 4, 6)
        commuters.append(v)

    for day in range(days):
        date = t0 + timedelta(days=day)
        for v in commuters:
            if rng.random() < 0.15:   # vehicle not on the road today
                continue
            dep_out = date + timedelta(hours=rng.gauss(9, 0.7), minutes=rng.randint(-20, 20))
            passes += simulator.simulate_episode(
                rng, dep_out, v["plate"], v, f"hist-c{v['plate']}-{day}-out",
                route=v["route_to"] if rng.random() < 0.7 else None)
            if rng.random() < 0.9:
                dep_back = date + timedelta(hours=rng.gauss(18.5, 0.8), minutes=rng.randint(-20, 20))
                passes += simulator.simulate_episode(
                    rng, dep_back, v["plate"], v, f"hist-c{v['plate']}-{day}-back",
                    route=v["route_back"] if rng.random() < 0.7 else None)
        # episodic traffic
        for _ in range(simulator.EPISODE_VEHICLES // 7):
            v = simulator.make_vehicle(rng, 0)
            v["plate"] = imagegen.random_plate(rng)
            start = date + timedelta(hours=rng.uniform(6, 23))
            passes += simulator.simulate_episode(
                rng, start, v["plate"], v, f"hist-e{rng.randint(0,10**9)}")
    return passes


async def write_detections(passes: list[simulator.Pass], fast: bool) -> None:
    """Persist passes as detections. For a sampled subset, render the actual image,
    run real OCR + Re-ID; for the rest, use the simulator's ground-truth OCR model
    (fast mode) — image generation is the bottleneck."""
    img_dir = settings.image_dir / "history"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(777)
    rows = []
    image_fraction = 0.0 if fast else 0.12
    n_img = 0
    for i, p in enumerate(passes):
        if p.missed:
            continue
        crop_path = plate_path = None
        embed_vec = None
        ocr_plate, ocr_conf = p.ocr_plate, p.ocr_conf
        if not fast and rng.random() < image_fraction:
            s = imagegen.generate_sample(plate=p.plate, vehicle_type=p.vehicle["type"],
                                         color=p.vehicle["color"], condition=p.condition,
                                         seed=rng.randint(0, 10**9))
            stem = f"det{i:07d}"
            crop_path, plate_path = imagegen.save_sample(s, img_dir, stem)
            # real OCR over the rendered plate (validation of the OCR error model)
            text, conf = ocr.read_plate(s.plate_img)
            if text:
                ocr_plate, ocr_conf = text, conf
            else:
                ocr_plate, ocr_conf = None, conf
            embed_vec = reid.embed(s.vehicle_img, s.vehicle["type"])
            n_img += 1
        else:
            # synthetic embedding: deterministic per vehicle with small noise so
            # same-vehicle embeddings cluster
            base = reid.embed(imagegen.generate_sample(
                plate=p.plate, vehicle_type=p.vehicle["type"], color=p.vehicle["color"],
                condition="clean", seed=hash(p.plate) % 10**9).vehicle_img, p.vehicle["type"])
            noise = np.random.default_rng(i).normal(0, 0.02, base.shape).astype(np.float32)
            embed_vec = base + noise
            embed_vec /= np.linalg.norm(embed_vec) + 1e-8
        rows.append((
            p.camera_id, p.ts, ocr_plate, ocr_conf, p.plate, p.vehicle["type"],
            p.vehicle["color"], crop_path, plate_path,
            [float(x) for x in embed_vec] if embed_vec is not None else None,
            p.session_id, p.camera_id,
        ))
    # chunk inserts
    CH = 500
    for k in range(0, len(rows), CH):
        await db.executemany(
            """INSERT INTO detections (camera_id, ts, plate_raw, ocr_conf, plate_correct,
                 vehicle_type, vehicle_color, crop_path, plate_crop_path, embed, session_id, gt_cam)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", rows[k:k + CH])
    print(f"detections: {len(rows)} rows written ({n_img} with real images/OCR)")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--fast", action="store_true", help="skip image rendering entirely")
    args = ap.parse_args()

    await seed_network()
    passes = generate_history(args.days, args.fast)
    print(f"simulated {len(passes)} passes over {args.days} days "
          f"({sum(p.missed for p in passes)} missed)")
    await write_detections(passes, args.fast)
    await fit_travel_time_stats()
    db.pool().close()


if __name__ == "__main__":
    asyncio.run(main())
