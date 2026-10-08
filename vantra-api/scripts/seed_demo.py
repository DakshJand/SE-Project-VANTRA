"""Seed the 6 demo scenarios.

Each scenario is a scripted simulated trip with known ground truth, injected into
the DB as a session (detections with images for demo crops), plus demo_scenarios
metadata pointing at the plates. The demo runner and dashboard use these.

Run: python scripts/seed_demo.py
"""
from __future__ import annotations

import asyncio
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app import db
from app.config import settings
from app.services import network
from app.ml import simulator, imagegen, reid

DEMO_T0 = datetime(2026, 9, 5, 9, 0, 0)   # demo day


async def insert_passes(passes: list[simulator.Pass], session: str,
                        render_images: bool = True) -> list[int]:
    """Write passes (skipping missed) as detections, with rendered images + embeddings."""
    img_dir = settings.image_dir / "demo"
    rng = random.Random(hash(session) % 10**9)
    ids = []
    for i, p in enumerate(passes):
        if p.missed:
            continue
        s = imagegen.generate_sample(
            plate=p.plate, vehicle_type=p.vehicle["type"], color=p.vehicle["color"],
            condition=p.condition, seed=rng.randint(0, 10**9))
        crop_path = plate_path = None
        embed = None
        if render_images:
            crop_path, plate_path = imagegen.save_sample(s, img_dir, f"{session}_{i}")
            embed = reid.embed(s.vehicle_img, p.vehicle["type"])
        else:
            base = reid.embed(imagegen.generate_sample(
                plate=p.plate, vehicle_type=p.vehicle["type"], color=p.vehicle["color"],
                condition="clean", seed=rng.randint(0, 10**9)).vehicle_img, p.vehicle["type"])
            embed = base + np.random.default_rng(i).normal(0, 0.02, base.shape).astype(np.float32)
            embed = embed / (np.linalg.norm(embed) + 1e-8)
        det_id = await db.fetch_val(
            """INSERT INTO detections (camera_id, ts, plate_raw, ocr_conf, plate_correct,
                 vehicle_type, vehicle_color, crop_path, plate_crop_path, embed, session_id, gt_cam)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12) RETURNING det_id""",
            [p.camera_id, p.ts, p.ocr_plate, p.ocr_conf, p.plate, p.vehicle["type"],
             p.vehicle["color"], crop_path, plate_path, [float(x) for x in embed],
             session, p.camera_id])
        ids.append(det_id)
    return ids


async def main() -> None:
    rng = random.Random(60)
    await db.execute("DELETE FROM demo_scenarios")

    scenarios = []

    # ------------------------------------------------------------------ 1. clean baseline
    plate1 = "KA01VX2026"
    v1 = {"type": "car", "color": "white"}
    route1 = ["CAM01", "CAM02", "CAM03", "CAM05", "CAM07", "CAM06"]
    passes1 = simulator.simulate_episode(
        rng, DEMO_T0, plate1, v1, "demo1-clean", route=route1,
        camera_miss_rate=0.0, force_conditions={i: "clean" for i in range(len(route1))})
    ids1 = await insert_passes(passes1, "demo1-clean")
    scenarios.append(("demo1_clean", "Scenario 1 — Clean baseline trajectory",
                      "Fully readable plate across all 6 hops (CAM01→CAM06). "
                      "Every hop is an exact match on a route-plausible edge.",
                      plate1, {"session": "demo1-clean", "det_ids": ids1, "route": route1,
                               "narrative": "A perfectly readable plate crosses all six cameras — watch every hop link by exact plate match at high confidence."}))

    # ------------------------------------------------- 2. blurred hop resolved by appearance + route
    plate2 = "KA05HR7741"
    v2 = {"type": "suv", "color": "red"}
    route2 = ["CAM13", "CAM03", "CAM05", "CAM07", "CAM06"]
    # hop index 1 (CAM03) blurred -> OCR garbles the plate; matcher must still link
    passes2 = simulator.simulate_episode(
        rng, DEMO_T0 + timedelta(hours=1), plate2, v2, "demo2-blur", route=route2,
        camera_miss_rate=0.0,
        force_conditions={i: ("blur" if i == 1 else "clean") for i in range(len(route2))})
    # force a partial read at CAM03 (half-read plate). OCR conf 0.35 — consistent
    # with the EasyOCR distribution for a badly blurred 3-char-corrupted read,
    # and low enough that the matcher's confidence-aware threshold widens to ed 3.
    for p in passes2:
        if p.camera_id == "CAM03" and p.ocr_plate:
            p.ocr_plate = simulator.corrupt_plate(plate2, rng, 3)
            p.ocr_conf = 0.35
    ids2 = await insert_passes(passes2, "demo2-blur")
    scenarios.append(("demo2_blur_hop", "Scenario 2 — Blurred hop, appearance + route",
                      "CAM03 capture is blurred: plate half-read (3 char errors, OCR conf 0.35). "
                      "Hop resolved via fuzzy match + Re-ID appearance + route plausibility. "
                      "Open the trajectory and hover hop 2 for the evidence.",
                      plate2, {"session": "demo2-blur", "det_ids": ids2, "route": route2,
                               "narrative": "The plate is half-read at CAM03 because of motion blur (OCR confidence 0.35) — watch the system fall back to fuzzy matching backed by vehicle appearance and route plausibility instead of guessing."}))

    # ------------------------------------------------------------- 3. missed camera, still stitched
    plate3 = "KA53MN1108"
    v3 = {"type": "truck", "color": "blue"}
    route3 = ["CAM09", "CAM01", "CAM02", "CAM03", "CAM04"]
    passes3 = simulator.simulate_episode(
        rng, DEMO_T0 + timedelta(hours=2), plate3, v3, "demo3-missed", route=route3,
        camera_miss_rate=0.0, skip_cameras={"CAM02"},
        force_conditions={i: "clean" for i in range(len(route3))})
    ids3 = await insert_passes(passes3, "demo3-missed")
    scenarios.append(("demo3_missed_camera", "Scenario 3 — Missed camera, hop skipped",
                      "CAM02 fails to detect the vehicle entirely. The matcher widens the "
                      "time window across two hops (CAM01→CAM03) and still stitches the full path.",
                      plate3, {"session": "demo3-missed", "det_ids": ids3, "route": route3,
                               "narrative": "One camera fails to detect the vehicle at all — watch the matcher widen its time window across two hops and still stitch the full path without inventing a sighting."}))

    # ------------------------------------------- 4. long dwell on stop-eligible edge, lower confidence
    plate4 = "KA02PS3390"
    v4 = {"type": "auto", "color": "yellow"}
    route4 = ["CAM03", "CAM05", "CAM04"]      # CAM05->CAM04 is stop-eligible (100ft Rd market)
    passes4 = simulator.simulate_episode(
        rng, DEMO_T0 + timedelta(hours=3), plate4, v4, "demo4-dwell", route=route4,
        camera_miss_rate=0.0, stop_dwell=True,
        force_conditions={i: "clean" for i in range(len(route4))})
    # force a long dwell at the market edge (40 min)
    edges = {e.edge_id: e for e in network.build_edges()}
    p_last = passes4[-1]
    from datetime import datetime as dt
    passes4[-1] = simulator.Pass(
        p_last.camera_id, p_last.ts + timedelta(minutes=40), p_last.plate, p_last.vehicle,
        p_last.session_id, p_last.condition, p_last.ocr_plate, p_last.ocr_conf, p_last.missed)
    ids4 = await insert_passes(passes4, "demo4-dwell")
    scenarios.append(("demo4_dwell", "Scenario 4 — Long dwell at market edge",
                      "Vehicle stops 40 minutes on the stop-eligible CAM05→CAM04 market edge. "
                      "Still matched (never-zero decay), visibly lower hop confidence.",
                      plate4, {"session": "demo4-dwell", "det_ids": ids4, "route": route4,
                               "narrative": "The vehicle stops 40 minutes at a market stretch — watch confidence visibly decay (but never hit zero) instead of the trail being silently dropped."}))

    # --------------------------------------------------- 5. impossible traversal (cloned plate)
    plate5 = "KA41CL9042"
    v5 = {"type": "car", "color": "black"}
    route5 = ["CAM11", "CAM01", "CAM02", "CAM03"]
    passes5 = simulator.simulate_episode(
        rng, DEMO_T0 + timedelta(hours=4), plate5, v5, "demo5-clone", route=route5,
        camera_miss_rate=0.0, impossible_at=(1, 2),   # CAM01->CAM02 in < min_time
        force_conditions={i: "clean" for i in range(len(route5))})
    ids5 = await insert_passes(passes5, "demo5-clone")
    scenarios.append(("demo5_clone", "Scenario 5 — Spoofed/cloned plate (hard anomaly)",
                      "Same plate seen at CAM01 and CAM02 within a physically impossible time "
                      "(below the min_time floor). Flagged as a HARD anomaly with evidence — "
                      "not silently linked into a trajectory.",
                      plate5, {"session": "demo5-clone", "det_ids": ids5, "route": route5,
                               "narrative": "The same plate shows up at two cameras in less time than physically possible — watch the system flag a cloned-plate anomaly instead of silently linking them into one trip."}))

    # ------------------------------------------------------------ 6. blacklisted plate, live alert
    plate6 = "KA09ST0555"
    v6 = {"type": "suv", "color": "black"}
    route6 = ["CAM14", "CAM13", "CAM12", "CAM11", "CAM01"]
    await db.execute(
        """INSERT INTO blacklist (plate, reason) VALUES ($1, $2)
           ON CONFLICT (plate) DO UPDATE SET reason = EXCLUDED.reason""",
        [plate6, "SIH demo: reported stolen vehicle (simulated)"])
    passes6 = simulator.simulate_episode(
        rng, DEMO_T0 + timedelta(hours=5), plate6, v6, "demo6-blacklist", route=route6,
        camera_miss_rate=0.0,
        force_conditions={i: "clean" for i in range(len(route6))})
    ids6 = await insert_passes(passes6, "demo6-blacklist")
    # trigger the alert now
    from app.services import alerts as A
    from app.services.reconstruct import det_from_row
    rows = await db.fetch(
        "SELECT * FROM detections WHERE session_id = $1 ORDER BY ts", ["demo6-blacklist"])
    dets = [det_from_row(r) for r in rows]
    bl_alerts = A.check_blacklist(dets, {plate6})
    for a in bl_alerts:
        await db.execute(
            """INSERT INTO alerts (kind, severity, plate, title, ts, det_ids, evidence)
               VALUES ($1,$2,$3,$4,$5,$6,$7)""",
            [a.kind, a.severity, a.plate, a.title, a.ts,
             [int(x) for x in a.det_ids], a.evidence])
    scenarios.append(("demo6_blacklist", "Scenario 6 — Blacklisted plate detected live",
                      "Reported-stolen vehicle KA09ST0555 crosses the network "
                      "(CAM14→CAM01). Blacklist alert fires at every camera with full "
                      "evidence (exact plate match, camera, OCR confidence).",
                      plate6, {"session": "demo6-blacklist", "det_ids": ids6,
                               "route": route6, "n_alerts": len(bl_alerts),
                               "narrative": "A reported-stolen vehicle crosses the network — watch a real-time alert fire with the full chain of evidence for every camera it passes."}))

    for key, title, desc, plate, payload in scenarios:
        await db.execute(
            """INSERT INTO demo_scenarios (key, title, description, plate, payload)
               VALUES ($1,$2,$3,$4,$5)
               ON CONFLICT (key) DO UPDATE SET title=EXCLUDED.title,
                 description=EXCLUDED.description, plate=EXCLUDED.plate, payload=EXCLUDED.payload""",
            [key, title, desc, plate, payload])
    print(f"seeded {len(scenarios)} demo scenarios:")
    for key, title, plate, payload in [(s[0], s[1], s[3], s[4]) for s in scenarios]:
        print(f"  {key:22s} plate={plate} dets={len(payload['det_ids'])}")
    db.pool().close()


if __name__ == "__main__":
    asyncio.run(main())
