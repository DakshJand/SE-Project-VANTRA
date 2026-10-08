"""Diagnostic: why does full-path stitching drop at 2 missed cameras?

Replays the stitching benchmark and classifies every failed hop as:
- NO_CANDIDATES   — no other detection was route+time plausible at all
- CAUTIOUS_REFUSAL— plausible candidates existed but none matched plate/appearance,
                    or an ambiguity tie was (correctly) not forced
- WRONG_MATCH     — a hop linked detections from DIFFERENT ground-truth trips (bug)

Run: python scripts/diagnose_stitching.py [--miss 2] [--trials 40]
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import network, matcher as M
from app.services.reconstruct import build_matcher
from app.ml import imagegen
from app.ml import reid as reid_mod


def _veh_embed(vehicle):
    s = imagegen.generate_sample(plate="BASE0", vehicle_type=vehicle["type"],
                                 color=vehicle["color"], condition="clean",
                                 seed=hash((vehicle["type"], vehicle["color"])) % 10**9)
    return reid_mod.embed(s.vehicle_img, vehicle["type"])
from app.ml.simulator import simulate_episode


async def diagnose(n_miss: int, n_trials: int) -> dict:
    rng = random.Random(4242)   # same seed family as evaluate.py for comparability
    m = await build_matcher()

    outcomes = Counter()
    failure_examples: list[str] = []
    wrong_match_details: list[str] = []
    hop_by_outcome = Counter()

    for t in range(n_trials):
        route = random.Random(4242 + t).random and None  # placeholder, replaced below
    # NOTE: evaluate.py uses rng = random.Random(4242) then draws sequentially;
    # replicate that exactly for comparability.
    rng = random.Random(4242)
    trials = 0
    for t in range(n_trials):
        route = _walk(rng)
        if len(set(route)) < len(route):
            continue
        skip = set(rng.sample(route[1:], min(n_miss, len(route) - 1)))
        start = datetime(2026, 9, 1, 10) + timedelta(minutes=rng.randint(0, 300))
        vehicle = {"type": rng.choice(["car", "suv", "truck", "auto"]),
                   "color": rng.choice(["white", "black", "silver", "red", "blue"])}
        plate = imagegen.random_plate(rng)
        passes = simulate_episode(rng, start, plate, vehicle, f"diag-{t}",
                                  route=route, camera_miss_rate=0.0,
                                  skip_cameras=skip, stop_dwell=True)
        emitted = [p for p in passes if not p.missed]
        if len(emitted) < 2:
            continue
        trials += 1
        base_embed = _veh_embed(vehicle)
        dets = []
        for i, p in enumerate(emitted):
            import numpy as _np
            noise = _np.random.default_rng(9000 + t * 100 + i).normal(0, 0.02, base_embed.shape)
            e = base_embed + noise.astype(_np.float32)
            e = e / (_np.linalg.norm(e) + 1e-8)
            dets.append(M.Detection(
                5000000 + t * 100 + i, p.camera_id, p.ts, p.ocr_plate, p.ocr_conf,
                p.plate, vehicle["type"], vehicle["color"], None, None,
                [float(x) for x in e], p.session_id))
        traj = m.stitch_session(dets)
        gt_ids = {d.det_id for d in dets}

        if traj:
            # check every hop for cross-session (wrong) links — with a single
            # session's detections this cannot happen, so also verify hop ORDER
            for h in traj.hops:
                a = next(d for d in dets if d.det_id == h.from_det)
                b = next(d for d in dets if d.det_id == h.to_det)
                if a.ts >= b.ts:
                    outcomes["WRONG_MATCH"] += 1
                    wrong_match_details.append(f"trial {t}: time-inverted hop {a.camera_id}->{b.camera_id}")
                elif not _in_order(traj, a, b):
                    outcomes["WRONG_MATCH"] += 1
                    wrong_match_details.append(f"trial {t}: out-of-order hop in trajectory")
            linked = len(traj.hops)
        else:
            linked = 0
        n_expected = len(emitted) - 1
        if linked == n_expected:
            outcomes["FULL_PATH"] += 1
            continue

        # some hop(s) failed — classify each consecutive pair that didn't link
        linked_pairs = {(h.from_det, h.to_det) for h in (traj.hops if traj else [])}
        for i in range(len(dets) - 1):
            a, b = dets[i], dets[i + 1]
            if (a.det_id, b.det_id) in linked_pairs:
                hop_by_outcome["LINKED"] += 1
                continue
            reason = classify_gap(m, dets, a, b, linked_pairs)
            hop_by_outcome[reason] += 1
            if reason != "LINKED" and len(failure_examples) < 12:
                failure_examples.append(
                    f"trial {t} [{plate}] hop {a.camera_id}->{b.camera_id} "
                    f"({(b.ts-a.ts).total_seconds():.0f}s): {reason}")
    return {
        "trials": trials,
        "outcomes": dict(outcomes),
        "hop_outcomes": dict(hop_by_outcome),
        "examples": failure_examples,
        "wrong": wrong_match_details,
    }


def _walk(rng):
    from app.ml.simulator import random_walk_route
    return random_walk_route(rng, 4, 7)


def _in_order(traj: M.Trajectory, a: M.Detection, b: M.Detection) -> bool:
    order = {d.det_id: i for i, d in enumerate(traj.detections)}
    return order.get(a.det_id, -1) < order.get(b.det_id, -1)


def classify_gap(m: M.Matcher, dets: list[M.Detection], a: M.Detection,
                 b: M.Detection, linked_pairs: set) -> str:
    """Why did a->b not link?"""
    dt = (b.ts - a.ts).total_seconds()
    # 1. any route plausibility at all?
    plausibles = m.route_plausible(a.camera_id, b.camera_id, dt)
    if not plausibles:
        # check hard-floor too — would it have been an anomaly (that's a refusal of a
        # different kind, but in this benchmark no anomalies are injected)
        return "NO_ROUTE_PLAUSIBILITY"
    # 2. would the direct link have scored as exact/fuzzy?
    ev = m.link(a, b)
    if ev is None:
        # plate mismatch beyond threshold AND appearance insufficient
        ed = M.edit_distance(a.plate_raw, b.plate_raw)
        thresh = M.fuzzy_threshold(a.ocr_conf, b.ocr_conf)
        cos = M.reid.cosine(a.embed, b.embed) if a.embed and b.embed else None
        if a.readable and b.readable and ed > thresh:
            return f"PLATE_MISMATCH(ed={ed}>thr={thresh})"
        if cos is not None and cos < m.appearance_floor:
            return f"APPEARANCE_BELOW_FLOOR(cos={cos:.2f})"
        return "OTHER_LINK_REFUSAL"
    # 3. link() returned evidence but stitch_session didn't use it? (ordering/gap)
    if ev.anomaly:
        return "HARD_ANOMALY(min_time)"
    # ev exists but wasn't in linked_pairs -> stitch_session dropped it (route not
    # plausible via link() internal, or time window)
    return "LINK_EXISTS_BUT_NOT_USED"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--miss", type=int, default=2)
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args()
    res = await diagnose(args.miss, args.trials)
    print(f"=== {args.miss} missed cameras, {res['trials']} trials ===")
    print("path outcomes:", res["outcomes"])
    print("hop outcomes:", res["hop_outcomes"])
    print("\nfailure examples:")
    for e in res["examples"]:
        print(" ", e)
    if res["wrong"]:
        print("\nWRONG MATCHES (bugs):")
        for w in res["wrong"]:
            print(" ", w)
    else:
        print("\nno wrong matches detected")


if __name__ == "__main__":
    asyncio.run(main())
