"""What-if camera placement: would a hypothetical camera close trajectory gaps?

Purely additive — builds a SHADOW matcher with the hypothetical camera added to
the graph, re-stitches a benchmark sample of ground-truth trips, and reports the
before/after. Never touches the production matcher or tables.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

from app.services import network, matcher as M
from app.services.reconstruct import build_matcher


def _candidate_junctions() -> list[dict]:
    """Plausible new-camera locations.

    Two families of candidates:
    1. Midpoints of long, poorly-covered existing edges.
    2. Midpoints of camera PAIRS that are 3+ hops apart on the graph but
       geographically close — a camera there bridges an unlinkable corridor
       (the skip-one design cannot stitch >2-hop gaps). Pure edge-midpoint
       candidates can never fix a corridor gap, which is the planning
       question this tool exists to answer.
    """
    edges = network.build_edges()
    cam = {c[0]: c for c in network.CAMERAS}
    adj = network.adjacency_map()
    ids = sorted(cam)

    def hop_distance(c1: str, c2: str) -> int:
        if c1 == c2:
            return 0
        seen, frontier = {c1}, [c1]
        for d in range(1, 5):
            nxt = []
            for c in frontier:
                for n in adj.get(c, []):
                    if n == c2:
                        return d
                    if n not in seen:
                        seen.add(n)
                        nxt.append(n)
            frontier = nxt
        return 99

    cands = []
    seen = set()

    def add(name, lat, lng, note):
        key = (round(lat, 3), round(lng, 3))
        if key in seen:
            return
        seen.add(key)
        cands.append({
            "id": f"NEW{len(cands)+1:02d}",
            "lat": round(lat, 5), "lng": round(lng, 5),
            "name": name,
            "note": note,
        })

    # family 2 first: corridor bridges (3+ hops apart, within 3.5 km of each other)
    pairs = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            h = hop_distance(a, b)
            if 3 <= h < 99:
                d = network.haversine_m(cam[a][2], cam[a][3], cam[b][2], cam[b][3]) * 1.35
                if d <= 7000:  # midpoint stays within ~3.5 km of both ends
                    pairs.append((d, h, a, b))
    for d, h, a, b in sorted(pairs, key=lambda p: p[0]):
        if len(cands) >= 4:
            break
        # skip if an already-selected candidate's midpoint is within 800 m —
        # it's the same corridor and would duplicate the suggestion
        lat = (cam[a][2] + cam[b][2]) / 2
        lng = (cam[a][3] + cam[b][3]) / 2
        if any(network.haversine_m(lat, lng, c["lat"], c["lng"]) < 800 for c in cands):
            continue
        add(f"corridor bridge {a}–{b}", lat, lng,
            f"{a} and {b} are {h} hops apart; a camera between them closes the gap")

    # family 1: long existing edges
    for e in sorted(edges, key=lambda e: -e.distance_m):
        if e.src > e.dst:
            continue
        a, b = cam[e.src], cam[e.dst]
        if len(cands) >= 6:
            break
        add(f"midpoint {e.src}–{e.dst}",
            (a[2] + b[2]) / 2, (a[3] + b[3]) / 2,
            f"longest existing edge ({e.distance_m} m)")
    return cands


async def evaluate_placement(lat: float, lng: float, n_trials: int = 40,
                             seed: int = 777) -> dict:
    """Before/after stitching impact of a camera at (lat,lng)."""
    base_matcher = await build_matcher()

    # --- build shadow network with the hypothetical camera
    new_cam_id = "HYPO01"
    edges = network.build_edges()
    cam_by_id = {c[0]: c for c in network.CAMERAS}
    shadow_edges = list(edges)
    # connect the new camera to every real camera within 3.5 km road distance
    connected = []
    for cid, name, clat, clng, road in network.CAMERAS:
        d = network.haversine_m(lat, lng, clat, clng) * 1.35
        if d <= 3500:
            connected.append(cid)
            stop = False
            min_t = d / (40 * 1.6) * 3.6
            for a, b in ((new_cam_id, cid), (cid, new_cam_id)):
                shadow_edges.append(M.network.Edge(
                    f"{a}->{b}", a, b, round(d), 40, stop, round(min_t, 1)))
    if not connected:
        return {"error": "location too far from the existing network (>2.2 km from every camera)"}

    # --- benchmark: ground-truth trips with 2 adjacent cameras skipped (the hard case)
    from app.ml.simulator import simulate_episode, random_walk_route
    from app.ml import imagegen
    import numpy as np
    from app.ml import reid as reid_mod

    rng = random.Random(seed)
    adj_base = network.adjacency_map()
    trips = []
    for t in range(n_trials * 2):
        route = random_walk_route(rng, 4, 7)
        if len(set(route)) < len(route):
            continue
        skip = set(rng.sample(route[1:], min(2, len(route) - 1)))
        start = datetime(2026, 9, 10, 10) + timedelta(minutes=rng.randint(0, 300))
        vehicle = {"type": rng.choice(["car", "suv", "truck", "auto"]),
                   "color": rng.choice(["white", "black", "silver", "red", "blue"])}
        plate = imagegen.random_plate(rng)
        passes = simulate_episode(rng, start, plate, vehicle, f"whatif-{t}",
                                  route=route, camera_miss_rate=0.0,
                                  skip_cameras=skip, stop_dwell=True)
        emitted = [p for p in passes if not p.missed]
        if len(emitted) < 2:
            continue
        base_embed = reid_mod.embed(imagegen.generate_sample(
            plate="BASE0", vehicle_type=vehicle["type"], color=vehicle["color"],
            condition="clean", seed=hash((vehicle["type"], vehicle["color"])) % 10**9
        ).vehicle_img, vehicle["type"])
        dets = []
        for i, p in enumerate(emitted):
            noise = np.random.default_rng(9100 + t * 100 + i).normal(0, 0.02, base_embed.shape)
            e = base_embed + noise.astype(np.float32)
            e = e / (np.linalg.norm(e) + 1e-8)
            dets.append(M.Detection(
                8000000 + t * 100 + i, p.camera_id, p.ts, p.ocr_plate, p.ocr_conf,
                p.plate, vehicle["type"], vehicle["color"], None, None,
                [float(x) for x in e], p.session_id))
        # does this trip contain a 2-hop gap whose endpoints both touch the new
        # camera's connection set? (only computed later; store flag lazily)
        trips.append((dets, route, skip))
        if len(trips) >= n_trials:
            break

    def _hop_distance(c1: str, c2: str) -> int:
        """BFS hop distance on the base graph."""
        if c1 == c2:
            return 0
        seen = {c1}
        frontier = [c1]
        for d in range(1, 5):
            nxt = []
            for c in frontier:
                for n in adj_base.get(c, []):
                    if n == c2:
                        return d
                    if n not in seen:
                        seen.add(n)
                        nxt.append(n)
            frontier = nxt
        return 99

    def trip_affected(trip) -> bool:
        """Trip has a gap that is UNLINKABLE today (>2 hops, outside the skip-one
        design) whose endpoints both touch the hypothetical camera — i.e. a gap
        the new camera would actually close."""
        dets, _route, _skip = trip
        for i in range(len(dets) - 1):
            a, b = dets[i], dets[i + 1]
            if a.camera_id == b.camera_id:
                continue
            if (_hop_distance(a.camera_id, b.camera_id) > 2
                    and a.camera_id in connected and b.camera_id in connected):
                return True
        return False

    # --- baseline: how many trips fully stitch today
    def score(m: M.Matcher) -> dict:
        full = 0; links = 0; possible = 0; weak_hops = 0
        for dets, route, skip in trips:
            traj = m.stitch_session(dets)
            n_h = len(traj.hops) if traj else 0
            links += n_h
            possible += len(dets) - 1
            if n_h == len(dets) - 1:
                full += 1
            if traj:
                weak_hops += sum(1 for h in traj.hops if h.confidence < 0.6)
        return {"trials": len(trips), "full_pct": round(100*full/max(len(trips),1)),
                "link_pct": round(100*links/max(possible,1)),
                "weak_hops": weak_hops}

    before = score(base_matcher)

    # --- shadow: pretend the vehicle ALSO passes the new camera when it lies between
    # consecutive detections of its route (i.e. when the missed-camera gap crosses it)
    def simulate_with_hypo(trips):
        out = []
        for dets, route, skip in trips:
            d2 = list(dets)
            for i in range(len(dets) - 1):
                a, b = dets[i], dets[i + 1]
                if a.camera_id == b.camera_id:
                    continue
                if (_hop_distance(a.camera_id, b.camera_id) > 2
                        and a.camera_id in connected and b.camera_id in connected):
                    mid_ts = a.ts + (b.ts - a.ts) / 2
                    d2.append(M.Detection(
                        8100000 + a.det_id + i, new_cam_id, mid_ts,
                        a.plate_raw, a.ocr_conf, a.plate_correct,
                        a.vehicle_type, a.vehicle_color, None, None,
                        a.embed, a.session_id))
                    break
            d2.sort(key=lambda d: d.ts)
            out.append(d2)
        return out

    # build shadow matcher: same stats, plus synthesized ones for new edges
    stats = dict(base_matcher.stats)
    for cid in connected:
        d = network.haversine_m(lat, lng, *cam_by_id[cid][2:4]) * 1.35
        nominal = d / (40 / 3.6)
        est = M.EdgeStats(f"{new_cam_id}->{cid}", 1, nominal*0.7, nominal*1.2,
                          nominal*2.8, nominal*1.4, nominal*0.4, 30)
        stats[(est.edge_id, 1)] = est
        est2 = M.EdgeStats(f"{cid}->{new_cam_id}", 1, nominal*0.7, nominal*1.2,
                           nominal*2.8, nominal*1.4, nominal*0.4, 30)
        stats[(est2.edge_id, 1)] = est2
    shadow_matcher = M.Matcher({e.edge_id: e for e in shadow_edges}, stats,
                               base_matcher.appearance_margin,
                               base_matcher.appearance_floor,
                               base_matcher.hop_skip_multiplier)
    shadow_matcher.adj = {c: [] for c in list(network.adjacency_map()) + [new_cam_id]}
    for e in shadow_edges:
        shadow_matcher.adj.setdefault(e.src, []).append(e.dst)

    after_trips = simulate_with_hypo(trips)
    full = 0; links = 0; possible = 0; weak_hops = 0
    for dets in after_trips:
        traj = shadow_matcher.stitch_session(dets)
        n_h = len(traj.hops) if traj else 0
        links += n_h
        possible += len(dets) - 1
        if n_h == len(dets) - 1:
            full += 1
        if traj:
            weak_hops += sum(1 for h in traj.hops if h.confidence < 0.6)
    after = {"trials": len(after_trips), "full_pct": round(100*full/max(len(after_trips),1)),
             "link_pct": round(100*links/max(possible,1)), "weak_hops": weak_hops}

    # affected subset: trips whose 2-hop gap the new camera could actually bridge.
    # This is the planning-relevant number ("of the trips that pass this corridor…").
    affected = [i for i, trip in enumerate(trips) if trip_affected(trip)]
    aff_before_full = 0; aff_after_full = 0
    for i in affected:
        dets = trips[i][0]
        traj_b = base_matcher.stitch_session(dets)
        if traj_b and len(traj_b.hops) == len(dets) - 1:
            aff_before_full += 1
        traj_a = shadow_matcher.stitch_session(after_trips[i])
        if traj_a and len(traj_a.hops) == len(after_trips[i]) - 1:
            aff_after_full += 1
    affected_report = {
        "n_affected_trips": len(affected),
        "full_pct_before": round(100*aff_before_full/max(len(affected),1)) if affected else None,
        "full_pct_after": round(100*aff_after_full/max(len(affected),1)) if affected else None,
    }

    return {
        "hypothetical_camera": {"id": new_cam_id, "lat": lat, "lng": lng,
                                 "connects_to": connected},
        "benchmark": "2-adjacent-cameras-skipped trips (the hardest stitching case)",
        "before": before,
        "after": after,
        "affected_trips": affected_report,
        "delta_full_pct": after["full_pct"] - before["full_pct"],
        "delta_link_pct": after["link_pct"] - before["link_pct"],
    }
