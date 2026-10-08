"""Shared helpers: build a matcher from DB state, reconstruct trajectories on the fly."""
from __future__ import annotations

from app import db
from app.services import network, matcher as M


async def build_matcher() -> M.Matcher:
    """Build a matcher from the CURRENT database's camera graph + fitted stats.
    Both edges and stats are DB-driven so the same code drives the 18-camera
    demo network and the 120-camera scale network alike."""
    edge_rows = await db.fetch(
        "SELECT edge_id, src_cam, dst_cam, distance_m, speed_limit_kmph, stop_eligible, min_time_s FROM edges")
    edges = {}
    for r in edge_rows:
        eid = r["edge_id"]
        edges[eid] = network.Edge(eid, r["src_cam"], r["dst_cam"], r["distance_m"],
                                  r["speed_limit_kmph"], r["stop_eligible"], r["min_time_s"])
    rows = await db.fetch("SELECT * FROM edge_time_stats")
    stats = {(r["edge_id"], r["hop_reach"]): M.EdgeStats(
        r["edge_id"], r["hop_reach"], r["p5"], r["p50"], r["p95"],
        r["mean"], r["std"], r["n_samples"]) for r in rows}
    m = M.Matcher(edges, stats)
    # adjacency from the DB edge list (replaces the hardcoded demo graph)
    adj: dict[str, list[str]] = {}
    for e in edges.values():
        adj.setdefault(e.src, []).append(e.dst)
    m.adj = adj
    two_hop: dict[str, list[str]] = {c: [] for c in adj}
    for c, nbrs in adj.items():
        hop2 = set()
        for n in nbrs:
            hop2.update(adj.get(n, []))
        hop2.discard(c)
        hop2.difference_update(nbrs)
        two_hop[c] = sorted(hop2)
    m.two_hop = two_hop
    return m


def det_from_row(r: dict) -> M.Detection:
    return M.Detection(
        r["det_id"], r["camera_id"], r["ts"], r["plate_raw"], r["ocr_conf"],
        r["plate_correct"], r["vehicle_type"], r["vehicle_color"],
        r["crop_path"], r["plate_crop_path"],
        list(r["embed"]) if r.get("embed") else None, r["session_id"],
    )


async def fetch_detections_for_plate(plate: str) -> list[M.Detection]:
    """Exact + fuzzy-near plate reads (OCR errors mean the stored read may differ)."""
    rows = await db.fetch(
        """SELECT * FROM detections
           WHERE plate_raw = $1
              OR (plate_raw IS NOT NULL AND abs(length(plate_raw) - length($1)) <= 3
                  AND levenshtein(plate_raw, $1) <= 3)
           ORDER BY ts""", [plate])
    # levenshtein needs fuzzystrmatch; fall back to python filter if unavailable
    if not rows:
        rows = await db.fetch(
            "SELECT * FROM detections WHERE plate_raw = $1 ORDER BY ts", [plate])
    # appearance-only candidates: unreadable detections that are time-adjacent
    # (±45 min) to one of this plate's reads AND at a camera on/next to that
    # read's camera (only those are route-plausible) — NOT the whole unread pool.
    base_ids = [r["det_id"] for r in rows]
    if base_ids:
        idlist = ",".join(map(str, base_ids))
        # cameras of the plate's own reads + their graph neighbors
        cam_ids = sorted({r["camera_id"] for r in rows})
        from app.services.network import adjacency_map
        nbrs = set(cam_ids)
        _adj = adjacency_map()
        for c in cam_ids:
            nbrs.update(_adj.get(c, []))
        camlist = ",".join(f"'{c}'" for c in nbrs)
        unread = await db.fetch(
            f"""SELECT u.* FROM detections u
                WHERE (u.plate_raw IS NULL OR u.ocr_conf < 0.30)
                  AND u.det_id NOT IN ({idlist})
                  AND u.camera_id IN ({camlist})
                  AND EXISTS (
                    SELECT 1 FROM detections b
                    WHERE b.det_id IN ({idlist})
                      AND abs(EXTRACT(EPOCH FROM (u.ts - b.ts))) < 2700)
                ORDER BY u.ts""")
        return [det_from_row(r) for r in rows + unread]
    return [det_from_row(r) for r in rows]


async def reconstruct_for_plate(plate: str, session: str | None = None) -> list[dict]:
    """Reconstruct trajectories for a plate; returns API-shaped dicts with evidence.
    If `session` is given (demo mode), reconstruct that ground-truth trip exactly —
    the fetch then includes all of the session's reads (including corrupted ones)."""
    m = await build_matcher()
    if session:
        rows = await db.fetch(
            "SELECT * FROM detections WHERE session_id = $1 ORDER BY ts", [session])
        dets = [det_from_row(r) for r in rows]
    else:
        dets = await fetch_detections_for_plate(plate)
    if not dets:
        return []
    trajs = m.stitch_session(dets) if session else m.stitch(dets)
    if session and trajs is not None:
        trajs = [trajs]
    elif session:
        trajs = []
    out = []
    for t in trajs:
        det_rows = await db.fetch(
            f"""SELECT det_id, camera_id, ts, plate_raw, ocr_conf, vehicle_type,
                       vehicle_color, crop_path, plate_crop_path
                FROM detections WHERE det_id IN ({",".join(str(d.det_id) for d in t.detections)})
                ORDER BY ts""")
        cams = {c[0]: (c[2], c[3]) for c in network.CAMERAS}
        out.append({
            "plate": t.plate or plate,
            "confidence": t.confidence,
            "n_detections": len(t.detections),
            "t_start": t.detections[0].ts.isoformat(),
            "t_end": t.detections[-1].ts.isoformat(),
            "detections": [{
                "det_id": d["det_id"], "camera_id": d["camera_id"],
                "lat": cams[d["camera_id"]][0], "lng": cams[d["camera_id"]][1],
                "ts": d["ts"].isoformat(), "plate_raw": d["plate_raw"],
                "ocr_conf": d["ocr_conf"], "vehicle_type": d["vehicle_type"],
                "vehicle_color": d["vehicle_color"],
                "crop_url": f"/images/{_rel(d['crop_path'])}" if d["crop_path"] else None,
                "plate_crop_url": f"/images/{_rel(d['plate_crop_path'])}" if d["plate_crop_path"] else None,
            } for d in det_rows],
            "hops": [{
                "seq": i, "tier": h.tier, "confidence": h.confidence,
                "edge_id": h.edge_id, "hop_reach": h.hop_reach,
                "dt_seconds": h.dt_seconds, "edit_distance": h.edit_distance,
                "ocr_conf_min": h.ocr_conf_min, "time_score": h.time_score,
                "appearance_score": h.appearance_score,
                "type_match": h.type_match, "color_match": h.color_match,
                "reason": h.reason, "anomaly": h.anomaly,
                "from_det": h.from_det, "to_det": h.to_det,
            } for i, h in enumerate(t.hops)],
        })
    return out


def _rel(path: str | None) -> str:
    if not path:
        return ""
    # crop paths are absolute under data/images — make relative for /images mount
    idx = path.find("images/")
    return path[idx + len("images/"):] if idx >= 0 else path
