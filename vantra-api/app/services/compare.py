"""Multi-vehicle comparison: overlaid trajectories + convoy evidence."""
from __future__ import annotations

from app import db
from app.services import network
from app.services.reconstruct import build_matcher
from app.services import alerts as A
from app.services.matcher import Detection as Det, edit_distance


async def compare_plates(plates: list[str]) -> dict:
    """Reconstruct each plate's best trajectory, overlay them on a shared timeline,
    and compute co-location/convoy evidence between every pair."""
    m = await build_matcher()
    cams = {c[0]: (c[2], c[3]) for c in network.CAMERAS}
    trajs = []
    for plate in plates:
        rows = await db.fetch(
            "SELECT * FROM detections WHERE plate_raw = $1 ORDER BY ts", [plate])
        if not rows:
            continue
        dets = [Det(r["det_id"], r["camera_id"], r["ts"], r["plate_raw"], r["ocr_conf"],
                    r["plate_correct"], r["vehicle_type"], r["vehicle_color"],
                    r["crop_path"], r["plate_crop_path"],
                    list(r["embed"]) if r.get("embed") else None, r["session_id"])
                for r in rows]
        stitched = m.stitch(dets)
        best = max(stitched, key=lambda t: len(t.detections)) if stitched else None
        det_list = best.detections if best else dets
        trajs.append({
            "plate": plate,
            "confidence": best.confidence if best else None,
            "detections": [{
                "camera_id": d.camera_id, "lat": cams[d.camera_id][0],
                "lng": cams[d.camera_id][1], "ts": d.ts.isoformat(),
                "plate": d.plate_raw,
            } for d in det_list],
        })

    # pairwise convoy/co-location evidence
    watch = set(plates)
    all_dets = []
    for t in trajs:
        for d in t["detections"]:
            all_dets.append(Det(0, d["camera_id"], __import__("datetime").datetime.fromisoformat(d["ts"]),
                                d["plate"], 0.9, d["plate"], None, None, None, None, None, None))
    convoys = []
    for i in range(len(plates)):
        for j in range(i + 1, len(plates)):
            p1, p2 = plates[i], plates[j]
            # co-location meetings: same camera within 3 min
            d1 = [d for d in all_dets if d.plate_raw == p1]
            d2 = [d for d in all_dets if d.plate_raw == p2]
            meetings = []
            for a in d1:
                for b in d2:
                    if a.camera_id == b.camera_id:
                        gap = abs((a.ts - b.ts).total_seconds())
                        if gap <= 180:
                            meetings.append({
                                "camera_id": a.camera_id,
                                "ts": a.ts.isoformat(),
                                "gap_s": gap,
                                "t1": a.ts.isoformat(), "t2": b.ts.isoformat(),
                            })
            cams_shared = sorted({mt["camera_id"] for mt in meetings})
            if meetings:
                convoys.append({
                    "plates": [p1, p2],
                    "is_convoy": len(cams_shared) >= 2,
                    "cameras": cams_shared,
                    "n_meetings": len(meetings),
                    "meetings": meetings[:10],
                    "explanation": (
                        f"Co-located at {len(cams_shared)} camera(s) within 3 minutes: "
                        f"{', '.join(cams_shared)}. "
                        + ("Meets convoy criteria (≥2 cameras)." if len(cams_shared) >= 2
                           else "Below convoy threshold (needs ≥2 cameras).")),
                })
    return {"trajectories": trajs, "convoys": convoys}
