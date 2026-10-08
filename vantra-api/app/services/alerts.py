"""Alert engine — blacklist, hard/soft anomalies, convoy detection.

Every alert carries full machine-readable evidence (explainability principle).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from app.services import matcher as M
from app.services import network


@dataclass
class Alert:
    kind: str            # blacklist | hard_anomaly | soft_anomaly | convoy
    severity: str        # high | medium | low
    plate: str | None
    title: str
    ts: datetime
    det_ids: list[int]
    evidence: dict


def edit_distance(a: str, b: str) -> int:
    return M.edit_distance(a, b)


# ------------------------------------------------------------------ blacklist

def check_blacklist(dets: list[M.Detection], blacklist: set[str],
                    fuzzy_ed: int = 1) -> list[Alert]:
    """Real-time blacklist flag: exact or high-confidence fuzzy (edit distance <= 1
    with OCR conf >= 0.75) match on any detection."""
    alerts = []
    for d in dets:
        if not d.readable:
            continue
        for bl in blacklist:
            if d.plate_raw == bl:
                alerts.append(Alert(
                    "blacklist", "high", bl,
                    f"Blacklisted plate {bl} detected at {d.camera_id}",
                    d.ts, [d.det_id],
                    {"rule": "blacklist_exact", "plate_detected": d.plate_raw,
                     "blacklist_plate": bl, "edit_distance": 0,
                     "camera_id": d.camera_id, "ocr_conf": d.ocr_conf,
                     "match_type": "exact"}))
            elif edit_distance(d.plate_raw, bl) <= fuzzy_ed and (d.ocr_conf or 0) >= 0.50:
                alerts.append(Alert(
                    "blacklist", "high", bl,
                    f"Blacklisted plate {bl} (fuzzy match {d.plate_raw}) detected at {d.camera_id}",
                    d.ts, [d.det_id],
                    {"rule": "blacklist_fuzzy", "plate_detected": d.plate_raw,
                     "blacklist_plate": bl, "edit_distance": edit_distance(d.plate_raw, bl),
                     "camera_id": d.camera_id, "ocr_conf": d.ocr_conf,
                     "match_type": "fuzzy"}))
    return alerts


# ------------------------------------------------------------------ hard anomaly

def hard_anomalies(hops: list[M.HopEvidence], dets_by_id: dict[int, M.Detection]) -> list[Alert]:
    """Impossible edge traversals (min_time violations) from stitched trajectories."""
    alerts = []
    for h in hops:
        if h.anomaly != "impossible_edge":
            continue
        a, b = dets_by_id.get(h.from_det), dets_by_id.get(h.to_det)
        alerts.append(Alert(
            "hard_anomaly", "high", a.plate_raw if a else None,
            (f"Impossible traversal {a.camera_id}->{b.camera_id} in {h.dt_seconds:.0f}s "
             f"(plate {a.plate_raw}) — likely cloned/spoofed"),
            b.ts if b else None, [h.from_det, h.to_det],
            {"rule": "min_time_violation", "edge": h.edge_id,
             "dt_seconds": h.dt_seconds,
             "plate": a.plate_raw if a else None,
             "ocr_conf": h.ocr_conf_min,
             "explanation": h.reason}))
    return alerts


# ------------------------------------------------------------------ soft anomaly

def soft_anomalies(plate: str, dets: list[M.Detection], stats_by_edge: dict,
                   edges: dict[str, network.Edge],
                   min_history: int = 8) -> list[Alert]:
    """Statistically unusual route or time-of-day for a specific plate, judged
    against that plate's own historical pattern."""
    hist = [d for d in dets if d.plate_raw == plate]
    if len(hist) < min_history:
        return []
    # time-of-day distribution of this plate's appearances
    hours = [d.ts.hour + d.ts.minute / 60 for d in hist]
    n = len(hours)
    mean_h = sum(hours) / n
    var_h = sum((h - mean_h) ** 2 for h in hours) / max(n - 1, 1)
    # circular std (hours are cyclical)
    import math
    sin_sum = sum(math.sin(2 * math.pi * h / 24) for h in hours)
    cos_sum = sum(math.cos(2 * math.pi * h / 24) for h in hours)
    r = math.sqrt(sin_sum**2 + cos_sum**2) / n
    circ_std = math.sqrt(-2 * math.log(max(r, 1e-9))) * 24 / (2 * math.pi)
    mean_angle = math.atan2(sin_sum / n, cos_sum / n) * 24 / (2 * math.pi) % 24

    # route distribution: edges this plate has been seen on
    from collections import Counter
    cam_counter = Counter(d.camera_id for d in hist)

    alerts = []
    # evaluate the latest detections
    for d in hist[-5:]:
        # time-of-day z-score (circular distance)
        diff = abs(d.ts.hour + d.ts.minute / 60 - mean_angle)
        diff = min(diff, 24 - diff)
        if circ_std > 0.5 and diff > 3 * circ_std:
            alerts.append(Alert(
                "soft_anomaly", "medium", plate,
                f"Unusual time-of-day for {plate}: {d.ts.strftime('%H:%M')} "
                f"(usual center {mean_angle:.1f}h ± {circ_std:.1f}h)",
                d.ts, [d.det_id],
                {"rule": "plate_time_of_day_outlier", "plate": plate,
                 "observed_hour": d.ts.hour + d.ts.minute / 60,
                 "history_mean_hour": mean_angle, "history_circular_std": circ_std,
                 "deviation_hours": diff, "n_history": n,
                 "z_score": diff / circ_std if circ_std else None,
                 "camera_id": d.camera_id}))
        # route novelty: camera never seen before for this plate
        if cam_counter[d.camera_id] == 1 and n >= min_history * 2:
            alerts.append(Alert(
                "soft_anomaly", "low", plate,
                f"Unusual location for {plate}: first-ever detection at {d.camera_id}",
                d.ts, [d.det_id],
                {"rule": "plate_route_novelty", "plate": plate,
                 "camera_id": d.camera_id,
                 "known_cameras": sorted(set(x.camera_id for x in hist[:-1])),
                 "n_history": n}))
    return alerts


# ------------------------------------------------------------------ convoy

def convoy_detection(dets: list[M.Detection], watchlist: set[str],
                     window_s: int = 180, min_cameras: int = 2) -> list[Alert]:
    """Two or more watchlisted plates appearing at the same camera within a short
    window, repeatedly across multiple cameras."""
    watch_dets = sorted(
        [d for d in dets if d.readable and d.plate_raw in watchlist],
        key=lambda d: d.ts)
    # group by camera, look for close-in-time pairs
    from collections import defaultdict
    pairs: dict[tuple[str, str], list] = defaultdict(list)
    for i, a in enumerate(watch_dets):
        for b in watch_dets[i + 1:]:
            if (b.ts - a.ts).total_seconds() > window_s:
                break
            if a.camera_id == b.camera_id and a.plate_raw != b.plate_raw:
                key = tuple(sorted([a.plate_raw, b.plate_raw]))
                pairs[key].append({
                    "camera_id": a.camera_id,
                    "ts_a": a.ts.isoformat(), "ts_b": b.ts.isoformat(),
                    "gap_s": (b.ts - a.ts).total_seconds(),
                    "det_a": a.det_id, "det_b": b.det_id,
                })
    alerts = []
    for (p1, p2), meetings in pairs.items():
        cams = {m["camera_id"] for m in meetings}
        if len(cams) >= min_cameras:
            alerts.append(Alert(
                "convoy", "high", f"{p1}+{p2}",
                f"Convoy pattern: {p1} and {p2} co-located within {window_s}s "
                f"at {len(cams)} cameras",
                max(watch_dets, key=lambda d: d.ts).ts,
                [m["det_a"] for m in meetings] + [m["det_b"] for m in meetings],
                {"rule": "convoy_co_location", "plates": [p1, p2],
                 "window_s": window_s,
                 "cameras": sorted(cams),
                 "meetings": meetings,
                 "n_cameras": len(cams)}))
    return alerts
