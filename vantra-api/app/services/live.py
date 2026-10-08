"""Live replay engine: streams historical detections at real-world pace.

A replay session picks a chunk of the simulated history (one busy hour by default),
then exposes a virtual clock. The dashboard polls /live/session/{id}/tick, which
returns every event whose simulated timestamp has elapsed since the last tick —
detections, trajectory extensions (computed by the real matcher), and alerts
(blacklist hits during the window). All state is per-session in memory; nothing
touches the production tables. Speed multiplier compresses wall-clock time.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from app import db
from app.services import matcher as M
from app.services.reconstruct import build_matcher, det_from_row
from app.services import alerts as A
from app.services import network

_sessions: dict[str, "ReplaySession"] = {}
_next_id = [1]


@dataclass
class ReplaySession:
    id: str
    speed: float = 8.0
    running: bool = True
    started_wall: float = field(default_factory=time.monotonic)
    paused_at: float | None = None
    # replay window in simulated time
    t0: datetime = field(default_factory=datetime.utcnow)
    t1: datetime = field(default_factory=datetime.utcnow)
    # cursor: last simulated timestamp delivered
    cursor: datetime = field(default_factory=datetime.utcnow)
    # full event stream, sorted by ts
    events: list[dict] = field(default_factory=list)
    delivered: int = 0
    # live-computed state
    seen_dets: list[dict] = field(default_factory=list)
    alerts: list[dict] = field(default_factory=list)
    trajectories: list[list[int]] = field(default_factory=list)  # det-id chains

    def sim_now(self) -> datetime:
        """Current simulated time given wall-clock progress and speed."""
        if self.paused_at is not None:
            base = self.paused_at
        else:
            base = time.monotonic()
        elapsed_wall = base - self.started_wall
        return self.t0 + timedelta(seconds=elapsed_wall * self.speed)

    def wall_remaining(self) -> float:
        return max(0.0, (self.t1 - self.sim_now()).total_seconds() / self.speed)


def _cam_pos(cam_id: str) -> tuple[float, float]:
    for cid, _n, lat, lng, _r in network.CAMERAS:
        if cid == cam_id:
            return lat, lng
    return (0.0, 0.0)


async def create_session(speed: float = 8.0, minutes: int = 60,
                         session_hint: str | None = None) -> ReplaySession:
    """Pick the busiest hour in history and precompute the event stream."""
    # find the busiest 60-minute window in the dataset
    rows = await db.fetch("""
        SELECT ts FROM detections
        ORDER BY ts""")
    peak = datetime(2026, 8, 15, 9, 0)
    if rows:
        from collections import deque
        window: deque = deque()
        best_n, best_t = 0, rows[0]["ts"]
        for r in rows:
            window.append(r["ts"])
            while (r["ts"] - window[0]).total_seconds() > 3600:
                window.popleft()
            if len(window) > best_n:
                best_n, best_t = len(window), window[0]
        peak = best_t
    t0 = peak
    t1 = t0 + timedelta(minutes=minutes)

    dets = await db.fetch(
        """SELECT det_id, camera_id, ts, plate_raw, ocr_conf, plate_correct,
                  vehicle_type, vehicle_color, crop_path, plate_crop_path, session_id
           FROM detections WHERE ts >= $1 AND ts < $2 ORDER BY ts""", [t0, t1])

    events = []
    for d in dets:
        lat, lng = _cam_pos(d["camera_id"])
        events.append({
            "kind": "detection",
            "ts": d["ts"].isoformat(),
            "det_id": d["det_id"],
            "camera_id": d["camera_id"],
            "lat": lat, "lng": lng,
            "plate": d["plate_raw"],
            "ocr_conf": d["ocr_conf"],
            "vehicle_type": d["vehicle_type"],
            "vehicle_color": d["vehicle_color"],
        })

    sid = f"live-{_next_id[0]}"
    _next_id[0] += 1
    s = ReplaySession(id=sid, speed=speed, t0=t0, t1=t1, cursor=t0, events=events)
    _sessions[sid] = s
    return s


def get_session(sid: str) -> ReplaySession | None:
    return _sessions.get(sid)


async def tick(s: ReplaySession) -> dict:
    """Advance the session; return newly-elapsed events + live-computed state."""
    if not s.running:
        return {"running": False, "new_events": [], "alerts": [], "trajectories": []}

    now = s.sim_now()
    new_events = []
    while s.delivered < len(s.events):
        ev = s.events[s.delivered]
        ev_ts = datetime.fromisoformat(ev["ts"])
        if ev_ts <= now:
            new_events.append(ev)
            if ev["kind"] == "detection":
                s.seen_dets.append(ev)
            s.delivered += 1
        else:
            break

    # periodically (every tick with new data) run the matcher over recent dets
    new_alerts: list[dict] = []
    new_traj_links: list[dict] = []
    if new_events:
        m = await build_matcher()
        dets = [_ev_to_detection(d, i) for i, d in enumerate(s.seen_dets)]
        # link the newest detection back against the last 30 minutes of stream
        latest = dets[-1]
        for prev in reversed(dets[:-1]):
            if (latest.ts - prev.ts).total_seconds() > 1800:
                break
            ev = m.link(prev, latest)
            if ev and not ev.anomaly:
                new_traj_links.append({
                    "from_camera": prev.camera_id, "to_camera": latest.camera_id,
                    "plate": latest.plate_raw, "tier": ev.tier,
                    "confidence": ev.confidence,
                })
                break
        # blacklist + impossible-traversal alerts over the visible window
        bl_rows = await db.fetch("SELECT plate FROM blacklist")
        blacklist = {r["plate"] for r in bl_rows}
        fresh = [_ev_to_detection(d, i) for i, d in enumerate(s.seen_dets[-20:])]
        for a in A.check_blacklist(fresh, blacklist):
            new_alerts.append({
                "kind": a.kind, "severity": a.severity, "plate": a.plate,
                "title": a.title, "evidence": a.evidence,
            })
        # hard anomaly: pairwise impossible traversal on exact plates in stream
        by_plate: dict[str, list] = {}
        for d in fresh:
            if d.plate_raw:
                by_plate.setdefault(d.plate_raw, []).append(d)
        for plate, ds in by_plate.items():
            for i in range(len(ds) - 1):
                a, b = ds[i], ds[i + 1]
                if a.camera_id == b.camera_id:
                    continue
                ev = m.link(a, b)
                if ev and ev.anomaly == "impossible_edge":
                    new_alerts.append({
                        "kind": "hard_anomaly", "severity": "high", "plate": plate,
                        "title": (f"Impossible traversal {a.camera_id}->{b.camera_id} "
                                  f"in {ev.dt_seconds:.0f}s — likely cloned plate"),
                        "evidence": {"rule": "min_time_violation", "explanation": ev.reason},
                    })

    # dedupe alerts (same title once)
    seen_titles = {a["title"] for a in s.alerts}
    for a in new_alerts:
        if a["title"] not in seen_titles:
            seen_titles.add(a["title"])
            s.alerts.append(a)

    done = s.delivered >= len(s.events)
    if done:
        s.running = False
    return {
        "running": s.running and not done,
        "sim_time": now.isoformat(),
        "progress": s.delivered / max(len(s.events), 1),
        "new_events": new_events,
        "new_links": new_traj_links,
        "alerts": s.alerts[-10:],
        "n_detections": len(s.seen_dets),
        "done": done,
    }


def _ev_to_detection(ev: dict, idx: int) -> M.Detection:
    return M.Detection(
        det_id=ev.get("det_id", 9000000 + idx),
        camera_id=ev["camera_id"],
        ts=datetime.fromisoformat(ev["ts"]) if "ts" in ev and "T" in str(ev["ts"]) else datetime.now(timezone.utc),
        plate_raw=ev.get("plate"),
        ocr_conf=ev.get("ocr_conf"),
        plate_correct=None,
        vehicle_type=ev.get("vehicle_type"),
        vehicle_color=ev.get("vehicle_color"),
        crop_path=None, plate_crop_path=None,
        embed=None, session_id=None,
    )
