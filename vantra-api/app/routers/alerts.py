"""Alert endpoints: list, review, blacklist management, live evaluation."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import db
from app.services import alerts as A
from app.services.reconstruct import build_matcher, det_from_row

router = APIRouter(tags=["alerts"])


@router.get("/alerts")
async def list_alerts(kind: str | None = None, limit: int = 100):
    if kind:
        return await db.fetch(
            "SELECT * FROM alerts WHERE kind = $1 ORDER BY ts DESC LIMIT $2",
            [kind, limit])
    return await db.fetch("SELECT * FROM alerts ORDER BY ts DESC LIMIT $1", [limit])


@router.get("/alerts/{alert_id}")
async def alert_detail(alert_id: int):
    row = await db.fetch_one("SELECT * FROM alerts WHERE alert_id = $1", [alert_id])
    if not row:
        raise HTTPException(404, "alert not found")
    return row


@router.post("/alerts/{alert_id}/review")
async def review_alert(alert_id: int):
    await db.execute("UPDATE alerts SET reviewed = TRUE WHERE alert_id = $1", [alert_id])
    return {"ok": True}


# ---------------------------------------------------------------- blacklist

class BlacklistAdd(BaseModel):
    plate: str
    reason: str


@router.get("/blacklist")
async def get_blacklist():
    return await db.fetch("SELECT * FROM blacklist ORDER BY added_at DESC")


@router.post("/blacklist")
async def add_blacklist(item: BlacklistAdd):
    await db.execute(
        """INSERT INTO blacklist (plate, reason) VALUES ($1, $2)
           ON CONFLICT (plate) DO UPDATE SET reason = EXCLUDED.reason""",
        [item.plate, item.reason])
    return {"ok": True, "plate": item.plate}


@router.delete("/blacklist/{plate}")
async def remove_blacklist(plate: str):
    await db.execute("DELETE FROM blacklist WHERE plate = $1", [plate])
    return {"ok": True}


# ---------------------------------------------------------------- live evaluation

@router.post("/alerts/evaluate")
async def evaluate_alerts(since_hours: float = 24.0):
    """Run blacklist + anomaly + convoy evaluation over recent detections.
    Returns the alerts generated (also persisted)."""
    m = await build_matcher()
    rows = await db.fetch(
        """SELECT * FROM detections WHERE ts > now() - ($1 || ' hours')::interval
           ORDER BY ts""", [str(since_hours)])
    dets = [det_from_row(r) for r in rows]

    bl_rows = await db.fetch("SELECT plate FROM blacklist")
    blacklist = {r["plate"] for r in bl_rows}

    generated: list[A.Alert] = []
    generated += A.check_blacklist(dets, blacklist)

    # stitch recent trajectories for hard anomalies
    trajs = m.stitch(dets)
    dets_by_id = {d.det_id: d for d in dets}
    for t in trajs:
        generated += A.hard_anomalies(t.hops, dets_by_id)

    generated += A.convoy_detection(dets, blacklist)

    # persist (dedupe: skip if an identical alert already exists)
    existing = await db.fetch("SELECT kind, title, ts FROM alerts")
    existing_keys = {(r["kind"], r["title"], r["ts"]) for r in existing}
    n_new = 0
    for a in generated:
        key = (a.kind, a.title, a.ts)
        if key in existing_keys:
            continue
        existing_keys.add(key)
        n_new += 1
        await db.execute(
            """INSERT INTO alerts (kind, severity, plate, title, ts, det_ids, evidence)
               VALUES ($1,$2,$3,$4,$5,$6,$7)""",
            [a.kind, a.severity, a.plate, a.title, a.ts,
             [int(x) for x in a.det_ids], a.evidence])
        # notification hook: blacklist hits and hard anomalies trigger the stub
        if a.kind in ("blacklist", "hard_anomaly"):
            from app.services import notify
            await notify.fire_notify(a.kind, a.title, a.evidence)
    return {"generated": n_new,
            "alerts": [{
                "kind": a.kind, "severity": a.severity, "plate": a.plate,
                "title": a.title, "ts": a.ts.isoformat(),
                "evidence": a.evidence,
            } for a in generated]}
