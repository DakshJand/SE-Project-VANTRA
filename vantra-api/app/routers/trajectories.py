"""Trajectory endpoints: reconstruct, list, detail with evidence."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel
from app import db
from app.services import network, matcher as M
from app.services.casefile import build_casefile
from app.services.reconstruct import build_matcher, reconstruct_for_plate, det_from_row

router = APIRouter(tags=["trajectories"])


@router.get("/trajectories")
async def list_trajectories(plate: str | None = None, limit: int = 100):
    if plate:
        rows = await db.fetch(
            """SELECT t.* FROM trajectories t WHERE t.plate = $1
               ORDER BY t.t_start DESC LIMIT $2""", [plate, limit])
    else:
        rows = await db.fetch(
            "SELECT * FROM trajectories ORDER BY t_start DESC LIMIT $1", [limit])
    return rows


@router.get("/trajectories/reconstruct")
async def reconstruct(plate: str = Query(..., min_length=4),
                      session: str | None = None, request: Request = None):
    """On-the-fly trajectory reconstruction for a plate (used by demo + dashboard).
    Pass session=<id> to reconstruct a specific ground-truth trip (demo mode)."""
    from app.services import audit
    op = (request.headers.get("x-vantra-operator") if request else None) or "op-demo"
    await audit.record(op, "trajectory_lookup", f"plate={plate}" +
                       (f" session={session}" if session else ""), query=plate)
    trajs = await reconstruct_for_plate(plate, session)
    if not trajs:
        raise HTTPException(404, f"no detections found for plate {plate}")
    return [t for t in trajs]


@router.get("/trajectories/reconstruct/pdf")
async def reconstruct_pdf(plate: str = Query(..., min_length=4),
                           session: str | None = None):
    """Export the top reconstructed trajectory for a plate as a PDF case file."""
    trajs = await reconstruct_for_plate(plate, session)
    if not trajs:
        raise HTTPException(404, f"no detections found for plate {plate}")
    best = max(trajs, key=lambda t: t["confidence"])
    pdf = build_casefile(best)
    return Response(content=pdf, media_type="application/pdf",
                    headers={"Content-Disposition":
                             f'attachment; filename="vantra_case_{plate}.pdf"'})


class CombinedReportReq(BaseModel):
    plates: list[str]
    include_audit: bool = True


@router.post("/trajectories/case-report")
async def combined_case_report(body: CombinedReportReq):
    """Combined investigation case report: multiple plates' trajectories, convoy
    evidence, and audit-log entries in one PDF."""
    if not 2 <= len(body.plates) <= 6:
        raise HTTPException(400, "provide 2-6 plates")
    from app.services.casefile import build_combined_casefile
    pdf = await build_combined_casefile(body.plates, body.include_audit)
    return Response(content=pdf, media_type="application/pdf",
                    headers={"Content-Disposition":
                             'attachment; filename="vantra_combined_case.pdf"'})

@router.get("/trajectories/{traj_id}")
async def trajectory_detail(traj_id: int):
    traj = await db.fetch_one("SELECT * FROM trajectories WHERE traj_id = $1", [traj_id])
    if not traj:
        raise HTTPException(404, "trajectory not found")
    hops = await db.fetch("""
        SELECT h.*, da.camera_id AS from_cam, db.camera_id AS to_cam,
               da.ts AS from_ts, db.ts AS to_ts,
               da.plate_raw AS from_plate, db.plate_raw AS to_plate,
               da.crop_path AS from_crop, db.crop_path AS to_crop,
               da.plate_crop_path AS from_plate_crop, db.plate_crop_path AS to_plate_crop
        FROM trajectory_hops h
        JOIN detections da ON da.det_id = h.from_det_id
        JOIN detections db ON db.det_id = h.to_det_id
        WHERE h.traj_id = $1 ORDER BY h.seq""", [traj_id])
    return {"trajectory": traj, "hops": hops}
