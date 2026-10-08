"""Camera network + edges endpoints."""
from fastapi import APIRouter

from app import db

router = APIRouter(tags=["cameras"])


@router.get("/cameras")
async def list_cameras():
    rows = await db.fetch("SELECT camera_id, name, lat, lng, road FROM cameras ORDER BY camera_id")
    return rows


@router.get("/edges")
async def list_edges():
    rows = await db.fetch("""
        SELECT e.edge_id, e.src_cam, e.dst_cam, e.distance_m, e.speed_limit_kmph,
               e.stop_eligible, e.min_time_s, e.tt_p5, e.tt_p50, e.tt_p95,
               s.n_samples
        FROM edges e LEFT JOIN edge_time_stats s ON s.edge_id = e.edge_id AND s.hop_reach = 1
        ORDER BY e.edge_id
    """)
    return rows
