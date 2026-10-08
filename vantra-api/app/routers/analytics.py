"""Traffic analytics endpoints: heatmaps, segment speeds, route density."""
from fastapi import APIRouter, Query

from app import db

router = APIRouter(tags=["analytics"])


@router.get("/analytics/heatmap")
async def heatmap(hours_back: int = Query(168, ge=1, le=24 * 60)):
    """Detection density per camera (heatmap weights)."""
    rows = await db.fetch("""
        SELECT c.camera_id, c.lat, c.lng, count(d.det_id) AS weight
        FROM cameras c LEFT JOIN detections d
          ON d.camera_id = c.camera_id AND d.ts > now() - ($1 || ' hours')::interval
        GROUP BY c.camera_id, c.lat, c.lng ORDER BY c.camera_id""", [str(hours_back)])
    return rows


@router.get("/analytics/segment-speeds")
async def segment_speeds(hours_back: int = Query(168, ge=1, le=24 * 60)):
    """Average speed per edge from observed travel times of consecutive same-plate
    detections, plus learned p50 travel time."""
    rows = await db.fetch("""
        WITH pairs AS (
          SELECT a.camera_id AS src, b.camera_id AS dst,
                 EXTRACT(EPOCH FROM (b.ts - a.ts)) AS dt
          FROM detections a
          JOIN detections b
            ON b.plate_raw = a.plate_raw AND b.ts > a.ts
           AND b.ts - a.ts < interval '45 minutes'
           AND b.ts - a.ts > interval '1 minute'
          WHERE a.ts > now() - ($1 || ' hours')::interval AND a.plate_raw IS NOT NULL
        )
        SELECT e.edge_id, e.src_cam, e.dst_cam, e.distance_m, e.speed_limit_kmph,
               e.stop_eligible, e.tt_p50,
               count(p.dt) AS n_obs,
               CASE WHEN count(p.dt) >= 3 THEN e.distance_m / (avg(p.dt) / 3.6) ELSE NULL END
                 AS observed_speed_kmph
        FROM edges e
        LEFT JOIN pairs p ON p.src = e.src_cam AND p.dst = e.dst_cam
        GROUP BY e.edge_id ORDER BY e.edge_id""", [str(hours_back)])
    return rows


@router.get("/analytics/route-density")
async def route_density(hours_back: int = Query(168, ge=1, le=24 * 60)):
    """Traffic volume per camera per hour-of-day (route density matrix)."""
    rows = await db.fetch("""
        SELECT camera_id, EXTRACT(HOUR FROM ts) AS hour, count(*) AS n
        FROM detections
        WHERE ts > now() - ($1 || ' hours')::interval
        GROUP BY camera_id, hour ORDER BY camera_id, hour""", [str(hours_back)])
    return rows


@router.get("/analytics/summary")
async def summary():
    totals = await db.fetch_one("""
        SELECT (SELECT count(*) FROM detections) AS n_detections,
               (SELECT count(DISTINCT camera_id) FROM detections) AS n_cameras_active,
               (SELECT count(DISTINCT plate_raw) FROM detections WHERE plate_raw IS NOT NULL)
                 AS n_plates,
               (SELECT count(*) FROM trajectories) AS n_trajectories,
               (SELECT count(*) FROM alerts) AS n_alerts""")
    return totals
