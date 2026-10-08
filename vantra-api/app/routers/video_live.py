"""Video-based live detection endpoints (additional live mode; JSON replay stays default)."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import video_live

router = APIRouter(tags=["video-live"])


class VideoSessionReq(BaseModel):
    video: str = "cam_traffic.mp4"
    camera_id: str = "CAM03"
    sample_every: int = 10


@router.get("/video-live/clips")
async def list_clips():
    vids = video_live.VID_DIR.glob("*.mp4")
    return [{"name": v.name, "size_kb": v.stat().st_size // 1024} for v in vids]


@router.post("/video-live/session")
async def create_session(body: VideoSessionReq):
    s = video_live.create_session(body.video, body.camera_id, body.sample_every)
    if not s:
        raise HTTPException(404, f"video clip not found: {body.video}")
    return {"session_id": s.id, "total_frames": s.total_frames, "fps": s.fps,
            "sample_every": s.sample_every, "camera_id": s.camera_id}


@router.get("/video-live/session/{sid}/tick")
async def tick(sid: str):
    s = video_live.get_session(sid)
    if not s:
        raise HTTPException(404, "session not found")
    return video_live.tick(s)


@router.post("/video-live/session/{sid}/control")
async def control(sid: str, action: str, sample_every: int | None = None):
    s = video_live.get_session(sid)
    if not s:
        raise HTTPException(404, "session not found")
    if action not in ("pause", "resume", "rate"):
        raise HTTPException(400, "action must be pause|resume|rate")
    video_live.control(s, action, sample_every)
    return {"session_id": s.id, "running": s.running, "sample_every": s.sample_every}
