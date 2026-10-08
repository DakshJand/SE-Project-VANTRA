"""Live replay endpoints: session create/control/tick."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import live

router = APIRouter(tags=["live"])


class SessionCreate(BaseModel):
    speed: float = 8.0
    minutes: int = 60


@router.post("/live/session")
async def create_session(body: SessionCreate):
    s = await live.create_session(speed=body.speed, minutes=body.minutes)
    return {"session_id": s.id, "t0": s.t0.isoformat(), "t1": s.t1.isoformat(),
            "speed": s.speed, "n_events": len(s.events)}


@router.get("/live/session/{sid}/tick")
async def tick(sid: str):
    s = live.get_session(sid)
    if not s:
        raise HTTPException(404, "session not found")
    return await live.tick(s)


@router.post("/live/session/{sid}/control")
async def control(sid: str, action: str, speed: float | None = None):
    s = live.get_session(sid)
    if not s:
        raise HTTPException(404, "session not found")
    if action == "pause":
        s.running = False
        s.paused_at = time_monotonic()
    elif action == "resume":
        if s.paused_at is not None:
            # shift start so pause doesn't count as elapsed
            s.started_wall += time_monotonic() - s.paused_at
            s.paused_at = None
        s.running = True
    elif action == "speed":
        if speed and speed > 0:
            old = s.sim_now()
            s.speed = speed
            s.started_wall = time_monotonic() - (old - s.t0).total_seconds() / speed
    else:
        raise HTTPException(400, "action must be pause|resume|speed")
    return {"session_id": s.id, "running": s.running, "speed": s.speed}


def time_monotonic() -> float:
    import time
    return time.monotonic()
