"""VANTRA API — FastAPI application."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routers import cameras, search, trajectories, analytics, alerts, ingest, demo, live, features, video_live

app = FastAPI(title="VANTRA API", version="1.0.0",
              description="City-wide multi-camera ANPR trajectory tracking & analytics")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.security import SecurityMiddleware
app.add_middleware(SecurityMiddleware)

app.include_router(cameras.router, prefix="/api")
app.include_router(search.router, prefix="/api")
app.include_router(trajectories.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")
app.include_router(alerts.router, prefix="/api")
app.include_router(ingest.router, prefix="/api")
app.include_router(demo.router, prefix="/api")
app.include_router(live.router, prefix="/api")
app.include_router(features.router, prefix="/api")
app.include_router(video_live.router, prefix="/api")
# serve detection images
IMG_ROOT = Path(settings.image_dir).parent
settings.image_dir.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(settings.image_dir)), name="images")


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "vantra-api"}
