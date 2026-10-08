"""Video-based live detection: runs the REAL detection+OCR pipeline against
video frames (mp4), sampled at a configurable frame rate, feeding the same live
dashboard view as the JSON replay mode.

Frame extraction uses ffmpeg (piped, no temp files). The honest trade-off:
the full pipeline (localize + OCR variants + embedding) takes ~1-3s per frame,
so the effective processing rate is well below the clip's native 10fps — we
sample every Nth frame and report the real vs simulated rate in the session
status. JSON replay remains the default live mode.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings
from app.ml import ocr, reid

_VID_CANDIDATES = [
    settings.image_dir.parent / "videos",          # repo layout: <root>/data/videos
    Path("/app/data/videos"),                      # container layout
]
VID_DIR = next((d for d in _VID_CANDIDATES if d.exists()), _VID_CANDIDATES[0])

_video_sessions: dict[str, "VideoSession"] = {}
_next_id = [1]


@dataclass
class VideoSession:
    id: str
    video_path: Path
    camera_id: str
    sample_every: int = 10          # process every Nth frame
    running: bool = True
    frame_idx: int = 0              # next frame to process
    total_frames: int = 0
    fps: float = 10.0
    events: list[dict] = field(default_factory=list)
    sim_t0: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_wall: float = field(default_factory=time.monotonic)
    proc_seconds: float = 0.0      # cumulative pipeline time (for honest rate)
    frames_processed: int = 0


def probe(video_path: Path) -> tuple[int, float]:
    """Return (total_frames, fps) via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-count_frames", "-show_entries",
         "stream=nb_read_frames,avg_frame_rate", "-of", "csv=p=0",
         str(video_path)],
        capture_output=True, text=True, check=True)
    parts = out.stdout.strip().split(",")
    # ffprobe emits avg_frame_rate FIRST (e.g. "10/1,200")
    fps_s, frames_s = parts[0], parts[1] if len(parts) > 1 else "0"
    if "/" in fps_s:
        num, den = fps_s.split("/")
        fps = float(num) / float(den) if float(den) else 10.0
    else:
        fps = float(fps_s) or 10.0
    return int(frames_s), fps

def extract_frame(video_path: Path, idx: int) -> Image.Image | None:
    """Decode a single frame (seek + one-frame pipe)."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video_path),
         "-vf", f"select=eq(n\\,{idx})", "-fps_mode", "passthrough",
         "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True, check=False)
    if not out.stdout:
        return None
    import io
    return Image.open(io.BytesIO(out.stdout)).convert("RGB")


def create_session(video_name: str = "cam_traffic.mp4", camera_id: str = "CAM03",
                   sample_every: int = 10) -> VideoSession | None:
    path = VID_DIR / video_name
    if not path.exists():
        return None
    total, fps = probe(path)
    sid = f"video-{_next_id[0]}"
    _next_id[0] += 1
    s = VideoSession(id=sid, video_path=path, camera_id=camera_id,
                     sample_every=sample_every, total_frames=total, fps=fps)
    _video_sessions[sid] = s
    return s


def get_session(sid: str) -> VideoSession | None:
    return _video_sessions.get(sid)


def detect_and_read_tiled(frame: Image.Image, tiles: int = 4):
    """Detect+OCR a large video frame by scanning overlapping tiles at the scale
    the detector was tuned for (~420x300 vehicle crops). Returns the best read
    across tiles: (text, conf, global_box)."""
    W, H = frame.size
    best = ("", 0.0, None)
    tw, th = W // 2, H // 2
    for ty in range(tiles // 2):
        for tx in range(tiles // 2):
            x0, y0 = tx * (tw // 2), ty * (th // 2)   # 50% overlap
            tile = frame.crop((x0, y0, min(x0 + tw, W), min(y0 + th, H)))
            if tile.size[0] < 260 or tile.size[1] < 120:
                continue
            text, conf, box = ocr.detect_and_read(tile)
            if text and conf > best[1]:
                gbox = None
                if box:
                    gbox = (box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0)
                best = (text, conf, gbox)
    return best


def tick(s: VideoSession, max_frames: int = 1) -> dict:
    """Process the next sampled frame(s) through the real pipeline."""
    if not s.running:
        return {"running": False, "new_events": [], "done": True}

    new_events = []
    processed = 0
    while processed < max_frames and s.frame_idx < s.total_frames:
        t0 = time.perf_counter()
        frame = extract_frame(s.video_path, s.frame_idx)
        if frame is None:
            s.frame_idx += s.sample_every
            continue
        # REAL pipeline: tiled localize + OCR (tiles match the detector's tuned
        # scale for large video frames) + embedding
        text, conf, box = detect_and_read_tiled(frame)
        embed = reid.embed(frame, None)
        dt = time.perf_counter() - t0
        s.proc_seconds += dt
        s.frames_processed += 1
        sim_ts = s.sim_t0 + timedelta(seconds=s.frame_idx / s.fps)
        ev = {
            "kind": "detection",
            "ts": sim_ts.isoformat(),
            "camera_id": s.camera_id,
            "plate": text or None,
            "ocr_conf": round(conf, 3),
            "frame": s.frame_idx,
            "vehicle_type": None,
            "vehicle_color": reid.classify_color(frame),
            "box": box,
            "pipeline_ms": round(dt * 1000),
        }
        s.events.append(ev)
        new_events.append(ev)
        s.frame_idx += s.sample_every
        processed += 1

    done = s.frame_idx >= s.total_frames
    if done:
        s.running = False
    real_rate = s.frames_processed / s.proc_seconds if s.proc_seconds else 0
    return {
        "running": s.running and not done,
        "done": done,
        "frame": s.frame_idx,
        "total_frames": s.total_frames,
        "new_events": new_events,
        "frames_processed": s.frames_processed,
        "pipeline_ms_per_frame": round(s.proc_seconds / max(s.frames_processed, 1) * 1000),
        "effective_rate_fps": round(real_rate, 2),
        "native_fps": s.fps,
        "sample_every": s.sample_every,
        "n_detections": sum(1 for e in s.events if e["plate"]),
    }


def control(s: VideoSession, action: str, sample_every: int | None = None) -> None:
    if action == "pause":
        s.running = False
    elif action == "resume":
        s.running = True
    elif action == "rate" and sample_every and sample_every > 0:
        s.sample_every = sample_every
