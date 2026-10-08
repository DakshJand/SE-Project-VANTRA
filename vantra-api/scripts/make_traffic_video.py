"""Generate synthetic traffic-camera video clips for the video live mode.

Renders vehicles (with plates) moving through the frame at a chosen camera, then
encodes to mp4 with ffmpeg. The video live mode decodes these clips frame by
frame and runs the REAL detection+OCR pipeline on each sampled frame — the only
synthetic part is the footage itself (no real dashcam footage is bundled with the
repo; the pipeline is video-format-agnostic mp4/H.264).

Run: python scripts/make_traffic_video.py  (writes data/videos/cam03_traffic.mp4)
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw

from app.config import settings
from app.ml import imagegen

W, H = 1600, 1200
FPS = 10
SECONDS = 20
N_VEHICLES = 5
SPRITE_W = 420                 # native size — the detector is tuned to it


def make_clip(path: Path, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    import random
    pyrng = random.Random(seed)

    # pre-render vehicle sprites (clean + a couple of degraded ones) at near-native
    # size — downscaling destroys the plate glyphs
    vehicles = []
    for i in range(N_VEHICLES):
        plate = imagegen.random_plate(pyrng)
        cond = ["clean", "clean", "clean", "blur", "lowlight"][i % 5]
        s = imagegen.generate_sample(plate=plate, vehicle_type=pyrng.choice(["car", "suv", "truck"]),
                                     color=pyrng.choice(["white", "red", "black", "silver"]),
                                     condition=cond, seed=pyrng.randint(0, 10**6))
        sprite = s.vehicle_img.resize((SPRITE_W, int(SPRITE_W * s.vehicle_img.height / s.vehicle_img.width)))
        vehicles.append({
            "sprite": sprite,
            "plate": plate,
            "x": -460.0 + i * 340.0,
            "speed": rng.uniform(4.5, 7.5),       # px/frame
            "lane_y": 150 + (i % 3) * 330,
        })

    frames_dir = path.parent / "_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    n_frames = FPS * SECONDS

    for f in range(n_frames):
        img = Image.new("RGB", (W, H), (100, 110, 120))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W, 40], fill=(150, 165, 180))
        d.rectangle([0, H - 80, W, H], fill=(80, 82, 85))
        for y in range(60, H, 60):
            d.line([0, y, W, y], fill=(120, 122, 125), width=2)
        d.line([0, H - 40, W, H - 40], fill=(210, 210, 90), width=3)
        for v in vehicles:
            x = v["x"]
            if x > W + 50:            # wrap around
                v["x"] = x = -220.0
                v["speed"] = rng.uniform(4.5, 7.5)
            img.paste(v["sprite"], (int(x), int(v["lane_y"])))
            v["x"] = x + v["speed"]
        img.save(frames_dir / f"f{f:04d}.png")

    # encode
    out = path
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", str(frames_dir / "f%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        str(out),
    ], check=True, capture_output=True)
    # cleanup frames
    for p in frames_dir.glob("*.png"):
        p.unlink()
    frames_dir.rmdir()
    print(f"clip written: {out} ({out.stat().st_size // 1024} KB, "
          f"{n_frames} frames @ {FPS}fps)")


if __name__ == "__main__":
    vid_dir = settings.image_dir.parent / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)
    make_clip(vid_dir / "cam_traffic.mp4")
