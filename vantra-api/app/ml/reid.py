"""Vehicle appearance embedding (Re-ID) for VANTRA.

A deterministic, explainable appearance descriptor: color histogram + type one-hot
+ coarse shape stats, L2-normalized. Cosine similarity over these embeddings is the
appearance signal used by the matcher (Component 2 tier 3/4) and shown as evidence.

This stands in for a CNN re-ID model (VeRi-776 trained) — see NOTES.md. The
interface (embed(vehicle_img) -> vec, cosine(a,b) -> float) matches what a real
model would provide, so swapping is one function.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

VEHICLE_TYPES = ["car", "suv", "truck", "bus", "auto"]
COLOR_NAMES = ["white", "black", "silver", "red", "blue", "grey", "green", "yellow"]
_COLOR_HSV_RANGES = {
    # hue on PIL's 0-255 scale (0=red). Tuples: (h_lo, h_hi), (s_lo, s_hi), (v_lo, v_hi)
    "red": ((0, 8), (110, 255), (60, 255)),
    "yellow": ((26, 48), (110, 255), (140, 255)),
    "green": ((70, 115), (90, 255), (60, 255)),
    "blue": ((130, 185), (90, 255), (60, 255)),
    "white": ((0, 255), (0, 30), (170, 255)),
    "black": ((0, 255), (0, 50), (0, 70)),
    "silver": ((0, 255), (0, 25), (100, 170)),
    "grey": ((0, 255), (0, 40), (70, 100)),
}


def classify_color(img: Image.Image) -> str:
    """Dominant color name via HSV range voting over the central body region.
    Colored votes count 3x (white/black/silver are background-prone)."""
    arr = np.asarray(img.convert("HSV"))
    h, s, v = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
    ch, cw = arr.shape[0] // 2, arr.shape[1] // 2
    h = h[ch - 50:ch + 50, cw - 110:cw + 110]
    s = s[ch - 50:ch + 50, cw - 110:cw + 110]
    v = v[ch - 50:ch + 50, cw - 110:cw + 110]
    votes: dict[str, int] = {}
    for name, (hr, sr, vr) in _COLOR_HSV_RANGES.items():
        mask = ((h >= hr[0]) & (h <= hr[1]) & (s >= sr[0]) & (s <= sr[1])
                & (v >= vr[0]) & (v <= vr[1]))
        votes[name] = int(mask.sum())
    if "red" in votes:  # hue wrap-around: 350-360deg
        votes["red"] += int(((h > 248) & (s > 110) & (v > 60)).sum())
    weight = {c: (3 if c in ("red", "yellow", "green", "blue") else 1) for c in votes}
    best = max(votes, key=lambda c: votes[c] * weight[c])
    return best if votes[best] > 50 else "grey"

def _hsv_hist(img: Image.Image, bins: int = 32) -> np.ndarray:
    arr = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    h = arr[..., 0]
    hist = np.histogram(h, bins=bins, range=(0.0, 1.0))[0].astype(np.float32)
    return hist / (hist.sum() + 1e-6)


def _sv_stats(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    return np.array([arr[..., 1].mean(), arr[..., 2].mean(), arr[..., 2].std()], dtype=np.float32)

def embed(img: Image.Image, vehicle_type: str | None = None) -> np.ndarray:
    """Appearance embedding: hue hist (32) + color-name hist (8) + type one-hot (5)
    + saturation/value stats (3). L2-normalized."""
    parts = [_hsv_hist(img)]
    cname = classify_color(img)
    color_hist = np.zeros(len(COLOR_NAMES), dtype=np.float32)
    color_hist[COLOR_NAMES.index(cname)] = 1.0
    parts.append(color_hist)
    t = np.zeros(len(VEHICLE_TYPES), dtype=np.float32)
    t[VEHICLE_TYPES.index(vehicle_type) if vehicle_type in VEHICLE_TYPES else 0] = 1.0
    parts.append(t)
    parts.append(_sv_stats(img))
    vec = np.concatenate(parts).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
