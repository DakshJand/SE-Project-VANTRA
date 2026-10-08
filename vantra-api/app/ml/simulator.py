"""Simulation of historical ANPR detection logs over the camera network.

Produces weeks of synthetic timestamped plate-passes:
- vehicles follow graph routes (random walks biased to plausible paths)
- travel times sampled per-edge around distance/speed with lognormal noise,
  peak-hour slowdowns, and dwell stops at stop-eligible edges
- injected noise: OCR errors (garbled plates proportional to image condition),
  missed detections (per-camera miss rates), unreadable plates

Ground truth is preserved (session_id, plate_correct, gt_cam) for evaluation.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.services import network
from app.ml.imagegen import generate_sample, save_sample

# ---------------------------------------------------------------- plates/vehicles

REPEAT_VEHICLES = 120          # vehicles with daily commute patterns (for soft anomaly history)
EPISODE_VEHICLES = 400         # one-off trips


def make_vehicle(rng: random.Random, idx: int) -> dict:
    return {
        "plate": None,  # assigned lazily
        "type": rng.choices(
            ["car", "suv", "truck", "bus", "auto"],
            weights=[0.45, 0.25, 0.10, 0.08, 0.12])[0],
        "color": rng.choice(["white", "black", "silver", "red", "blue", "grey", "green", "yellow"]),
    }


# ---------------------------------------------------------------- OCR error model

# Character confusion model for OCR noise: visually similar chars swap more often.
CONFUSABLE = {
    "B": "8R", "8": "B", "S": "5", "5": "S", "Z": "2", "2": "Z",
    "G": "6", "6": "G", "O": "0", "0": "O", "D": "0", "Q": "0",
    "I": "1", "1": "I", "T": "7", "7": "T", "A": "4", "4": "A",
    "U": "V", "V": "U", "E": "F", "K": "X",
}


def corrupt_plate(plate: str, rng: random.Random, n_errors: int) -> str:
    chars = list(plate)
    idxs = rng.sample(range(len(chars)), min(n_errors, len(chars)))
    for i in idxs:
        if chars[i] in CONFUSABLE and rng.random() < 0.7:
            chars[i] = rng.choice(CONFUSABLE[chars[i]])
        else:
            chars[i] = rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    return "".join(chars)


# ---------------------------------------------------------------- simulation core

PEAK_HOURS = [(8, 11), (17, 21)]


def speed_factor(hour: int, rng: random.Random) -> float:
    """1.0 = free flow. Peak hours slow everything down."""
    f = 1.0
    for lo, hi in PEAK_HOURS:
        if lo <= hour < hi:
            f *= rng.uniform(1.4, 2.2)
    if 22 <= hour or hour < 6:
        f *= rng.uniform(0.85, 1.0)   # night: slightly faster
    return f


@dataclass
class Pass:
    camera_id: str
    ts: datetime
    plate: str
    vehicle: dict
    session_id: str
    condition: str            # image condition: clean | stress-*
    ocr_plate: str | None     # what OCR saw (None = unreadable)
    ocr_conf: float
    missed: bool = False      # camera failed to record (no detection row emitted)


def _sample_condition(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.70:
        return "clean"
    return rng.choice(["blur", "angle", "lowlight", "occlusion", "rain"])


def _ocr_result(plate: str, condition: str, rng: random.Random) -> tuple[str | None, float]:
    """Ground-truth model of how well OCR would read a plate in a condition."""
    # Confidence distribution calibrated to the EasyOCR (DL) model: clean reads
    # 0.6-0.95, stress-correct 0.5-0.85, misreads 0.2-0.6, unreadable <= 0.3.
    if condition == "clean":
        if rng.random() < 0.02:  # 2% clean misread
            return corrupt_plate(plate, rng, 1), rng.uniform(0.50, 0.70)
        return plate, rng.uniform(0.70, 0.95)
    # stress: worse confidence, more errors
    p_unread = {"blur": 0.06, "angle": 0.05, "lowlight": 0.10,
                "occlusion": 0.14, "rain": 0.08}[condition]
    if rng.random() < p_unread:
        return None, rng.uniform(0.05, 0.30)
    p_err = {"blur": 0.35, "angle": 0.30, "lowlight": 0.40,
             "occlusion": 0.45, "rain": 0.30}[condition]
    if rng.random() < p_err:
        n = 1 if rng.random() < 0.75 else 2
        return corrupt_plate(plate, rng, n), rng.uniform(0.30, 0.60)
    return plate, rng.uniform(0.55, 0.85)


def _travel_time(edge: network.Edge, hour: int, rng: random.Random,
                 stop_eligible_dwell: bool) -> float:
    """Sample a realistic traversal time (seconds) for an edge."""
    nominal = edge.distance_m / (edge.speed_limit_kmph / 3.6)
    factor = speed_factor(hour, rng)
    # lognormal congestion noise
    t = nominal * factor * rng.lognormvariate(0, 0.18)
    if stop_eligible_dwell and edge.stop_eligible and rng.random() < 0.22:
        t += rng.uniform(300, 1500)  # dwell at parking/market/fuel
    return t


def random_walk_route(rng: random.Random, min_len: int = 3, max_len: int = 7) -> list[str]:
    adj = network.adjacency_map()
    cams = list(adj)
    route = [rng.choice(cams)]
    while len(route) < rng.randint(min_len, max_len):
        prev = route[-2] if len(route) >= 2 else None
        nbrs = [c for c in adj[route[-1]] if c != prev or len(adj[c]) == 1]
        if not nbrs:
            break
        route.append(rng.choice(nbrs))
    return route


def simulate_episode(rng: random.Random, start: datetime, plate: str, vehicle: dict,
                     session_id: str, route: list[str] | None = None,
                     camera_miss_rate: float = 0.06,
                     stop_dwell: bool = True,
                     force_conditions: dict[int, str] | None = None,
                     skip_cameras: set[str] | None = None,
                     impossible_at: tuple[int, int] | None = None) -> list[Pass]:
    """One vehicle trip over `route`. Returns the emitted Passes (missed ones flagged).

    force_conditions: {route_index: condition} to force a specific image condition.
    skip_cameras: cameras that fail to detect this vehicle entirely (simulated miss).
    impossible_at: (i, i+1) route indices — plate cloned: car appears at route[i] and
        route[i+1] within < min_time (hard anomaly injection).
    """
    route = route or random_walk_route(rng)
    skip_cameras = skip_cameras or set()
    edges = {e.edge_id: e for e in network.build_edges()}
    passes: list[Pass] = []
    t = start
    for i, cam in enumerate(route):
        if i > 0:
            prev = route[i - 1]
            edge = edges[f"{prev}->{cam}"]
            if impossible_at and (i - 1, i) == impossible_at:
                # spoof: traverse in a fraction of min_time
                t = t + timedelta(seconds=edge.min_time_s * rng.uniform(0.25, 0.5))
            else:
                tt = _travel_time(edge, t.hour, rng, stop_dwell)
                t = t + timedelta(seconds=tt)
        cond = (force_conditions or {}).get(i, _sample_condition(rng))
        missed = cam in skip_cameras or rng.random() < camera_miss_rate
        ocr_plate, ocr_conf = _ocr_result(plate, cond, rng)
        passes.append(Pass(cam, t, plate, vehicle, session_id, cond, ocr_plate, ocr_conf, missed))
    return passes
