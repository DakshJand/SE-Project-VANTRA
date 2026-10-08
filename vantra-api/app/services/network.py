"""Synthetic city camera network for VANTRA.

18 camera nodes anchored to real Bengaluru coordinates (Indiranagar / Koramangala /
MG Road / CBD area). Adjacency graph approximates real road connectivity.
Everything here is SIMULATED — see NOTES.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# (camera_id, name, lat, lng, road)
CAMERAS: list[tuple[str, str, float, float, str]] = [
    ("CAM01", "MG Road / Trinity Junction",        12.9757, 77.6068, "MG Road"),
    ("CAM02", "Trinity Circle",                    12.9727, 77.6199, "Old Airport Rd start"),
    ("CAM03", "Domlur Flyover",                    12.9610, 77.6385, "Outer Ring Rd (Domlur)"),
    ("CAM04", "Indiranagar 100ft Rd (CMH Rd)",     12.9784, 77.6408, "100 Feet Rd"),
    ("CAM05", "Indiranagar KFC Junction",          12.9719, 77.6412, "80 Feet Rd / 100ft"),
    ("CAM06", "Koramangala Sony World Junction",   12.9351, 77.6240, "80 Feet Rd / ORR approach"),
    ("CAM07", "Ejipura Main Road",                 12.9502, 77.6296, "Ejipura Main Rd"),
    ("CAM08", "Sony World to MG Road via Airport", 12.9495, 77.6410, "Victoria Rd"),
    ("CAM09", "MG Road Metro Station",             12.9756, 77.6060, "MG Road"),
    ("CAM10", "Shivaji Nagar Bus Stand",           12.9849, 77.6050, "Shivaji Nagar"),
    ("CAM11", "Cubbon Road / Kasturba Rd",         12.9760, 77.5920, "Kasturba Rd"),
    ("CAM12", "Hudson Circle",                     12.9660, 77.5880, "Hudson Circle"),
    ("CAM13", "Richmond Circle",                   12.9608, 77.5956, "Richmond Rd"),
    ("CAM14", "Double Road (Lalbagh Rd)",          12.9500, 77.5850, "Lalbagh Rd"),
    ("CAM15", "Dairy Circle (Hosur Rd)",           12.9300, 77.5950, "Hosur Rd"),
    ("CAM16", "Bommanahalli (Hosur Rd)",           12.9000, 77.6180, "Hosur Rd"),
    ("CAM17", "Silk Board Junction",               12.9172, 77.6229, "ORR / Hosur Rd"),
    ("CAM18", "Ejipura Bore Bank Rd",              12.9455, 77.6380, "Bore Bank Rd"),
]

# Undirected road adjacency: (a, b, speed_limit_kmph, stop_eligible)
# stop_eligible edges are near markets/malls/parking/fuel (wider grace window).
ADJACENCY: list[tuple[str, str, int, bool]] = [
    ("CAM01", "CAM02", 40, False),
    ("CAM02", "CAM03", 50, False),
    ("CAM02", "CAM09", 30, False),
    ("CAM01", "CAM09", 30, False),
    ("CAM03", "CAM04", 40, False),
    ("CAM03", "CAM05", 40, False),
    ("CAM04", "CAM05", 30, True),    # 100ft Rd market stretch
    ("CAM05", "CAM06", 40, False),
    ("CAM05", "CAM07", 40, False),
    ("CAM06", "CAM07", 30, False),
    ("CAM06", "CAM08", 40, False),
    ("CAM08", "CAM18", 30, True),    # fuel station cluster
    ("CAM07", "CAM18", 30, False),
    ("CAM09", "CAM10", 30, False),
    ("CAM10", "CAM11", 40, False),
    ("CAM01", "CAM11", 40, False),
    ("CAM11", "CAM12", 40, False),
    ("CAM12", "CAM13", 40, False),
    ("CAM13", "CAM14", 40, False),
    ("CAM14", "CAM15", 40, False),
    ("CAM13", "CAM03", 50, False),
    ("CAM15", "CAM17", 50, False),
    ("CAM15", "CAM16", 50, False),
    ("CAM16", "CAM17", 50, False),
    ("CAM17", "CAM06", 50, False),
    ("CAM08", "CAM02", 50, False),
    ("CAM13", "CAM06", 40, False),
    ("CAM12", "CAM14", 40, False),
]


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


@dataclass
class Edge:
    edge_id: str
    src: str
    dst: str
    distance_m: float
    speed_limit_kmph: int
    stop_eligible: bool
    min_time_s: float

    @property
    def both(self) -> tuple[str, str]:
        return (self.src, self.dst)


def build_edges() -> list[Edge]:
    cam_by_id = {c[0]: c for c in CAMERAS}
    edges: list[Edge] = []
    for a, b, limit, stop in ADJACENCY:
        ca, cb = cam_by_id[a], cam_by_id[b]
        d = haversine_m(ca[2], ca[3], cb[2], cb[3])
        # road distance ~ 1.35x straight line for Indian urban roads
        d_road = round(d * 1.35)
        # hard floor: distance at 1.6x the speed limit (physically implausible below)
        min_t = d_road / (limit * 1.6) * 3.6
        edges.append(Edge(f"{a}->{b}", a, b, d_road, limit, stop, round(min_t, 1)))
        edges.append(Edge(f"{b}->{a}", b, a, d_road, limit, stop, round(min_t, 1)))
    return edges


def adjacency_map() -> dict[str, list[str]]:
    adj: dict[str, list[str]] = {c[0]: [] for c in CAMERAS}
    for a, b, _, _ in ADJACENCY:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def two_hop_pairs() -> dict[str, list[str]]:
    """cam -> cameras reachable in exactly 2 hops (through one skipped camera)."""
    adj = adjacency_map()
    out: dict[str, list[str]] = {}
    for cam, nbrs in adj.items():
        hop2: set[str] = set()
        for n in nbrs:
            hop2.update(adj[n])
        hop2.discard(cam)
        hop2.difference_update(nbrs)
        out[cam] = sorted(hop2)
    return out
