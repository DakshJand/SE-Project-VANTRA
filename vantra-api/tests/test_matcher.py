"""Unit tests for the VANTRA matcher — all confidence tiers, ambiguity refusal,
min_time hard anomaly, missed-camera window, appearance-only fallback.

These tests construct synthetic in-memory graphs/stats/detections and do NOT
touch the database. Run: python -m pytest tests/test_matcher.py -v
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.services import matcher as M
from app.services import network


# ------------------------------------------------------------------ fixtures

def make_stats(edge_id, p5, p50, p95, n=100):
    return M.EdgeStats(edge_id, 1, p5, p50, p95, (p5 + p95) / 2, (p95 - p5) / 4, n)


def build_test_matcher(**kwargs):
    """Matcher over the REAL camera network with fitted stats overridden for
    deterministic tests on the CAM01->CAM02 edge (min_time 111s, limit 40km/h)."""
    edges = {e.edge_id: e for e in network.build_edges()}
    stats = {}
    for e in edges.values():
        nominal = e.distance_m / (e.speed_limit_kmph / 3.6)
        stats[(e.edge_id, 1)] = make_stats(e.edge_id, nominal * 0.7, nominal * 1.2,
                                           nominal * 2.8)
    return M.Matcher(edges, stats, **kwargs)


def det(det_id, cam, ts, plate, conf=0.9, embed=None, vtype="car", color="white"):
    return M.Detection(det_id, cam, ts, plate, conf, plate, vtype, color,
                       None, None, embed, "test-session")


def embed_dir(direction: float) -> list[float]:
    """Distinct embeddings per direction: cosine(v_a, v_b) = cos(angle diff).
    direction=0 vs direction=pi/2 -> cosine 0."""
    import math
    v = [math.cos(direction), math.sin(direction)] + [0.0] * 46
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


T0 = datetime(2026, 9, 1, 10, 0, 0)


# ------------------------------------------------------------------ tiers

class TestExactTier:
    def test_exact_match_route_plausible(self):
        m = build_test_matcher()
        a = det(1, "CAM01", T0, "KA01AB1234")
        b = det(2, "CAM02", T0 + timedelta(seconds=250), "KA01AB1234")
        ev = m.link(a, b)
        assert ev is not None
        assert ev.tier == "exact"
        assert ev.edit_distance == 0
        assert ev.confidence > 0.8
        assert ev.hop_reach == 1
        assert "Exact plate match" in ev.reason

    def test_exact_not_route_plausible_refused(self):
        m = build_test_matcher()
        # CAM01 -> CAM15 is not adjacent nor 2-hop reachable
        a = det(1, "CAM01", T0, "KA01AB1234")
        b = det(2, "CAM15", T0 + timedelta(seconds=300), "KA01AB1234")
        assert m.link(a, b) is None

    def test_time_ordering_enforced(self):
        m = build_test_matcher()
        a = det(1, "CAM01", T0, "KA01AB1234")
        b = det(2, "CAM02", T0 - timedelta(seconds=250), "KA01AB1234")
        assert m.link(a, b) is None


class TestFuzzyTier:
    def test_fuzzy_within_threshold_links(self):
        m = build_test_matcher()
        a = det(1, "CAM01", T0, "KA01AB1234", conf=0.55)
        b = det(2, "CAM02", T0 + timedelta(seconds=250), "KA01AB1235", conf=0.9)
        ev = m.link(a, b)
        assert ev is not None
        assert ev.tier == "fuzzy_unique"
        assert ev.edit_distance == 1

    def test_fuzzy_threshold_widens_with_low_ocr_conf(self):
        m = build_test_matcher()
        a = det(1, "CAM01", T0, "KA01AB1234", conf=0.4)
        b = det(2, "CAM02", T0 + timedelta(seconds=250), "KA01XB1239", conf=0.9)
        ev = m.link(a, b)
        assert ev is not None
        assert ev.edit_distance == 2
        assert ev.confidence < 0.9   # fuzzy is always below exact

    def test_fuzzy_beyond_threshold_refused(self):
        m = build_test_matcher()
        a = det(1, "CAM01", T0, "KA01AB1234", conf=0.95)
        b = det(2, "CAM02", T0 + timedelta(seconds=250), "XX99ZZ9999", conf=0.95)
        ev = m.link(a, b)
        # high conf -> threshold 1; ed is huge -> no plate match; embeddings absent
        # -> no appearance fallback either
        assert ev is None


class TestAppearanceOnlyTier:
    def test_unreadable_plate_links_via_appearance(self):
        m = build_test_matcher()
        e = embed_dir(0.0)
        a = det(1, "CAM01", T0, None, conf=0.15, embed=e)
        b = det(2, "CAM02", T0 + timedelta(seconds=250), "KA01AB1234", conf=0.9, embed=e)
        ev = m.link(a, b)
        assert ev is not None
        assert ev.tier == "appearance_only"
        assert ev.appearance_score > 0.9
        assert ev.confidence <= 0.75   # capped at/under plate-matched tiers

    def test_appearance_below_floor_refused(self):
        m = build_test_matcher(appearance_floor=0.85)
        # orthogonal embeddings -> cosine ~0
        a = det(1, "CAM01", T0, None, conf=0.15, embed=embed_dir(0.0))
        b = det(2, "CAM02", T0 + timedelta(seconds=250), "KA01AB1234",
                conf=0.9, embed=embed_dir(1.5708))
        ev = m.link(a, b)
        assert ev is None

    def test_no_embedding_no_appearance_link(self):
        m = build_test_matcher()
        a = det(1, "CAM01", T0, None, conf=0.15, embed=None)
        b = det(2, "CAM02", T0 + timedelta(seconds=250), "KA01AB1234", conf=0.9, embed=None)
        assert m.link(a, b) is None

    def test_readable_plate_never_uses_appearance_tier(self):
        """Both readable + same embed but different plates -> refused, not
        downgraded to appearance-only."""
        m = build_test_matcher()
        e = embed_dir(0.0)
        a = det(1, "CAM01", T0, "KA01AB1234", conf=0.9, embed=e)
        b = det(2, "CAM02", T0 + timedelta(seconds=250), "KA05XX9999", conf=0.9, embed=e)
        assert m.link(a, b) is None


# ------------------------------------------------------------------ anomalies

class TestHardAnomaly:
    def test_min_time_violation_flags_impossible_edge(self):
        m = build_test_matcher()
        # CAM01->CAM02 min_time is 111s; traverse in 30s
        a = det(1, "CAM01", T0, "KA41CL9042")
        b = det(2, "CAM02", T0 + timedelta(seconds=30), "KA41CL9042")
        ev = m.link(a, b)
        assert ev is not None
        assert ev.anomaly == "impossible_edge"
        assert ev.confidence == 0.0
        assert "HARD ANOMALY" in ev.reason

    def test_min_time_violation_even_with_fuzzy_plate(self):
        m = build_test_matcher()
        a = det(1, "CAM01", T0, "KA41CL9042", conf=0.5)
        b = det(2, "CAM02", T0 + timedelta(seconds=40), "KA41CL9042", conf=0.9)
        ev = m.link(a, b)
        assert ev is not None
        assert ev.anomaly == "impossible_edge"

    def test_just_above_min_time_is_not_anomaly(self):
        m = build_test_matcher()
        # CAM01->CAM02 min_time 111s -> 200s is fine
        a = det(1, "CAM01", T0, "KA01AB1234")
        b = det(2, "CAM02", T0 + timedelta(seconds=200), "KA01AB1234")
        ev = m.link(a, b)
        assert ev is not None
        assert ev.anomaly is None


# ------------------------------------------------------------------ windows

class TestMissedCameraWindow:
    def test_two_hop_gap_still_links(self):
        """CAM01 -> CAM03 (CAM02 skipped): 2-hop window must link exact plates."""
        m = build_test_matcher()
        a = det(1, "CAM01", T0, "KA53MN1108")
        b = det(2, "CAM03", T0 + timedelta(seconds=600), "KA53MN1108")
        ev = m.link(a, b)
        assert ev is not None
        assert ev.hop_reach == 2
        assert ev.confidence < 0.99   # slightly discounted vs direct

    def test_three_hop_gap_refused(self):
        """CAM08 -> CAM11 is 3+ hops apart (verified in diagnostics) — refused."""
        m = build_test_matcher()
        a = det(1, "CAM08", T0, "KA01AB1234")
        b = det(2, "CAM11", T0 + timedelta(seconds=900), "KA01AB1234")
        assert m.link(a, b) is None

    def test_dwell_decay_never_zero(self):
        """Long dwell on a stop-eligible edge still scores > 0."""
        m = build_test_matcher()
        # CAM04->CAM05 is stop-eligible
        a = det(1, "CAM04", T0, "KA02PS3390")
        b = det(2, "CAM05", T0 + timedelta(seconds=3600), "KA02PS3390")
        ev = m.link(a, b)
        assert ev is not None
        assert ev.confidence > 0
        assert ev.time_score < 1.0   # decayed but not zero

    def test_confidence_decay_beyond_p95(self):
        m = build_test_matcher()
        a = det(1, "CAM01", T0, "KA01AB1234")
        b_fast = det(2, "CAM02", T0 + timedelta(seconds=250), "KA01AB1234")
        b_slow = det(3, "CAM02", T0 + timedelta(seconds=9000), "KA01AB1234")
        ev_fast = m.link(a, b_fast)
        ev_slow = m.link(a, b_slow)
        assert ev_fast.confidence > ev_slow.confidence
        assert ev_slow.confidence > 0


# ------------------------------------------------------------------ stitching

class TestStitching:
    def test_full_session_stitches(self):
        m = build_test_matcher()
        route = ["CAM01", "CAM02", "CAM03", "CAM04"]
        dets = []
        t = T0
        for i, cam in enumerate(route):
            if i:
                t = t + timedelta(seconds=240)
            dets.append(det(i + 1, cam, t, "KA01VX2026"))
        traj = m.stitch_session(dets)
        assert traj is not None
        assert len(traj.hops) == 3
        assert all(h.tier == "exact" for h in traj.hops)

    def test_broken_hop_keeps_detections(self):
        """A truly unlinkable gap doesn't drop detections — the trajectory keeps
        them; the hop is missing. CAM01 -> CAM15 is not adjacent/2-hop."""
        m = build_test_matcher()
        dets = [
            det(1, "CAM01", T0, "KA01VX2026"),
            det(2, "CAM02", T0 + timedelta(seconds=240), "KA01VX2026"),
            # teleport to a non-adjacent camera AND beyond any window:
            det(3, "CAM15", T0 + timedelta(seconds=240 + 30 * 3600), "KA01VX2026"),
        ]
        traj = m.stitch_session(dets)
        assert traj is not None
        assert len(traj.detections) == 3
        assert len(traj.hops) == 1   # only CAM01->CAM02 linked

    def test_rollup_confidence_dominated_by_weakest_hop(self):
        """A single weak hop must pull the rollup down (not average away)."""
        m = build_test_matcher()
        e = embed_dir(0.0)
        dets = [
            det(1, "CAM01", T0, "KA01AB1234", embed=e),
            det(2, "CAM02", T0 + timedelta(seconds=240), "KA01AB1234", embed=e),
            # unreadable plate at CAM03, linked by appearance (weakest hop)
            det(3, "CAM03", T0 + timedelta(seconds=480), None, conf=0.15, embed=e),
            det(4, "CAM04", T0 + timedelta(seconds=720), "KA01AB1234", embed=e),
        ]
        traj = m.stitch_session(dets)
        assert traj is not None and len(traj.hops) == 3
        weakest = min(h.confidence for h in traj.hops)
        strongest = max(h.confidence for h in traj.hops)
        mean = (weakest + strongest) / 2
        # rollup must sit closer to the weakest hop than to the arithmetic mean
        assert abs(traj.confidence - weakest) < abs(traj.confidence - mean)


# ------------------------------------------------------------------ helpers

class TestHelpers:
    def test_edit_distance(self):
        assert M.edit_distance("KA01AB1234", "KA01AB1234") == 0
        assert M.edit_distance("KA01AB1234", "KA01AB1235") == 1
        assert M.edit_distance("ABC", "XYZ") == 3

    def test_fuzzy_threshold_monotonic_in_confidence(self):
        # calibrated for EasyOCR confidence distribution
        assert M.fuzzy_threshold(0.9, 0.9) == 1
        assert M.fuzzy_threshold(0.6, 0.6) == 2
        assert M.fuzzy_threshold(0.3, 0.3) == 3
        # weaker side dominates
        assert M.fuzzy_threshold(0.95, 0.3) == 3

    def test_time_score_shapes(self):
        st = make_stats("X", 100, 200, 500)
        within, _, band = M.time_score(250, st, stop_eligible=False)
        assert within == 1.0 and band == "within_window"
        beyond, _, band2 = M.time_score(1500, st, stop_eligible=False)
        assert 0 < beyond < 1 and band2 == "beyond_p95"
        # stop-eligible decays slower at the same excess
        beyond_stop, _, _ = M.time_score(1500, st, stop_eligible=True)
        assert beyond_stop > beyond
        # extreme excess clamps at a non-zero floor on both curves
        floor_plain, _, _ = M.time_score(100000, st, stop_eligible=False)
        assert floor_plain >= 0.05
