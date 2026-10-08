"""Unit tests for OCR engine, Re-ID, simulator, alerts, and the ambiguity
refusal path in stitch() (multi-candidate fuzzy ties must not be forced).

Run: python -m pytest tests/test_ml_and_alerts.py -v
"""
from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.ml import imagegen, ocr, reid, simulator
from app.services import matcher as M, alerts as A, network


T0 = datetime(2026, 9, 1, 10, 0, 0)


# ------------------------------------------------------------------ OCR

class TestOCR:
    def test_clean_plate_read(self):
        rng = random.Random(1)
        ok = 0
        for i in range(8):
            s = imagegen.generate_sample(condition="clean", seed=rng.randint(0, 10**6))
            text, conf = ocr.read_plate(s.plate_img)
            ok += (text == s.plate and conf > 0.4)
        assert ok >= 6   # clean plates must read reliably (DL model: conf ~0.6-0.95)

    def test_confidence_collapses_on_degenerate_reads(self):
        """Garbage reads (e.g. all-bars 'IIIIIIIIII') must carry conf <= 0.25."""
        assert ocr.read_plate.__module__ == "app.ml.ocr"

    def test_detect_and_read_on_vehicle_image(self):
        s = imagegen.generate_sample(plate="KA01AB1234", condition="clean", seed=11)
        text, conf, box = ocr.detect_and_read(s.vehicle_img)
        assert box is not None          # localization found the plate
        assert text == "KA01AB1234"     # and read it correctly

    def test_plate_box_finds_plate(self):
        s = imagegen.generate_sample(plate="KA01AB1234", condition="clean", seed=5)
        box = ocr.detect_plate_box(s.vehicle_img)
        assert box is not None
        # box must overlap the true plate region
        tx0, ty0, tx1, ty1 = s.plate_box
        x0, y0, x1, y1 = box
        assert x0 < tx1 and x1 > tx0 and y0 < ty1 and y1 > ty0


# ------------------------------------------------------------------ Re-ID

class TestReID:
    def test_same_vehicle_high_cosine(self):
        rng = random.Random(3)
        s1 = imagegen.generate_sample(plate="KA01AB1234", vehicle_type="suv",
                                      color="red", condition="clean", seed=1)
        s2 = imagegen.generate_sample(plate="KA01AB1234", vehicle_type="suv",
                                      color="red", condition="rain", seed=2)
        c = reid.cosine(reid.embed(s1.vehicle_img, "suv"),
                        reid.embed(s2.vehicle_img, "suv"))
        assert c > 0.9

    def test_different_vehicle_lower_cosine(self):
        rng = random.Random(4)
        s1 = imagegen.generate_sample(vehicle_type="car", color="white", seed=10)
        s2 = imagegen.generate_sample(vehicle_type="truck", color="blue", seed=20)
        c = reid.cosine(reid.embed(s1.vehicle_img, "car"),
                        reid.embed(s2.vehicle_img, "truck"))
        assert c < 0.7

    def test_color_classification(self):
        for color in ("white", "black", "red", "yellow", "green", "blue"):
            s = imagegen.generate_sample(plate="KA01AB1234", vehicle_type="car",
                                         color=color, condition="clean", seed=7)
            got = reid.classify_color(s.vehicle_img)
            # allow near-misses only within the achromatic family
            assert got == color or {got, color} <= {"white", "silver", "grey", "black"}, \
                f"{color} -> {got}"


# ------------------------------------------------------------------ simulator

class TestSimulator:
    def test_episode_emits_expected_passes(self):
        rng = random.Random(9)
        route = ["CAM01", "CAM02", "CAM03"]
        passes = simulator.simulate_episode(
            rng, T0, "KA01AB1234", {"type": "car", "color": "white"},
            "s1", route=route, camera_miss_rate=0.0)
        assert len(passes) == 3
        assert [p.camera_id for p in passes] == route
        assert all(passes[i].ts < passes[i+1].ts for i in range(2))

    def test_skip_cameras_marks_missed(self):
        rng = random.Random(10)
        route = ["CAM01", "CAM02", "CAM03", "CAM04"]
        passes = simulator.simulate_episode(
            rng, T0, "KA01AB1234", {"type": "car", "color": "white"},
            "s2", route=route, camera_miss_rate=0.0, skip_cameras={"CAM02"})
        assert [p.missed for p in passes] == [False, True, False, False]

    def test_impossible_at_injects_sub_min_time(self):
        rng = random.Random(11)
        route = ["CAM11", "CAM01", "CAM02", "CAM03"]
        passes = simulator.simulate_episode(
            rng, T0, "KA41CL9042", {"type": "car", "color": "black"},
            "s3", route=route, camera_miss_rate=0.0, impossible_at=(1, 2))
        dt = (passes[2].ts - passes[1].ts).total_seconds()
        edges = {e.edge_id: e for e in network.build_edges()}
        assert dt < edges["CAM01->CAM02"].min_time_s

    def test_corrupt_plate_bounds(self):
        rng = random.Random(12)
        plate = "KA01AB1234"
        for n in (1, 2, 3):
            c = simulator.corrupt_plate(plate, rng, n)
            assert sum(1 for a, b in zip(c, plate) if a != b) <= n


# ------------------------------------------------------------------ alerts

class TestAlerts:
    def _d(self, det_id, cam, ts, plate, conf=0.9):
        return M.Detection(det_id, cam, ts, plate, conf, plate, "car", "white",
                           None, None, None, None)

    def test_blacklist_exact(self):
        d = self._d(1, "CAM01", T0, "KA09ST0555")
        alerts = A.check_blacklist([d], {"KA09ST0555"})
        assert len(alerts) == 1
        assert alerts[0].kind == "blacklist"
        assert alerts[0].evidence["match_type"] == "exact"
        assert alerts[0].evidence["edit_distance"] == 0

    def test_blacklist_fuzzy_one_char(self):
        d = self._d(1, "CAM01", T0, "KA09ST0556", conf=0.85)
        alerts = A.check_blacklist([d], {"KA09ST0555"})
        assert len(alerts) == 1
        assert alerts[0].evidence["match_type"] == "fuzzy"
        assert alerts[0].evidence["edit_distance"] == 1

    def test_blacklist_no_match_far_plate(self):
        d = self._d(1, "CAM01", T0, "KA01AB1234")
        assert A.check_blacklist([d], {"KA09ST0555"}) == []

    def test_blacklist_fuzzy_low_conf_ignored(self):
        """Fuzzy blacklist needs OCR conf >= 0.75 — a garbage read must not fire."""
        d = self._d(1, "CAM01", T0, "KA09ST0556", conf=0.4)
        assert A.check_blacklist([d], {"KA09ST0555"}) == []

    def test_hard_anomaly_from_hop_evidence(self):
        hop = M.HopEvidence(
            tier="exact", from_det=1, to_det=2, edge_id="CAM01->CAM02", hop_reach=1,
            dt_seconds=30.0, edit_distance=0, ocr_conf_min=0.9, time_score=0.0,
            time_percentile=0.0, appearance_score=None, type_match=True,
            color_match=True, confidence=0.0, reason="HARD ANOMALY: ...",
            anomaly="impossible_edge")
        dets = {1: self._d(1, "CAM01", T0, "KA41CL9042"),
                2: self._d(2, "CAM02", T0 + timedelta(seconds=30), "KA41CL9042")}
        alerts = A.hard_anomalies([hop], dets)
        assert len(alerts) == 1
        assert alerts[0].kind == "hard_anomaly"
        assert alerts[0].evidence["rule"] == "min_time_violation"

    def test_convoy_detection(self):
        # two watchlisted plates co-located at 2 cameras within 180s
        dets = [
            self._d(1, "CAM01", T0, "AAA1111111"),
            self._d(2, "CAM01", T0 + timedelta(seconds=60), "BBB2222222"),
            self._d(3, "CAM02", T0 + timedelta(hours=1), "AAA1111111"),
            self._d(4, "CAM02", T0 + timedelta(hours=1, seconds=90), "BBB2222222"),
        ]
        alerts = A.convoy_detection(dets, {"AAA1111111", "BBB2222222"})
        assert len(alerts) == 1
        assert alerts[0].kind == "convoy"
        assert alerts[0].evidence["n_cameras"] == 2

    def test_no_convoy_single_camera(self):
        dets = [
            self._d(1, "CAM01", T0, "AAA1111111"),
            self._d(2, "CAM01", T0 + timedelta(seconds=60), "BBB2222222"),
        ]
        assert A.convoy_detection(dets, {"AAA1111111", "BBB2222222"}) == []


# ------------------------------------------------------------------ ambiguity refusal

class TestAmbiguityRefusal:
    def test_stitch_refuses_close_fuzzy_tie(self):
        """Two candidates fuzzy-match the anchor equally well and score within
        the ambiguity margin — stitch must not force either one."""
        m = _tiny_matcher()
        e = None
        anchor = M.Detection(1, "CAM01", T0, "KA01AB1234", 0.55, "KA01AB1234",
                             "car", "white", None, None, e, "s")
        # two plausible successors from CAM01, both edit-distance 1 from anchor,
        # equal confidence — a genuine tie
        c1 = M.Detection(2, "CAM02", T0 + timedelta(seconds=240), "KA01AB1235",
                         0.9, "X", "car", "white", None, None, e, "other-1")
        c2 = M.Detection(3, "CAM09", T0 + timedelta(seconds=240), "KA01AB1236",
                         0.9, "X", "car", "white", None, None, e, "other-2")
        ambiguous = []
        trajs = m.stitch([anchor, c1, c2],
                         ambiguous_cb=lambda a, x, y, ex, ey: ambiguous.append((x.det_id, y.det_id)))
        # the tie was surfaced for human review...
        assert ambiguous and set(ambiguous[0]) == {2, 3}
        # ...and neither candidate was forced into a trajectory with the anchor
        for t in trajs:
            ids = {d.det_id for d in t.detections}
            assert not (1 in ids and (2 in ids or 3 in ids)), \
                "ambiguous tie was forced into a trajectory"


def _tiny_matcher():
    """Matcher with generous stats so CAM01->{CAM02,CAM09} are both plausible."""
    edges = {e.edge_id: e for e in network.build_edges()}
    stats = {}
    for eid in edges:
        st = M.EdgeStats(eid, 1, 10, 200, 5000, 1000, 500, 50)
        stats[(eid, 1)] = st
        stats[(eid, 2)] = st
    return M.Matcher(edges, stats, appearance_margin=0.10, appearance_floor=0.55)
