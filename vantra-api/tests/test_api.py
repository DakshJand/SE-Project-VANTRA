"""Integration tests for the VANTRA API — in-process via FastAPI TestClient
against the seeded demo database. No external server needed; coverage works
naturally.

Run: python -m pytest tests/test_api.py -v
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os

os.environ.setdefault("VANTRA_RATE_LIMIT", "100000")   # disable limiter for tests

import pytest
from fastapi.testclient import TestClient

from app.main import app


ADMIN_KEY = {"x-vantra-admin-key": "vantra-admin"}


@pytest.fixture(scope="module")
def api():
    with TestClient(app) as client:
        yield client


# ------------------------------------------------------------------ basics

class TestHealth:
    def test_health(self, api):
        r = api.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_cameras_seeded(self, api):
        r = api.get("/api/cameras")
        assert r.status_code == 200
        cams = r.json()
        assert len(cams) >= 18
        assert all("lat" in c and "lng" in c for c in cams)


# ------------------------------------------------------------------ trajectory

class TestTrajectory:
    def test_reconstruct_demo1(self, api):
        r = api.get("/api/trajectories/reconstruct?plate=KA01VX2026&session=demo1-clean")
        assert r.status_code == 200
        t = r.json()[0]
        assert t["n_detections"] == 6
        assert len(t["hops"]) == 5
        assert all(h["tier"] == "exact" for h in t["hops"])
        assert t["confidence"] > 0.9
        # every hop carries machine-readable evidence
        for h in t["hops"]:
            assert h["reason"]
            assert h["edge_id"].startswith("CAM")

    def test_reconstruct_clone_flags_anomaly(self, api):
        r = api.get("/api/trajectories/reconstruct?plate=KA41CL9042&session=demo5-clone")
        assert r.status_code == 200
        hops = r.json()[0]["hops"]
        assert any(h["anomaly"] == "impossible_edge" for h in hops)
        assert r.json()[0]["confidence"] < 0.5

    def test_reconstruct_blur_hop_is_fuzzy(self, api):
        r = api.get("/api/trajectories/reconstruct?plate=KA05HR7741&session=demo2-blur")
        assert r.status_code == 200
        hops = r.json()[0]["hops"]
        assert any(h["tier"] == "fuzzy_unique" for h in hops)

    def test_reconstruct_missing_plate_404(self, api):
        r = api.get("/api/trajectories/reconstruct?plate=ZZ00ZZ0000")
        assert r.status_code == 404

    def test_reconstruct_short_plate_422(self, api):
        r = api.get("/api/trajectories/reconstruct?plate=KA")
        assert r.status_code == 422

    def test_pdf_export(self, api):
        r = api.get("/api/trajectories/reconstruct/pdf?plate=KA01VX2026&session=demo1-clean")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/pdf"
        assert r.content[:5] == b"%PDF-"


# ------------------------------------------------------------------ search

class TestSearch:
    def test_wildcard(self, api):
        r = api.get("/api/search?q=KA09ST*")
        assert r.status_code == 200
        plates = [m["plate"] for m in r.json()["matches"]]
        assert "KA09ST0555" in plates

    def test_exact(self, api):
        r = api.get("/api/search?q=KA01VX2026&fuzzy=false")
        assert r.status_code == 200
        assert any(m["plate"] == "KA01VX2026" for m in r.json()["matches"])

    def test_short_query_rejected(self, api):
        assert api.get("/api/search?q=K").status_code == 422


# ------------------------------------------------------------------ compare / whatif / tuning

class TestCompare:
    def test_compare_two_plates(self, api):
        r = api.post("/api/compare", json={"plates": ["KA01VX2026", "KA09ST0555"]})
        assert r.status_code == 200
        d = r.json()
        assert len(d["trajectories"]) == 2
        assert all(t["detections"] for t in d["trajectories"])

    def test_compare_validation(self, api):
        r = api.post("/api/compare", json={"plates": ["KA01VX2026"]})
        assert r.status_code == 400


class TestWhatIf:
    def test_candidates(self, api):
        r = api.get("/api/whatif/candidates")
        assert r.status_code == 200
        assert len(r.json()) >= 3

    def test_evaluate(self, api):
        r = api.post("/api/whatif/evaluate?lat=12.9708&lng=77.597")
        assert r.status_code == 200
        d = r.json()
        assert "before" in d and "after" in d
        assert "hypothetical_camera" in d

    def test_evaluate_far_location_error(self, api):
        r = api.post("/api/whatif/evaluate?lat=13.5&lng=78.5")
        assert r.status_code == 200
        assert "error" in r.json()


class TestTuning:
    def test_get_defaults(self, api):
        r = api.get("/api/tuning")
        assert r.status_code == 200
        eff = r.json()["effective"]
        assert eff["appearance_floor"] == 0.55

    def test_preview_is_dry_run(self, api):
        before = api.get("/api/tuning").json()["effective"]
        r = api.post("/api/tuning/preview", json={"fuzzy_ed_mid_conf": 3}, headers=ADMIN_KEY)
        assert r.status_code == 200
        d = r.json()
        assert "before" in d and "after" in d
        assert "nothing saved" in d["note"]
        # dry-run must not change stored settings
        after = api.get("/api/tuning").json()["effective"]
        assert after == before

    def test_save_and_reset_roundtrip(self, api):
        api.post("/api/tuning/save", json={"values": {"appearance_floor": 0.7}}, headers=ADMIN_KEY)
        eff = api.get("/api/tuning").json()["effective"]
        assert eff["appearance_floor"] == 0.7
        api.post("/api/tuning/reset", headers=ADMIN_KEY)
        eff2 = api.get("/api/tuning").json()["effective"]
        assert eff2["appearance_floor"] == 0.55

    def test_save_unknown_key_rejected(self, api):
        r = api.post("/api/tuning/save", json={"values": {"nonsense": 1}}, headers=ADMIN_KEY)
        assert r.status_code == 400


# ------------------------------------------------------------------ alerts / live

class TestAlerts:
    def test_blacklist_alerts_exist(self, api):
        r = api.get("/api/alerts?kind=blacklist&limit=5")
        assert r.status_code == 200
        alerts = r.json()
        assert len(alerts) >= 1
        assert alerts[0]["evidence"]["rule"].startswith("blacklist")

    def test_evaluate_dedupes(self, api):
        n1 = api.post("/api/alerts/evaluate", json={"since_hours": 720}).json()["generated"]
        n2 = api.post("/api/alerts/evaluate", json={"since_hours": 720}).json()["generated"]
        assert n2 <= n1
        assert n2 == 0 or n1 == 0  # second run adds nothing new


class TestLive:
    def test_session_lifecycle(self, api):
        r = api.post("/api/live/session", json={"speed": 240, "minutes": 20})
        assert r.status_code == 200
        sid = r.json()["session_id"]
        assert r.json()["n_events"] > 0
        saw_event = False
        for _ in range(30):
            t = api.get(f"/api/live/session/{sid}/tick").json()
            saw_event = saw_event or t["n_detections"] > 0
            if t.get("done"):
                break
            time.sleep(0.4)
        assert saw_event
        r = api.post(f"/api/live/session/{sid}/control?action=pause")
        assert r.status_code == 200
        r = api.post(f"/api/live/session/{sid}/control?action=speed&speed=480")
        assert r.status_code == 200

    def test_unknown_session_404(self, api):
        assert api.get("/api/live/session/nope/tick").status_code == 404

    def test_bad_control_action(self, api):
        r = api.post("/api/live/session/x/control?action=explode")
        assert r.status_code == 404  # unknown session fails first


# ------------------------------------------------------------------ audit / notify

class TestAudit:
    def test_records_and_lists(self, api):
        r = api.get("/api/audit")
        assert r.status_code == 200
        d = r.json()
        assert isinstance(d["entries"], list)
        assert isinstance(d["operators"], list)

    def test_filter_by_operator(self, api):
        api.get("/api/trajectories/reconstruct?plate=KA01VX2026",
                headers={"x-vantra-operator": "op-tester"})
        r = api.get("/api/audit?op=op-tester")
        entries = r.json()["entries"]
        assert any(e["operator"] == "op-tester" for e in entries)


class TestNotify:
    def test_config_roundtrip(self, api):
        r = api.post("/api/notify/config", headers=ADMIN_KEY, json={
            "webhook_url": "https://example.com/hook", "enabled": True})
        assert r.status_code == 200
        assert r.json()["webhook_url"] == "https://example.com/hook"
        r = api.post("/api/notify/test", headers=ADMIN_KEY)
        assert r.status_code == 200
        api.post("/api/notify/config", headers=ADMIN_KEY, json={"webhook_url": None, "enabled": True})


# ------------------------------------------------------------------ ingest

class TestIngest:
    def test_simulated_ingestion_runs_pipeline(self, api):
        r = api.post("/api/ingest/simulate", data={
            "camera_id": "CAM17", "plate": "KA77TE7777",
            "condition": "clean", "vehicle_type": "car", "color": "white"})
        assert r.status_code == 200
        d = r.json()
        assert d["detection"]["ground_truth_plate"] == "KA77TE7777"
        assert d["detection"]["plate_raw"]  # OCR read something
        # cleanup the test detection
        from app import db
        import asyncio
        asyncio.get_event_loop()
