"""Threshold tuning: dry-run preview + explicit save.

The production matcher thresholds live in Matcher instances built by
build_matcher(). This service: (1) shows current effective values, (2) re-runs
matching over a historical sample with PROPOSED values and reports the delta
(matches by tier, hard anomalies) without touching anything, (3) on explicit
save, persists overrides to a settings table used by build_matcher.
"""
from __future__ import annotations

from app import db
from app.services import network, matcher as M
from app.services.reconstruct import det_from_row

DEFAULTS = {
    # fuzzy_threshold pivot: min OCR conf mapping -> max edit distance accepted
    "fuzzy_ed_high_conf": 1,     # conf >= 0.88 -> this max ed
    "fuzzy_ed_mid_conf": 2,      # conf >= 0.62
    "fuzzy_ed_low_conf": 3,      # below
    "appearance_floor": 0.55,    # min cosine to link appearance-only hops
    "appearance_margin": 0.06,   # ambiguity gap (top-2 too close -> refuse)
    "min_time_factor": 1.0,      # scales the hard floor (strictness)
}

_schema_done = False


async def ensure_schema() -> None:
    global _schema_done
    if _schema_done:
        return
    await db.execute("""CREATE TABLE IF NOT EXISTS matcher_settings (
        key TEXT PRIMARY KEY, value DOUBLE PRECISION NOT NULL)""")
    _schema_done = True


async def get_overrides() -> dict:
    await ensure_schema()
    rows = await db.fetch("SELECT key, value FROM matcher_settings")
    return {r["key"]: r["value"] for r in rows}


async def get_effective() -> dict:
    ov = await get_overrides()
    return {**DEFAULTS, **ov}


async def save_overrides(values: dict) -> None:
    await ensure_schema()
    for k, v in values.items():
        if k not in DEFAULTS:
            raise ValueError(f"unknown setting {k}")
        await db.execute(
            """INSERT INTO matcher_settings (key, value) VALUES ($1, $2)
               ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value""", [k, float(v)])


async def reset_overrides() -> None:
    await ensure_schema()
    await db.execute("DELETE FROM matcher_settings")



def make_threshold_fn(effective: dict):
    def fuzzy_threshold(a, b, _max_gap=None):
        conf = min(a or 0.5, b or 0.5)
        if conf >= 0.88:
            return int(effective["fuzzy_ed_high_conf"])
        if conf >= 0.62:
            return int(effective["fuzzy_ed_mid_conf"])
        return int(effective["fuzzy_ed_low_conf"])
    return fuzzy_threshold


async def preview(values: dict, n_sessions: int = 250) -> dict:
    """Dry-run: re-stitch a sample of historical trips with proposed thresholds.
    Returns before/after counts. Nothing is persisted."""
    await ensure_schema()
    current = await get_effective()
    proposed = {**current, **{k: float(v) for k, v in values.items() if k in DEFAULTS}}

    # load a sample of ground-truth sessions (mixed quality)
    # sample ground-truth sessions, biased toward sessions containing misread plates
    # (that's where fuzzy thresholds actually bite)
    rows = await db.fetch("""
        SELECT session_id, count(*) FILTER (WHERE plate_raw <> plate_correct) AS n_mis
        FROM detections
        WHERE session_id LIKE 'hist-%' AND plate_raw IS NOT NULL
        GROUP BY session_id HAVING count(*) >= 3
        ORDER BY n_mis DESC, random()
        LIMIT $1""", [n_sessions])

    m = await __import__("app.services.reconstruct", fromlist=["build_matcher"]).build_matcher()

    # monkeypatch thresholds for the after-run
    orig_fn = M.fuzzy_threshold
    orig_floor = m.appearance_floor
    orig_margin = m.appearance_margin

    def run_all():
        tiers = {}
        anomalies = 0
        links = 0
        total_hops = 0
        confs = []
        for r in rows:
            dets_rows = _sess_cache.get(r["session_id"], [])
            if len(dets_rows) < 2:
                continue
            traj = m.stitch_session(dets_rows)
            total_hops += len(dets_rows) - 1
            if traj:
                links += len(traj.hops)
                confs.append(traj.confidence)
                for h in traj.hops:
                    tiers[h.tier] = tiers.get(h.tier, 0) + 1
                    if h.anomaly:
                        anomalies += 1
        return {"tiers": tiers, "hard_anomalies": anomalies,
                "links": links, "total_hops": total_hops,
                "mean_conf": round(sum(confs)/len(confs), 3) if confs else None}

    # preload session detections
    _sess_cache = {}
    for r in rows:
        dets = await db.fetch(
            "SELECT * FROM detections WHERE session_id = $1 ORDER BY ts", [r["session_id"]])
        _sess_cache[r["session_id"]] = [det_from_row(d) for d in dets]

    # BEFORE (production thresholds)
    M.fuzzy_threshold = orig_fn
    m.appearance_floor = orig_floor
    m.appearance_margin = orig_margin
    before = run_all()

    # AFTER (proposed)
    M.fuzzy_threshold = make_threshold_fn(proposed)
    m.appearance_floor = proposed["appearance_floor"]
    m.appearance_margin = proposed["appearance_margin"]
    after = run_all()

    # restore
    M.fuzzy_threshold = orig_fn
    m.appearance_floor = orig_floor
    m.appearance_margin = orig_margin

    return {
        "proposed": proposed,
        "before": before,
        "after": after,
        "delta_links": after["links"] - before["links"],
        "delta_anomalies": after["hard_anomalies"] - before["hard_anomalies"],
        "note": "preview only — nothing saved",
    }
