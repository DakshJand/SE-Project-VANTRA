"""Trajectory reconstruction engine — VANTRA's core matcher.

Tiered, explainable matching over the camera route graph:
  1. exact        — exact plate match on a route-plausible edge
  2. fuzzy_unique — fuzzy plate match (edit distance, OCR-conf-aware threshold) on a
                    uniquely route-plausible edge
  3. fuzzy_disambiguated — fuzzy match with multiple route-plausible candidates,
                    disambiguated by appearance similarity; ambiguous ties are NOT
                    forced (flagged for human review)
  4. appearance_only — plate unreadable; appearance similarity + route plausibility

Every hop carries machine-readable evidence: tier, edit distance, travel-time
percentile, cosine similarity, OCR confidences, and the rule that fired.

Travel-time model per edge: min_time hard floor (violation => hard anomaly, not a
miss), learned p5/p50/p95 distribution, and a confidence-decay curve beyond the
window that never reaches zero (wider decay on stop-eligible edges).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from app.ml import reid
from app.services import network

# ---------------------------------------------------------------- data classes


@dataclass
class EdgeStats:
    edge_id: str
    hop_reach: int
    p5: float
    p50: float
    p95: float
    mean: float
    std: float
    n_samples: int


@dataclass
class Detection:
    det_id: int
    camera_id: str
    ts: datetime
    plate_raw: str | None
    ocr_conf: float | None
    plate_correct: str | None   # ground truth (evaluation only)
    vehicle_type: str | None
    vehicle_color: str | None
    crop_path: str | None
    plate_crop_path: str | None
    embed: list[float] | None
    session_id: str | None

    @property
    def readable(self) -> bool:
        return bool(self.plate_raw) and (self.ocr_conf or 0) > 0.30


@dataclass
class HopEvidence:
    tier: str
    from_det: int
    to_det: int
    edge_id: str | None
    hop_reach: int
    dt_seconds: float
    edit_distance: int | None
    ocr_conf_min: float | None
    time_score: float | None
    time_percentile: float | None
    appearance_score: float | None
    type_match: bool | None
    color_match: bool | None
    confidence: float
    reason: str
    candidates: list[dict] = field(default_factory=list)   # for ambiguous review
    anomaly: str | None = None      # "impossible_edge" when min_time violated


@dataclass
class Trajectory:
    detections: list[Detection]
    hops: list[HopEvidence]
    plate: str

    @property
    def confidence(self) -> float:
        """Rollup confidence: harmonic-mean-like. A single weak hop pulls the whole
        score down hard (not diluted by an average)."""
        if not self.hops:
            return 1.0 if self.detections else 0.0
        scores = [h.confidence for h in self.hops]
        # soft-min: arithmetic mean dragged toward the minimum hop
        return float((min(scores) * 2 + sum(scores) / len(scores)) / 3)


# ---------------------------------------------------------------- scoring primitives


def edit_distance(a: str, b: str) -> int:
    if a is None or b is None:
        return 99
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def fuzzy_threshold(ocr_conf_a: float | None, ocr_conf_b: float | None,
                    max_gap: float = 0.15) -> int:
    """OCR-conf-aware edit-distance threshold. High conf -> tight (1), low conf ->
    wider (up to 3). Both confidences contribute; the weaker one dominates."""
    conf = min(ocr_conf_a or 0.5, ocr_conf_b or 0.5)
    # Calibrated for the EasyOCR confidence distribution (correct reads p10≈0.59,
    # wrong reads mean≈0.29): high conf -> tight; low conf -> wider window.
    if conf >= 0.75:
        return 1
    if conf >= 0.45:
        return 2
    return 3


def time_score(dt: float, stats: EdgeStats, stop_eligible: bool) -> tuple[float, float, str]:
    """Score a travel time against the learned edge distribution.
    Returns (score in [0,1], percentile-ish position, band label).
    - below min_time: handled by caller as hard anomaly
    - within [p5, p95]: full score
    - beyond p95: exponential decay (never 0); stop-eligible edges decay slower
    """
    if dt < stats.p5:
        # between min_time and p5: partial credit scaled by how deep below p5
        span = max(stats.p5 - stats.min_time_floor(), 1.0) if hasattr(stats, "min_time_floor") else stats.p5
        frac = dt / stats.p5 if stats.p5 > 0 else 0.0
        return max(0.10, frac * 0.9), frac, "below_p5"
    if dt <= stats.p95:
        return 1.0, 0.5, "within_window"
    # decay beyond p95
    excess = (dt - stats.p95) / max(stats.p50, 1.0)
    decay_tau = 6.0 if stop_eligible else 2.5   # stop-eligible: much wider grace
    score = max(0.05, math.exp(-excess / decay_tau) * 0.85)
    return score, 1.0, "beyond_p95"


def appearance_score(det_a: Detection, det_b: Detection) -> tuple[float | None, bool | None, bool | None]:
    """Cosine similarity over Re-ID embeddings + type/color agreement."""
    if det_a.embed and det_b.embed:
        cos = reid.cosine(det_a.embed, det_b.embed)
    else:
        cos = None
    tmatch = (det_a.vehicle_type == det_b.vehicle_type) if det_a.vehicle_type and det_b.vehicle_type else None
    cmatch = (det_a.vehicle_color == det_b.vehicle_color) if det_a.vehicle_color and det_b.vehicle_color else None
    return cos, tmatch, cmatch


# ---------------------------------------------------------------- the matcher


class Matcher:
    def __init__(self, edges: dict[str, network.Edge],
                 stats: dict[tuple[str, int], EdgeStats],
                 appearance_margin: float = 0.06,
                 appearance_floor: float = 0.55,
                 hop_skip_multiplier: float = 1.9):
        self.edges = edges
        self.stats = stats
        self.adj = network.adjacency_map()
        self.two_hop = network.two_hop_pairs()
        self.appearance_margin = appearance_margin
        self.appearance_floor = appearance_floor
        self.hop_skip_multiplier = hop_skip_multiplier

    # -------------------------------------------------- candidate generation

    def route_plausible(self, from_cam: str, to_cam: str, dt: float) -> list[tuple[str, int, EdgeStats]]:
        """All (edge_id, hop_reach, stats) making from_cam->to_cam plausible within dt."""
        out = []
        direct = f"{from_cam}->{to_cam}"
        if to_cam in self.adj.get(from_cam, []):
            st = self.stats.get((direct, 1))
            if st and dt >= st.p5 * 0.10:  # some floor; min_time handled separately
                out.append((direct, 1, st))
        if to_cam in self.two_hop.get(from_cam, []):
            st2 = self.stats.get((direct, 2))
            if st2:
                out.append((direct, 2, st2))
            else:
                # synthesize 2-hop window from the two constituent direct edges
                mids = [m for m in self.adj[from_cam] if to_cam in self.adj[m]]
                if mids:
                    st_a = self.stats.get((f"{from_cam}->{mids[0]}", 1))
                    st_b = self.stats.get((f"{mids[0]}->{to_cam}", 1))
                    if st_a and st_b:
                        out.append((direct, 2, EdgeStats(
                            direct, 2,
                            p5=st_a.p5 + st_b.p5, p50=st_a.p50 + st_b.p50,
                            p95=st_a.p95 + st_b.p95, mean=st_a.mean + st_b.mean,
                            std=math.sqrt(st_a.std**2 + st_b.std**2),
                            n_samples=st_a.n_samples + st_b.n_samples)))
        return out

    def min_time_for(self, from_cam: str, to_cam: str, hop_reach: int) -> float | None:
        """Hard floor across the reach: min direct edge time, or sum for 2-hop."""
        if hop_reach == 1:
            e = self.edges.get(f"{from_cam}->{to_cam}")
            return e.min_time_s if e else None
        mids = [m for m in self.adj.get(from_cam, []) if to_cam in self.adj.get(m, [])]
        if not mids:
            return None
        best = None
        for m in mids:
            ea = self.edges.get(f"{from_cam}->{m}")
            eb = self.edges.get(f"{m}->{to_cam}")
            if ea and eb:
                s = ea.min_time_s + eb.min_time_s
                best = s if best is None else min(best, s)
        return best

    def stop_eligible_between(self, from_cam: str, to_cam: str) -> bool:
        if to_cam in self.adj.get(from_cam, []):
            e = self.edges.get(f"{from_cam}->{to_cam}")
            if e and e.stop_eligible:
                return True
        mids = [m for m in self.adj.get(from_cam, []) if to_cam in self.adj.get(m, [])]
        return any(self.edges.get(f"{from_cam}->{m}") and self.edges[f"{from_cam}->{m}"].stop_eligible
                   or self.edges.get(f"{m}->{to_cam}") and self.edges[f"{m}->{to_cam}"].stop_eligible
                   for m in mids)

    # -------------------------------------------------- hop linking

    def link(self, prev: Detection, cand: Detection) -> HopEvidence | None:
        """Attempt to link prev -> cand. Returns evidence or None (not linkable)."""
        if cand.ts <= prev.ts:
            return None
        dt = (cand.ts - prev.ts).total_seconds()
        plausibles = self.route_plausible(prev.camera_id, cand.camera_id, dt)
        if not plausibles:
            return None

        # hard floor check across all plausible routes
        min_times = [mt for mt in (self.min_time_for(prev.camera_id, cand.camera_id, hr)
                                   for _, hr, _ in plausibles) if mt is not None]
        if min_times and dt < min(min_times):
            # physically impossible -> hard anomaly evidence (not a silent no-match)
            best_edge, best_hr, best_st = plausibles[0]
            return HopEvidence(
                tier="exact" if (prev.plate_raw and prev.plate_raw == cand.plate_raw) else "fuzzy_unique",
                from_det=prev.det_id, to_det=cand.det_id, edge_id=best_edge, hop_reach=best_hr,
                dt_seconds=dt, edit_distance=edit_distance(prev.plate_raw, cand.plate_raw),
                ocr_conf_min=min(filter(None, [prev.ocr_conf, cand.ocr_conf]), default=None),
                time_score=0.0, time_percentile=0.0,
                appearance_score=None, type_match=None, color_match=None,
                confidence=0.0,
                reason=(f"HARD ANOMALY: traversed {prev.camera_id}->{cand.camera_id} in "
                        f"{dt:.0f}s, below physical minimum {min(min_times):.0f}s "
                        f"(distance/speed-limit floor). Likely spoofed or cloned plate."),
                anomaly="impossible_edge",
            )

        # pick the most plausible reach (prefer direct if time fits better)
        def reach_fit(item):
            _eid, hr, st = item
            if st.p5 <= dt <= st.p95:
                return (0, hr)          # within window; prefer direct
            return (1, hr)
        edge_id, hop_reach, st = sorted(plausibles, key=reach_fit)[0]

        ts, pct_pos, band = time_score(dt, st, self.stop_eligible_between(prev.camera_id, cand.camera_id))
        cos, tmatch, cmatch = appearance_score(prev, cand)
        ocr_min = min(filter(None, [prev.ocr_conf, cand.ocr_conf]), default=None)

        # ---- tier decision
        if prev.readable and cand.readable:
            ed = edit_distance(prev.plate_raw, cand.plate_raw)
            thresh = fuzzy_threshold(prev.ocr_conf, cand.ocr_conf)
            if ed == 0:
                tier = "exact"
                conf = 0.55 + 0.30 * ts + 0.10 * (ocr_min or 0.5) + (0.05 * (cos or 0.5) if cos else 0)
                reason = (f"Exact plate match {prev.plate_raw} on route-plausible edge "
                          f"{edge_id} (hop {hop_reach}); travel time {dt:.0f}s in {band} "
                          f"(p5={st.p5:.0f}s p95={st.p95:.0f}s).")
                if hop_reach == 2:
                    conf *= 0.92
                    reason += " One intermediate camera missed (2-hop window)."
                return HopEvidence("exact", prev.det_id, cand.det_id, edge_id, hop_reach, dt,
                                   ed, ocr_min, ts, pct_pos, cos, tmatch, cmatch,
                                   float(min(conf, 0.99)), reason)
            if ed <= thresh:
                # fuzzy: unique or ambiguous depends on OTHER candidates — caller
                # (stitch) resolves; return the base evidence with tier fuzzy_unique
                conf = 0.35 + 0.25 * ts + 0.15 * ((thresh - ed) / thresh) + 0.15 * (cos or 0.5) \
                    + 0.10 * (ocr_min or 0.5)
                reason = (f"Fuzzy plate match: edit distance {ed} <= threshold {thresh} "
                          f"(OCR conf {ocr_min:.2f} widens threshold), plates "
                          f"{prev.plate_raw} vs {cand.plate_raw}; travel time {dt:.0f}s "
                          f"({band}).")
                return HopEvidence("fuzzy_unique", prev.det_id, cand.det_id, edge_id, hop_reach,
                                   dt, ed, ocr_min, ts, pct_pos, cos, tmatch, cmatch,
                                   float(min(conf, 0.9)), reason)
            return None
        # ---- appearance-only path (one side unreadable)
        if cos is not None and cos >= self.appearance_floor and (tmatch or tmatch is None):
            tier = "appearance_only"
            conf = 0.20 + 0.25 * ts + 0.30 * cos + (0.05 if tmatch else 0.0) \
                + (0.05 if cmatch else 0.0)
            reason = (f"Plate unreadable (OCR conf {ocr_min}); matched on appearance: "
                      f"Re-ID cosine {cos:.2f}, type_match={tmatch}, color_match={cmatch}, "
                      f"route-plausible edge {edge_id} with travel time {dt:.0f}s ({band}).")
            if hop_reach == 2:
                conf *= 0.9
            return HopEvidence(tier, prev.det_id, cand.det_id, edge_id, hop_reach, dt,
                               None, ocr_min, ts, pct_pos, cos, tmatch, cmatch,
                               float(min(conf, 0.75)), reason)
        return None

    # -------------------------------------------------- trajectory stitching

    def stitch(self, detections: list[Detection], ambiguous_cb=None) -> list[Trajectory]:
        """Greedy temporal stitching: sort by time; grow trajectories from seeds.
        Exact/fuzzy plate matches chain; appearance-only matches attach when the plate
        is unreadable. Ambiguous fuzzy candidates (multiple plausible, close scores)
        are NOT forced — routed to ambiguous_cb for human review."""
        detections = sorted(detections, key=lambda d: d.ts)
        used: set[int] = set()
        trajectories: list[Trajectory] = []
        by_camera: dict[str, list[Detection]] = {}
        for d in detections:
            by_camera.setdefault(d.camera_id, []).append(d)

        # seed on readable plates first (strongest anchors)
        seeds = [d for d in detections if d.readable] + [d for d in detections if not d.readable]
        for seed in seeds:
            if seed.det_id in used:
                continue
            traj = Trajectory(detections=[seed], hops=[], plate=seed.plate_raw or "")
            used.add(seed.det_id)
            current = seed
            while True:
                # candidate pruning: only detections at route-reachable cameras
                # (adjacent or 2-hop from the current camera) can ever link —
                # bucketing keeps this O(n·k) at scale instead of O(n²)
                reachable = set(self.adj.get(current.camera_id, []))
                reachable.update(self.two_hop.get(current.camera_id, []))
                cands = [d for d in by_camera.get(current.camera_id, [])
                         if d.det_id not in used] \
                        + [d for c in reachable for d in by_camera.get(c, [])
                           if d.det_id not in used]
                cands = [d for d in cands
                         if d.ts > current.ts
                         and 0 < (d.ts - current.ts).total_seconds() < 3 * 3600]
                if not cands:
                    break
                # score all candidates for this hop
                scored = []
                for c in cands:
                    ev = self.link(current, c)
                    if ev and not ev.anomaly:
                        # combined score: plate/tier confidence + appearance
                        comb = ev.confidence + (0.15 * (ev.appearance_score or 0.5) if ev.appearance_score else 0)
                        scored.append((comb, ev, c))
                if not scored:
                    break
                scored.sort(key=lambda x: -x[0])
                if len(scored) >= 2:
                    top, second = scored[0], scored[1]
                    # ambiguity guard for fuzzy tiers: if both fuzzy-match the current
                    # plate and are too close, don't force
                    if (top[1].tier in ("fuzzy_unique",) and second[1].tier == "fuzzy_unique"
                            and top[1].edit_distance == second[1].edit_distance
                            and top[0] - second[0] < self.appearance_margin):
                        if ambiguous_cb:
                            ambiguous_cb(current, top[2], second[2], top[1], second[1])
                        # fall through to appearance-only if available
                        ap = [s for s in scored if s[1].tier == "appearance_only"]
                        if ap:
                            comb, ev, c = ap[0]
                        else:
                            break
                    else:
                        comb, ev, c = top
                else:
                    comb, ev, c = scored[0]
                traj.hops.append(ev)
                traj.detections.append(c)
                used.add(c.det_id)
                current = c
            if len(traj.detections) >= 2:
                trajectories.append(traj)
            elif not traj.plate:
                # single unreadable detection: no trajectory
                pass
        return trajectories

    def stitch_session(self, detections: list[Detection]) -> Trajectory | None:
        """Stitch a known session (evaluation/demo): all detections for one ground-truth
        trip in temporal order; link sequentially."""
        dets = sorted(detections, key=lambda d: d.ts)
        if not dets:
            return None
        traj = Trajectory(detections=[dets[0]], hops=[], plate=dets[0].plate_raw or "")
        for prev, nxt in zip(dets, dets[1:]):
            ev = self.link(prev, nxt)
            if ev is None:
                # not linkable — trajectory broken here; keep both but record gap
                traj.detections.append(nxt)
                continue
            traj.hops.append(ev)
            traj.detections.append(nxt)
        return traj
