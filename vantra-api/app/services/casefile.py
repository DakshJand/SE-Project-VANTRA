"""PDF case-file export for reconstructed trajectories.

Uses reportlab: map snapshot (static OSM tile composite or edge-line drawing),
timeline, per-hop evidence table, detection snapshots.
"""
from __future__ import annotations

import io
import math
from datetime import datetime
from pathlib import Path

import reportlab.lib.colors as rl_colors
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from app.config import settings

W, H = A4


def _mercator(lat: float, lng: float, zoom: int = 13):
    n = 2 ** zoom
    x = (lng + 180) / 360 * n
    lat_r = math.radians(lat)
    y = (1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n
    return x, y


def _tile_url(x: int, y: int, z: int) -> str:
    return f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"


def _fetch_tiles(latlngs: list[tuple[float, float]], zoom: int = 13) -> PILImage.Image | None:
    """Fetch a small OSM tile composite covering the trajectory. Offline-safe:
    returns None on any failure (caller falls back to a plain canvas map)."""
    try:
        import urllib.request
        xs, ys = [], []
        for lat, lng in latlngs:
            x, y = _mercator(lat, lng, zoom)
            xs.append(x)
            ys.append(y)
        x0, x1 = int(min(xs)), int(max(xs))
        y0, y1 = int(min(ys)), int(max(ys))
        if (x1 - x0) > 4 or (y1 - y0) > 4:
            return None
        w, h = (x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256
        img = PILImage.new("RGB", (w, h), (230, 230, 230))
        for tx in range(x0, x1 + 1):
            for ty in range(y0, y1 + 1):
                try:
                    req = urllib.request.Request(_tile_url(tx, ty, zoom),
                                                 headers={"User-Agent": "VANTRA/1.0"})
                    with urllib.request.urlopen(req, timeout=6) as r:
                        tile = PILImage.open(io.BytesIO(r.read())).convert("RGB")
                    img.paste(tile, ((tx - x0) * 256, (ty - y0) * 256))
                except Exception:
                    pass
        return img
    except Exception:
        return None


def _project_points(latlngs, img: PILImage.Image, zoom: int = 13):
    w, h = img.size
    xs, ys = [], []
    for lat, lng in latlngs:
        x, y = _mercator(lat, lng, zoom)
        xs.append(x)
        ys.append(y)
    x0, x1 = int(min(xs)), int(max(xs))
    y0, y1 = int(min(ys)), int(max(ys))
    pts = []
    for x, y in zip(xs, ys):
        px = (x - x0) * 256 / w * img.size[0]
        py = (y - y0) * 256 / h * img.size[1]
        pts.append((px, py))
    return pts


def build_casefile(traj: dict, path: Path | None = None) -> bytes:
    """traj: dict as returned by /api/trajectories/reconstruct."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    # ---- header
    c.setFillColorRGB(0.08, 0.09, 0.25)
    c.rect(0, H - 28 * mm, W, 28 * mm, stroke=0, fill=1)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(15 * mm, H - 15 * mm, "VANTRA")
    c.setFont("Helvetica", 10)
    c.drawString(15 * mm, H - 21 * mm, "One vehicle, one trail, across every camera in the city.")
    c.setFont("Helvetica-Bold", 12)
    c.drawRightString(W - 15 * mm, H - 15 * mm, "TRAJECTORY CASE FILE")
    c.setFont("Helvetica", 9)
    c.drawRightString(W - 15 * mm, H - 21 * mm,
                      f"Plate {traj['plate']}  |  {traj['t_start'][:16]} – {traj['t_end'][:16]}")

    y = H - 34 * mm
    # ---- summary
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(15 * mm, y, "Summary")
    y -= 6 * mm
    c.setFont("Helvetica", 9)
    conf = traj["confidence"]
    conf_txt = f"{conf*100:.0f}%"
    c.drawString(15 * mm, y, f"Overall confidence (rollup): {conf_txt}    "
                             f"Detections: {traj['n_detections']}    Hops: {len(traj['hops'])}")
    y -= 8 * mm

    # ---- map
    dets = traj["detections"]
    latlngs = [(d["lat"], d["lng"]) for d in dets]
    map_img = _fetch_tiles(latlngs)
    map_w = W - 30 * mm
    map_h = 80 * mm
    if map_img is not None:
        # draw route on tiles
        from PIL import ImageDraw
        pts = _project_points(latlngs, map_img)
        dr = ImageDraw.Draw(map_img)
        dr.line(pts, fill=(10, 60, 220), width=4)
        for i, (px, py) in enumerate(pts):
            dr.ellipse([px-5, py-5, px+5, py+5], outline=(10,60,220), width=3)
            dr.text((px+7, py-7), str(i+1), fill=(10,60,220))
        tmp = Path(settings.image_dir) / "_casefile_map.png"
        map_img.save(tmp)
        c.drawImage(str(tmp), 15 * mm, y - map_h, width=map_w, height=map_h)
        tmp.unlink()
    else:
        # fallback: plain coordinate map
        c.setFillColorRGB(0.95, 0.95, 0.97)
        c.rect(15 * mm, y - map_h, map_w, map_h, stroke=1, fill=1)
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica-Oblique", 8)
        c.drawString(18 * mm, y - 6 * mm, "map tiles unavailable offline — schematic only")
        xs = [d["lng"] for d in dets]
        ys = [d["lat"] for d in dets]
        def sx(x): return 20 * mm + (x - min(xs)) / (max(xs) - min(xs) + 1e-9) * (map_w - 12 * mm)
        def sy(yv): return y - map_h + 8 * mm + (yv - min(ys)) / (max(ys) - min(ys) + 1e-9) * (map_h - 16 * mm)
        c.setStrokeColorRGB(0.1, 0.3, 0.9)
        c.setLineWidth(1.5)
        p = c.beginPath()
        p.moveTo(sx(xs[0]), sy(ys[0]))
        for x, yv in zip(xs[1:], ys[1:]):
            p.lineTo(sx(x), sy(yv))
        c.drawPath(p)
        for i, d in enumerate(dets):
            cx, cy = sx(d["lng"]), sy(d["lat"])
            c.setFillColorRGB(0.1, 0.3, 0.9)
            c.circle(cx, cy, 1.6 * mm, stroke=0, fill=1)
            c.setFillColorRGB(0, 0, 0)
            c.setFont("Helvetica", 7)
            c.drawString(cx + 2 * mm, cy - 1 * mm, f"{i+1}. {d['camera_id']}")
    y -= map_h + 8 * mm

    # ---- timeline + evidence per hop
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(15 * mm, y, "Hop-by-hop evidence")
    y -= 6 * mm

    for i, h in enumerate(traj["hops"]):
        if y < 60 * mm:
            c.showPage()
            y = H - 20 * mm
            c.setFont("Helvetica-Bold", 11)
            c.drawString(15 * mm, y, "Hop-by-hop evidence (cont.)")
            y -= 6 * mm
        d_from = next(d for d in dets if d["det_id"] == h["from_det"])
        d_to = next(d for d in dets if d["det_id"] == h["to_det"])
        c.setFont("Helvetica-Bold", 9)
        color = (0.7, 0.1, 0.1) if h.get("anomaly") else (0.1, 0.3, 0.5)
        c.setFillColorRGB(*color)
        c.drawString(15 * mm, y,
                     f"Hop {i+1}: {d_from['camera_id']} {d_from['ts'][11:19]} -> "
                     f"{d_to['camera_id']} {d_to['ts'][11:19]}  "
                     f"[{h['tier']}  conf {h['confidence']*100:.0f}%]")
        y -= 4.5 * mm
        c.setFillColorRGB(0.15, 0.15, 0.15)
        c.setFont("Helvetica", 8)
        wrap = _wrap(h["reason"], 110)
        for line in wrap[:3]:
            c.drawString(18 * mm, y, line)
            y -= 4 * mm
        extras = []
        if h.get("edit_distance") is not None:
            extras.append(f"edit-distance {h['edit_distance']}")
        if h.get("appearance_score") is not None:
            extras.append(f"cosine {h['appearance_score']:.2f}")
        if h.get("time_score") is not None:
            extras.append(f"time-score {h['time_score']:.2f}")
        if h.get("hop_reach") == 2:
            extras.append("one camera missed (2-hop window)")
        if extras:
            c.setFont("Helvetica-Oblique", 8)
            c.drawString(18 * mm, y, "signals: " + ", ".join(extras))
            y -= 4 * mm
        y -= 2 * mm

    # ---- snapshots page
    c.showPage()
    c.setFillColorRGB(0.08, 0.09, 0.25)
    c.rect(0, H - 16 * mm, W, 16 * mm, stroke=0, fill=1)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(15 * mm, H - 10 * mm, f"VANTRA — detection snapshots — {traj['plate']}")
    y = H - 24 * mm
    x = 15 * mm
    img_root = Path(settings.image_dir).parent
    for i, d in enumerate(dets):
        if d.get("crop_url"):
            ipath = img_root / "data" / d["crop_url"].lstrip("/").replace("images/", "images/", 1)
            # crop_url like /images/history/xxx_veh.png ; image_dir is data/images
            ipath = img_root / "data" / d["crop_url"].lstrip("/")
            if ipath.exists():
                if x + 55 * mm > W - 10 * mm:
                    x = 15 * mm
                    y -= 42 * mm
                try:
                    c.drawImage(str(ipath), x, y - 36 * mm, width=52 * mm, height=36 * mm)
                    c.setFillColorRGB(0, 0, 0)
                    c.setFont("Helvetica", 7)
                    c.drawString(x, y - 38.5 * mm,
                                 f"{i+1}. {d['camera_id']} {d['ts'][11:16]}")
                    x += 57 * mm
                except Exception:
                    pass
    c.save()
    pdf = buf.getvalue()
    if path:
        path.write_bytes(pdf)
    return pdf


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


async def build_combined_casefile(plates: list[str], include_audit: bool = True) -> bytes:
    """Combined investigation case report: multiple plates' reconstructed
    trajectories, their pairwise co-location/convoy evidence, and the audit-log
    entries for queries on those plates. Reuses build_casefile's page primitives
    (header/map/evidence blocks) by composing per-trajectory sections, then adds
    the comparison + audit sections."""
    from app import db as dbm
    from app.services.reconstruct import reconstruct_for_plate
    from app.services.compare import compare_plates

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    # gather trajectories (best per plate)
    trajs = []
    for plate in plates:
        rs = await reconstruct_for_plate(plate)
        if rs:
            trajs.append(max(rs, key=lambda t: t["n_detections"]))

    # ---- combined cover
    c.setFillColorRGB(0.08, 0.09, 0.25)
    c.rect(0, H - 28 * mm, W, 28 * mm, stroke=0, fill=1)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(15 * mm, H - 15 * mm, "VANTRA")
    c.setFont("Helvetica", 10)
    c.drawString(15 * mm, H - 21 * mm, "One vehicle, one trail, across every camera in the city.")
    c.setFont("Helvetica-Bold", 12)
    c.drawRightString(W - 15 * mm, H - 15 * mm, "COMBINED CASE REPORT")
    c.setFont("Helvetica", 9)
    c.drawRightString(W - 15 * mm, H - 21 * mm,
                      f"{len(trajs)} vehicles · generated {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    y = H - 34 * mm
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(15 * mm, y, "Vehicles in this report")
    y -= 6 * mm
    c.setFont("Helvetica", 9)
    for t in trajs:
        c.drawString(15 * mm, y,
                     f"{t['plate']}  ·  {t['n_detections']} detections · "
                     f"confidence {t['confidence']*100:.0f}%  ·  "
                     f"{t['t_start'][:16]} – {t['t_end'][:16]}")
        y -= 5 * mm

    # ---- co-location / convoy evidence
    if len(trajs) >= 2:
        comp = await compare_plates(plates)
        y -= 4 * mm
        c.setFont("Helvetica-Bold", 11)
        c.drawString(15 * mm, y, "Co-location / convoy evidence")
        y -= 6 * mm
        c.setFont("Helvetica", 9)
        if comp.get("convoys"):
            for cv in comp["convoys"]:
                for line in _wrap(cv["explanation"], 100)[:3]:
                    c.drawString(15 * mm, y, line)
                    y -= 4.5 * mm
                y -= 2 * mm
        else:
            c.drawString(15 * mm, y, "No co-locations found between the selected plates.")
            y -= 5 * mm

    c.showPage()

    # ---- per-vehicle sections: reuse the single-trajectory renderer per plate by
    # drawing its evidence rows directly (same table style)
    for t in trajs:
        c.setFillColorRGB(0.08, 0.09, 0.25)
        c.rect(0, H - 16 * mm, W, 16 * mm, stroke=0, fill=1)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(15 * mm, H - 10 * mm,
                     f"VANTRA — vehicle {t['plate']} — {t['confidence']*100:.0f}% confidence")
        y = H - 24 * mm
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(15 * mm, y, "Hop-by-hop evidence")
        y -= 6 * mm
        dets = t["detections"]
        for i, h in enumerate(t["hops"]):
            if y < 30 * mm:
                c.showPage()
                y = H - 20 * mm
            d_from = next(d for d in dets if d["det_id"] == h["from_det"])
            d_to = next(d for d in dets if d["det_id"] == h["to_det"])
            c.setFont("Helvetica-Bold", 9)
            color = (0.7, 0.1, 0.1) if h.get("anomaly") else (0.1, 0.3, 0.5)
            c.setFillColorRGB(*color)
            c.drawString(15 * mm, y,
                         f"Hop {i+1}: {d_from['camera_id']} -> {d_to['camera_id']}  "
                         f"[{h['tier']}  conf {h['confidence']*100:.0f}%]")
            y -= 4.5 * mm
            c.setFillColorRGB(0.15, 0.15, 0.15)
            c.setFont("Helvetica", 8)
            for line in _wrap(h["reason"], 110)[:3]:
                c.drawString(18 * mm, y, line)
                y -= 4 * mm
            y -= 2 * mm
        c.showPage()

    # ---- audit trail for these plates
    if include_audit:
        c.setFillColorRGB(0.08, 0.09, 0.25)
        c.rect(0, H - 16 * mm, W, 16 * mm, stroke=0, fill=1)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(15 * mm, H - 10 * mm, "VANTRA — audit trail (queries on these plates)")
        y = H - 24 * mm
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica", 8)
        plate_list = ",".join(f"'{p}'" for p in plates)
        entries = await dbm.fetch(
            f"""SELECT ts, operator, action, detail FROM audit_log
                WHERE detail LIKE ANY (ARRAY[{plate_list}])
                   OR query LIKE ANY (ARRAY[{plate_list}])
                ORDER BY ts DESC LIMIT 200""")
        if not entries:
            c.drawString(15 * mm, y, "No audit entries recorded for these plates.")
        for e in entries:
            if y < 20 * mm:
                c.showPage()
                y = H - 20 * mm
            c.drawString(15 * mm, y,
                         f"{e['ts'].strftime('%Y-%m-%d %H:%M')}  {e['operator']:12s} "
                         f"{e['action']:18s} {e['detail'] or ''}")
            y -= 4 * mm
        c.showPage()

    c.save()
    return buf.getvalue()
