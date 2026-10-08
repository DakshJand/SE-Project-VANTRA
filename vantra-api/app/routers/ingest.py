"""Detection ingestion: the camera-pipeline entry point.

POST /api/ingest — accepts a vehicle image (multipart), runs the full detection
pipeline (localize + OCR + Re-ID embed), stores the detection, runs blacklist +
anomaly checks, returns the detection with any alerts. This is the simulated
"camera feed ingestion" endpoint.
"""
from __future__ import annotations

import io
import random
from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, UploadFile
from PIL import Image

from app import db
from app.config import settings
from app.ml import ocr, reid, imagegen
from app.services import alerts as A, network

router = APIRouter(tags=["ingest"])


@router.post("/ingest")
async def ingest(camera_id: str = Form(...),
                 ts: datetime | None = Form(None),
                 image: UploadFile | None = File(None),
                 vehicle_type: str | None = Form(None)):
    """Ingest one camera frame. If no image is supplied (simulation mode), a
    synthetic frame is generated from the plate passed in `simulate_plate`."""
    rng = random.Random()
    ts = ts or datetime.now(timezone.utc)

    if image is not None:
        img = Image.open(io.BytesIO(await image.read())).convert("RGB")
        text, conf, box = ocr.detect_and_read(img)
        crop_path = plate_path = None
        embed = reid.embed(img, vehicle_type)
    else:
        raise ValueError("image file required (or use /ingest/simulate)")

    det_id = await db.fetch_val(
        """INSERT INTO detections (camera_id, ts, plate_raw, ocr_conf, vehicle_type,
             vehicle_color, crop_path, plate_crop_path, embed)
           VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9) RETURNING det_id""",
        [camera_id, ts, text or None, conf, vehicle_type,
         reid.classify_color(img), crop_path, plate_path,
         [float(x) for x in embed]])
    det = {"det_id": det_id, "camera_id": camera_id, "ts": ts, "plate_raw": text,
           "ocr_conf": conf}

    # blacklist check on ingest
    bl_rows = await db.fetch("SELECT plate FROM blacklist")
    blacklist = {r["plate"] for r in bl_rows}
    bl_alerts = A.check_blacklist([_mk_det(det_id, camera_id, ts, text, conf)], blacklist)
    for a in bl_alerts:
        await db.execute(
            """INSERT INTO alerts (kind, severity, plate, title, ts, det_ids, evidence)
               VALUES ($1,$2,$3,$4,$5,$6,$7)""",
            [a.kind, a.severity, a.plate, a.title, a.ts,
             [int(x) for x in a.det_ids], a.evidence])
    return {"detection": det, "alerts": [
        {"kind": a.kind, "severity": a.severity, "plate": a.plate, "title": a.title,
         "evidence": a.evidence} for a in bl_alerts]}


def _mk_det(det_id, camera_id, ts, plate, conf):
    from app.services.matcher import Detection
    return Detection(det_id, camera_id, ts, plate, conf, None, None, None, None, None, None, None)


@router.post("/ingest/simulate")
async def ingest_simulate(camera_id: str = Form(...),
                          plate: str = Form(...),
                          ts: datetime | None = Form(None),
                          condition: str = Form("clean"),
                          vehicle_type: str = Form("car"),
                          color: str = Form("white")):
    """Simulated camera ingestion: renders the vehicle image, runs the REAL detection
    pipeline (localization + OCR + embedding) and stores everything. This powers the
    live demo: 'a blacklisted vehicle just drove past CAM14'."""
    ts = ts or datetime.now(timezone.utc)
    if camera_id not in {c[0] for c in network.CAMERAS}:
        from fastapi import HTTPException
        raise HTTPException(400, f"unknown camera {camera_id}")

    s = imagegen.generate_sample(plate=plate, vehicle_type=vehicle_type,
                                 color=color, condition=condition,
                                 seed=rng_seed())
    img_dir = settings.image_dir / "ingest"
    stem = f"ing{int(ts.timestamp())}_{plate}"
    crop_path, plate_path = imagegen.save_sample(s, img_dir, stem)

    # full pipeline on the vehicle image
    text, conf, box = ocr.detect_and_read(s.vehicle_img)
    if not text:
        text, conf = None, 0.1
    embed = reid.embed(s.vehicle_img, vehicle_type)

    det_id = await db.fetch_val(
        """INSERT INTO detections (camera_id, ts, plate_raw, ocr_conf, plate_correct,
             vehicle_type, vehicle_color, crop_path, plate_crop_path, embed, session_id, gt_cam)
           VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12) RETURNING det_id""",
        [camera_id, ts, text, conf, plate, vehicle_type, color,
         crop_path, plate_path, [float(x) for x in embed],
         f"live-{plate}-{int(ts.timestamp())}", camera_id])

    bl_rows = await db.fetch("SELECT plate FROM blacklist")
    blacklist = {r["plate"] for r in bl_rows}
    det = _mk_det(det_id, camera_id, ts, text, conf)
    bl_alerts = A.check_blacklist([det], blacklist)
    for a in bl_alerts:
        await db.execute(
            """INSERT INTO alerts (kind, severity, plate, title, ts, det_ids, evidence)
               VALUES ($1,$2,$3,$4,$5,$6,$7)""",
            [a.kind, a.severity, a.plate, a.title, a.ts,
             [int(x) for x in a.det_ids], a.evidence])

    return {
        "detection": {"det_id": det_id, "camera_id": camera_id, "ts": ts,
                      "plate_raw": text, "ocr_conf": round(conf, 3),
                      "ground_truth_plate": plate, "condition": condition,
                      "crop_url": f"/images/{_rel(crop_path)}",
                      "plate_crop_url": f"/images/{_rel(plate_path)}"},
        "alerts": [{"kind": a.kind, "severity": a.severity, "plate": a.plate,
                    "title": a.title, "evidence": a.evidence} for a in bl_alerts],
    }


def rng_seed() -> int:
    return random.randint(0, 10**9)


def _rel(path: str) -> str:
    idx = path.find("images/")
    return path[idx + len("images/"):] if idx >= 0 else path
