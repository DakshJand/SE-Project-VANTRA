"""Synthetic Indian license-plate and vehicle image generator for VANTRA.

Generates:
- Full vehicle crops (simple rendered vehicles: body shape, color, type)
- Plate crops (Indian formats: KA01AB1234, BH series, commercial yellow, etc.)
- Augmentations for stress-testing: motion blur, rotation/angle, low light,
  rain/fog overlay, partial occlusion (dirt/damage overlay on characters).

All images are SIMULATED — see NOTES.md.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

_FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_FONT_PATH = next((p for p in _FONT_PATHS if Path(p).exists()), _FONT_PATHS[0])
_FONT = ImageFont.truetype(_FONT_PATH, 40)
_FONT_SMALL = ImageFont.truetype(_FONT_PATH, 14)

# Indian state RTO codes commonly seen in Bengaluru
RTO_CODES = ["KA01", "KA02", "KA03", "KA04", "KA05", "KA41", "KA51", "KA53", "BH12", "KA09"]

COLORS = [
    ("white", (235, 235, 235)),
    ("black", (30, 30, 32)),
    ("silver", (170, 172, 175)),
    ("red", (178, 34, 34)),
    ("blue", (40, 70, 160)),
    ("grey", (105, 105, 110)),
    ("green", (40, 120, 70)),
    ("yellow", (220, 190, 40)),   # taxis
]

PLATE_CHARS = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"  # no I,O,Q (real RTO practice)
VEHICLE_TYPES = ["car", "suv", "truck", "bus", "auto"]
TYPE_WEIGHTS = [0.45, 0.25, 0.10, 0.08, 0.12]

OCR_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def random_plate(rng: random.Random) -> str:
    rto = rng.choice(RTO_CODES)
    letters = "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(2))
    digits = f"{rng.randint(0, 9999):04d}"
    return f"{rto}{letters}{digits}"


@dataclass
class VehicleSample:
    plate: str
    vehicle_type: str
    color_name: str
    vehicle_img: Image.Image
    plate_img: Image.Image          # plate region as it appears on the vehicle crop
    condition: str                  # clean | blur | angle | lowlight | occlusion | rain
    # ground truth localization within vehicle crop:
    plate_box: tuple[int, int, int, int]  # x0,y0,x1,y1 in vehicle image coords


def _draw_plate(plate: str, rng: random.Random, w=300, h=80) -> Image.Image:
    """Render an Indian plate: black text on white (private) or yellow (commercial).
    Fixed 300px canvas (the OCR geometry is tuned to it); the font size shrinks
    until the 10 chars fit, so wide fonts (DejaVu in containers) don't clip the
    last glyph the way a fixed size would."""
    commercial = rng.random() < 0.25
    bg = (235, 180, 20) if commercial else (248, 248, 248)
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w - 1, h - 1], outline=(20, 20, 20), width=3)
    d.text((10, h - 12), "IND", font=_FONT_SMALL, fill=(10, 10, 10), anchor="lm")
    # pick the largest font size whose text fits the text area (x 30..w-8)
    size = 40
    while size > 18:
        f = ImageFont.truetype(_FONT_PATH, size)
        if 30 + sum(f.getlength(ch) for ch in plate) <= w - 8:
            break
        size -= 2
    # plate text, slight kerning jitter
    x = 30.0
    for ch in plate:
        adv = f.getlength(ch)
        d.text((x + adv / 2, 38), ch, font=f, fill=(15, 15, 15), anchor="mm")
        x += adv + rng.uniform(-0.3, 0.3)
    return img


def _draw_vehicle(vtype: str, color_rgb: tuple, plate_img: Image.Image,
                  rng: random.Random) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Render a simple vehicle with the plate affixed; returns (image, plate_box)."""
    W, H = 420, 300
    img = Image.new("RGB", (W, H), (105, 115, 125))  # road-ish background
    d = ImageDraw.Draw(img)
    # background banding to look like a road scene
    d.rectangle([0, 0, W, 60], fill=(140, 160, 175))
    d.rectangle([0, 210, W, H], fill=(90, 92, 95))
    d.line([0, 255, W, 255], fill=(210, 210, 90), width=4)

    body = color_rgb
    shadow = tuple(max(0, c - 60) for c in color_rgb)
    if vtype in ("car", "suv"):
        bw, bh = (300, 110) if vtype == "car" else (320, 130)
        x0, y0 = (W - bw) // 2, 120
        d.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], 18, fill=body)
        # cabin
        cw = bw * 0.55
        cx0 = x0 + (bw - cw) / 2
        d.rounded_rectangle([cx0, y0 - 45, cx0 + cw, y0 + 25], 12, fill=shadow)
        d.rectangle([cx0 + 10, y0 - 35, cx0 + cw / 2 - 5, y0 + 10], fill=(150, 190, 210))
        d.rectangle([cx0 + cw / 2 + 5, y0 - 35, cx0 + cw - 10, y0 + 10], fill=(150, 190, 210))
        # wheels
        for wx in (x0 + 45, x0 + bw - 45):
            d.ellipse([wx - 26, y0 + bh - 26, wx + 26, y0 + bh + 26], fill=(25, 25, 25))
            d.ellipse([wx - 12, y0 + bh - 12, wx + 12, y0 + bh + 12], fill=(120, 120, 125))
        plate_pos = (x0 + bw // 2 - plate_img.width // 2, y0 + bh - 34)
    elif vtype in ("truck", "bus"):
        bw, bh = 330, 150
        x0, y0 = (W - bw) // 2, 90
        d.rectangle([x0, y0, x0 + bw, y0 + bh], fill=body)
        d.rectangle([x0 + 15, y0 + 12, x0 + 120, y0 + 60], fill=(150, 190, 210))
        for wx in (x0 + 50, x0 + bw - 50):
            d.ellipse([wx - 28, y0 + bh - 28, wx + 28, y0 + bh + 28], fill=(25, 25, 25))
        plate_pos = (x0 + bw // 2 - plate_img.width // 2, y0 + bh - 30)
    else:  # auto
        bw, bh = 200, 90
        x0, y0 = (W - bw) // 2, 140
        d.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], 15, fill=body)
        d.arc([x0 + 30, y0 - 70, x0 + bw - 30, y0 + 20], 180, 360, fill=(30, 130, 30), width=10)
        d.ellipse([x0 + 20, y0 + bh - 20, x0 + 70, y0 + bh + 30], fill=(25, 25, 25))
        d.ellipse([x0 + bw - 70, y0 + bh - 20, x0 + bw - 20, y0 + bh + 30], fill=(25, 25, 25))
        plate_pos = (x0 + bw // 2 - plate_img.width // 2, y0 + bh - 26)

    img.paste(plate_img, (int(plate_pos[0]), int(plate_pos[1])))
    box = (int(plate_pos[0]), int(plate_pos[1]),
           int(plate_pos[0]) + plate_img.width,
           int(plate_pos[1]) + plate_img.height)
    return img, box


# ---------------- augmentations ----------------

def aug_clean(img: Image.Image, rng: random.Random) -> Image.Image:
    return img


def aug_blur(img: Image.Image, rng: random.Random) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(2.0, 4.0)))


def aug_angle(img: Image.Image, rng: random.Random) -> Image.Image:
    angle = rng.uniform(-22, 22)
    rot = img.rotate(angle, expand=False, fillcolor=(60, 60, 60))
    # slight perspective squeeze: scale x down
    return rot.resize((int(rot.width * rng.uniform(0.72, 0.95)), rot.height))


def aug_lowlight(img: Image.Image, rng: random.Random) -> Image.Image:
    dark = ImageEnhance.Brightness(img).enhance(rng.uniform(0.38, 0.60))
    noise = np.array(dark).astype(np.int16)
    noise += np.random.default_rng(rng.randint(0, 2**30)).integers(-18, 18, noise.shape, dtype=np.int16)
    return Image.fromarray(np.clip(noise, 0, 255).astype(np.uint8))


def aug_occlusion(img: Image.Image, rng: random.Random) -> Image.Image:
    """Dirt/damage overlay on part of the plate (via alpha mask region)."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    n = rng.randint(1, 3)
    for _ in range(n):
        x = rng.randint(0, w - 40)
        y = rng.randint(0, h - 12)
        bw = rng.randint(30, 80)
        d.rectangle([x, y, x + bw, y + rng.randint(6, 18)], fill=(92, 74, 48))  # dirt
    return out


def aug_rain(img: Image.Image, rng: random.Random) -> Image.Image:
    """Rain streaks + slight fog (brightness lift, contrast drop)."""
    out = ImageEnhance.Contrast(img).enhance(0.8)
    arr = np.array(out).astype(np.int16)
    g = np.random.default_rng(rng.randint(0, 2**30))
    for _ in range(rng.randint(60, 140)):
        x, y = g.integers(0, arr.shape[1]), g.integers(0, arr.shape[0] - 8)
        ln = g.integers(6, 16)
        arr[y:y + ln, x, :] = np.clip(arr[y:y + ln, x, :] + 70, 0, 255)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


AUGMENTATIONS = {
    "clean": aug_clean,
    "blur": aug_blur,
    "angle": aug_angle,
    "lowlight": aug_lowlight,
    "occlusion": aug_occlusion,
    "rain": aug_rain,
}


def generate_sample(plate: str | None = None, vehicle_type: str | None = None,
                    color: str | None = None, condition: str = "clean",
                    seed: int | None = None) -> VehicleSample:
    rng = random.Random(seed)
    plate = plate or random_plate(rng)
    vtype = vehicle_type or rng.choices(VEHICLE_TYPES, TYPE_WEIGHTS)[0]
    color_name, color_rgb = (color, dict(COLORS)[color]) if color else rng.choice(COLORS)

    plate_img = _draw_plate(plate, rng)
    vehicle_img, box = _draw_vehicle(vtype, color_rgb, plate_img, rng)
    aug = AUGMENTATIONS[condition]
    vehicle_img = aug(vehicle_img, rng)
    # plate crop = the (augmented) region of the vehicle image
    x0, y0, x1, y1 = box
    plate_crop = vehicle_img.crop((x0, y0, x1, y1))
    return VehicleSample(plate, vtype, color_name, vehicle_img, plate_crop, condition, box)


def save_sample(sample: VehicleSample, out_dir: Path, stem: str) -> tuple[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    vpath = out_dir / f"{stem}_veh.png"
    ppath = out_dir / f"{stem}_plate.png"
    sample.vehicle_img.save(vpath)
    sample.plate_img.save(ppath)
    return str(vpath), str(ppath)
