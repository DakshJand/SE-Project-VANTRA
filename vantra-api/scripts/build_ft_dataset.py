"""Build the fine-tuning corpus from multiple public real-plate datasets.

Sources (partitioned BY SOURCE to prevent leakage between visually similar images):
  A: sonnetechnology/license-plate-text-recognition-full (US/EU plates,
     bbox + target text) — train 6176 / val 1765 / test 882
  B: EZCon/taiwan-license-plate-recognition (Asian plates, rotated xywhr boxes
     + license_number) — train 2540 / test 259 (val split unused, kept source-separated)

Output: data/realworld/ft/{train,val,test}/  with crops saved as PNGs + labels.txt.
Held-out test = A-test (882) + B-test (259), written once, never touched until
final evaluation.
"""
from __future__ import annotations

import io
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path("data/realworld")
OUT = ROOT / "ft"


def crop_from_xywhr(img: Image.Image, box) -> tuple[int, int, int, int]:
    """Axis-aligned bbox from rotated xywhr (flat list cx, cy, w, h, angle_radians):
    take the enclosing rectangle of the rotated rect."""
    b = box if isinstance(box, (list, tuple)) else list(box)
    if len(b) < 5:
        raise ValueError("short box")
    cx, cy, w, h, angle = b[0], b[1], b[2], b[3], b[4]
    a = math.radians(angle)
    # corners of the centered rect
    pts = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        x = cx + sx * w / 2 * math.cos(a) - sy * h / 2 * math.sin(a)
        y = cy + sx * w / 2 * math.sin(a) + sy * h / 2 * math.cos(a)
        pts.append((x, y))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (max(0, int(min(xs))), max(0, int(min(ys))),
            min(img.width, int(max(xs))), min(img.height, int(max(ys))))


def clean_label(t: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", t).upper()


def extract_sonnet(split: str, out_dir: Path, limit: int | None = None) -> int:
    t = pq.read_table(str(ROOT / f"{split}.parquet"))
    rows = t.to_pylist()
    n = 0
    labels = []
    for i, r in enumerate(rows):
        if limit and n >= limit:
            break
        label = clean_label(r["target"][0])
        if len(label) < 3:
            continue
        img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        bb = r["bbox"][0]
        x0, y0, x1, y1 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        if x1 - x0 < 20 or y1 - y0 < 8:
            continue
        crop = img.crop((x0, y0, x1, y1))
        crop.save(out_dir / f"a_{split}_{i:05d}.png")
        labels.append(f"a_{split}_{i:05d}.png\t{label}")
        n += 1
    (out_dir / "labels.txt").write_text("\n".join(labels))
    return n


def extract_taiwan(files: list[Path], prefix: str, out_dir: Path) -> int:
    n = 0
    labels = []
    for f in files:
        t = pq.read_table(str(f))
        for i, r in enumerate(t.to_pylist()):
            label = clean_label(r["license_number"])
            if len(label) < 3 or not r["xywhr"]:
                continue
            img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
            box = r["xywhr"]
            x0, y0, x1, y1 = crop_from_xywhr(img, box)
            if x1 - x0 < 20 or y1 - y0 < 8:
                continue
            crop = img.crop((x0, y0, x1, y1))
            name = f"{prefix}_{f.stem}_{i:04d}.png"
            crop.save(out_dir / name)
            labels.append(f"{name}\t{label}")
            n += 1
    existing = (out_dir / "labels.txt")
    if existing.exists():
        labels = existing.read_text().splitlines() + labels
    (out_dir / "labels.txt").write_text("\n".join(labels))
    return n


def main() -> None:
    for split in ("train", "val", "test"):
        (OUT / split).mkdir(parents=True, exist_ok=True)

    # ---- training corpus: A-train + B-train
    n_a = extract_sonnet("train", OUT / "train")
    import glob
    tw_train = sorted(glob.glob(str(ROOT / "taiwan/train-*.parquet")))
    n_b = extract_taiwan([Path(p) for p in tw_train], "b", OUT / "train")
    print(f"train: {n_a} (US/EU) + {n_b} (Taiwan) = {n_a + n_b}")

    # ---- validation: A-val only (source-separated from test)
    n_v = extract_sonnet("validation", OUT / "val", limit=800)
    print(f"val: {n_v} (US/EU validation split)")

    # ---- held-out test: A-test + B-test — written once
    n_ta = extract_sonnet("test", OUT / "test")
    tw_test = sorted(glob.glob(str(ROOT / "taiwan_test/test-*.parquet")))
    n_tb = extract_taiwan([Path(p) for p in tw_test], "bt", OUT / "test")
    print(f"test (HELD OUT): {n_ta} (US/EU) + {n_tb} (Taiwan) = {n_ta + n_tb}")


if __name__ == "__main__":
    main()
