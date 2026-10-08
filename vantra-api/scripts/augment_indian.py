"""Augment the leak-free Indian training crops (3 variants per image, same
recipe as the international ft corpus) and build the combined Indian+
international training dir for config B.

Outputs:
  data/realworld/ft_indian/in_train/       + *_aug{0,1,2}.png rows in labels.txt
  data/realworld/ft_indian/combined_train/ labels.txt referencing both corpora
    (relative ../in_train/... and ../../ft/train/... paths)
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageEnhance, ImageFilter

import numpy as np

BASE = Path(__file__).resolve().parents[2] / "data" / "realworld"
IN_TRAIN = BASE / "ft_indian" / "in_train"
FT_TRAIN = BASE / "ft" / "train"
COMBINED = BASE / "ft_indian" / "combined_train"

rng = random.Random(9909)


def aug_variants(img: Image.Image) -> list[Image.Image]:
    outs = []
    b = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.8, 1.8)))
    outs.append(b)
    c = ImageEnhance.Brightness(img).enhance(rng.uniform(0.55, 1.4))
    c = ImageEnhance.Contrast(c).enhance(rng.uniform(0.7, 1.3))
    outs.append(c)
    rot = img.rotate(rng.uniform(-8, 8), expand=False,
                     fillcolor=(110, 110, 110), resample=Image.BICUBIC)
    arr = np.array(rot)
    h, w = arr.shape[:2]
    y = rng.randint(0, max(0, h - 10))
    arr[y:y + rng.randint(4, min(10, h)), :, :] = (92, 74, 48)
    outs.append(Image.fromarray(arr))
    return outs


def main() -> None:
    # 1. augment Indian train
    labels = (IN_TRAIN / "labels.txt").read_text().splitlines()
    existing_aug = sum(1 for l in labels if "_aug" in l.split("\t")[0])
    if existing_aug:
        print(f"already augmented ({existing_aug} aug rows) — skipping")
    else:
        new_labels = []
        n = 0
        for line in labels:
            name, label = line.split("\t")
            img = Image.open(IN_TRAIN / name).convert("RGB")
            base = name.rsplit(".", 1)[0]
            for k, v in enumerate(aug_variants(img)):
                an = f"{base}_aug{k}.png"
                v.save(IN_TRAIN / an)
                new_labels.append(f"{an}\t{label}")
                n += 1
        (IN_TRAIN / "labels.txt").write_text("\n".join(labels + new_labels))
        print(f"augmented: {n} new rows -> {len(labels) + n} Indian training rows")

    # 2. combined dir (config B)
    COMBINED.mkdir(exist_ok=True)
    rows = []
    for line in (IN_TRAIN / "labels.txt").read_text().splitlines():
        name, label = line.split("\t")
        rows.append(f"../in_train/{name}\t{label}")
    for line in (FT_TRAIN / "labels.txt").read_text().splitlines():
        name, label = line.split("\t")
        rows.append(f"../../ft/train/{name}\t{label}")
    (COMBINED / "labels.txt").write_text("\n".join(rows))
    print(f"combined_train: {len(rows)} rows "
          f"(Indian {len((IN_TRAIN / 'labels.txt').read_text().splitlines())} + "
          f"international {len((FT_TRAIN / 'labels.txt').read_text().splitlines())})")


if __name__ == "__main__":
    main()
