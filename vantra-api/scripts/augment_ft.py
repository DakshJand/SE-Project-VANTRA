"""Augment the fine-tuning training crops: blur, rotation, brightness/contrast,
partial occlusion — 3 augmented variants per source image, written as extra
training rows (labels preserved: plates remain legible under light augmentation)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

TRAIN = Path("data/realworld/ft/train")
rng = random.Random(4242)


def aug_variants(img: Image.Image) -> list[Image.Image]:
    outs = []
    # 1. motion blur
    b = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.8, 1.8)))
    outs.append(b)
    # 2. brightness/contrast jitter
    c = ImageEnhance.Brightness(img).enhance(rng.uniform(0.55, 1.4))
    c = ImageEnhance.Contrast(c).enhance(rng.uniform(0.7, 1.3))
    outs.append(c)
    # 3. small rotation + occlusion strip
    rot = img.rotate(rng.uniform(-8, 8), expand=False,
                     fillcolor=(110, 110, 110), resample=Image.BICUBIC)
    arr = np.array(rot)
    h, w = arr.shape[:2]
    y = rng.randint(0, max(0, h - 10))
    arr[y:y + rng.randint(4, min(10, h)), :, :] = (92, 74, 48)
    outs.append(Image.fromarray(arr))
    return outs


def main() -> None:
    labels = (TRAIN / "labels.txt").read_text().splitlines()
    new_labels = []
    n = 0
    for line in labels:
        name, label = line.split("\t")
        img = Image.open(TRAIN / name).convert("RGB")
        base = name.rsplit(".", 1)[0]
        for k, v in enumerate(aug_variants(img)):
            an = f"{base}_aug{k}.png"
            v.save(TRAIN / an)
            new_labels.append(f"{an}\t{label}")
            n += 1
    (TRAIN / "labels.txt").write_text("\n".join(labels + new_labels))
    print(f"augmented: {n} new rows -> {len(labels) + n} total training rows")


if __name__ == "__main__":
    main()
