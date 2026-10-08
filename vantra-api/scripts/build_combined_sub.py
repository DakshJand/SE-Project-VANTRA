"""Build the resized combined corpus for config B: full Indian training rows
(5,180 incl. augmentation) + a fixed-seed random sample of 12,000 international
rows, preserving the source/condition mix of the 91,776-row international corpus.
"""
from __future__ import annotations

import random
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "data" / "realworld"
IN_TRAIN = BASE / "ft_indian" / "in_train"
FT_TRAIN = BASE / "ft" / "train"
OUT = BASE / "ft_indian" / "combined_train_sub"

N_INTL = 12_000


def main() -> None:
    OUT.mkdir(exist_ok=True)
    in_rows = (IN_TRAIN / "labels.txt").read_text().splitlines()
    intl_rows = (FT_TRAIN / "labels.txt").read_text().splitlines()
    rng = random.Random(20260909)
    sample = rng.sample(intl_rows, N_INTL)
    rows = [f"../in_train/{n}\t{l}" for n, l in (r.split("\t") for r in in_rows)]
    rows += [f"../../ft/train/{n}\t{l}" for n, l in (r.split("\t") for r in sample)]
    (OUT / "labels.txt").write_text("\n".join(rows))
    print(f"combined_train_sub: {len(rows)} rows "
          f"(Indian {len(in_rows)} + international sample {len(sample)})")


if __name__ == "__main__":
    main()
