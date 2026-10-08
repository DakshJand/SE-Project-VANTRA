"""Leak-free Indian test evaluation (2026-09-09 split rebuild).

Evaluates one or more checkpoints ONCE on the new plate-partitioned Indian
test split (312 images / 152 unique plates, zero train/val/test plate-text
overlap). Writes results/indian_leakfree_eval.json.

Usage:
  python eval_indian_leakfree.py ckpt1 [ckpt2 ...]
  checkpoint paths are relative to data/models/, or "pretrained" for stock.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

WORK = Path("/Users/paarth_mendiratta/Desktop/VANTRA/vantra")
sys.path.insert(0, str(WORK / "scripts"))
sys.path.insert(0, str(WORK / "vantra-api"))

import numpy as np
import torch

from finetune_ocr import load_pretrained, load_labels, PlateDataset, collate, lev  # noqa
from eval_final import predict, condition_of  # noqa

TEST_DIR = WORK / "data" / "realworld" / "ft_indian" / "in_test"
MODELS = WORK / "data" / "models"
RESULTS = WORK / "results"


def run_eval(name, model, converter, items, device):
    preds = predict(model, converter, items, device)
    n = exact = 0
    char_correct = n_chars = 0
    confs_correct, confs_wrong = [], []
    cond_stats = {}
    examples = []
    for (path, tgt), (pred, conf) in zip(items, preds):
        n += 1
        conds = condition_of(path, tgt)
        keys = (conds["size"], conds["light"], conds["aspect"])
        for k in keys:
            cond_stats.setdefault(k, {"n": 0, "exact": 0, "chars": 0, "tchars": 0})
            cond_stats[k]["n"] += 1
            cond_stats[k]["tchars"] += len(tgt)
        d = lev(pred, tgt)
        char_correct += max(0, len(tgt) - d)
        n_chars += len(tgt)
        ok = d == 0
        if ok:
            exact += 1
            confs_correct.append(conf)
        else:
            confs_wrong.append(conf)
            if len(examples) < 20:
                examples.append({"target": tgt, "pred": pred,
                                 "conf": round(conf, 2)})
        for k in keys:
            if ok:
                cond_stats[k]["exact"] += 1
            cond_stats[k]["chars"] += max(0, len(tgt) - d)
    # unique-plate exact (plate-level accuracy, one vote per plate)
    by_plate = {}
    for (path, tgt), (pred, conf) in zip(items, preds):
        by_plate.setdefault(tgt, []).append(pred == tgt)
    plate_acc = 100 * sum(all(v) for v in by_plate.values()) / len(by_plate)
    # Wilson 95% CI on image-level exact
    p, nn = exact / max(n, 1), n
    z = 1.96
    denom = 1 + z * z / nn
    center = (p + z * z / (2 * nn)) / denom
    half = z * ((p * (1 - p) / nn + z * z / (4 * nn * nn)) ** 0.5) / denom
    res = {
        "name": name,
        "n_images": n,
        "n_unique_plates": len(by_plate),
        "exact": exact,
        "exact_pct": round(100 * exact / max(n, 1), 1),
        "exact_pct_ci95": [round(100 * (center - half), 1), round(100 * (center + half), 1)],
        "plate_level_exact_pct": round(plate_acc, 1),
        "char_pct": round(100 * char_correct / max(n_chars, 1), 1),
        "mean_conf_correct": round(float(np.mean(confs_correct)), 3) if confs_correct else None,
        "mean_conf_wrong": round(float(np.mean(confs_wrong)), 3) if confs_wrong else None,
        "by_condition": {k: {"n": v["n"], "exact_pct": round(100 * v["exact"] / max(v["n"], 1), 1),
                             "char_pct": round(100 * v["chars"] / max(v["tchars"], 1), 1)}
                         for k, v in sorted(cond_stats.items())},
        "error_examples": examples,
    }
    return res


def load_ckpt(path):
    m, c, _ = load_pretrained()
    if path:
        state = torch.load(path, map_location="cpu", weights_only=False)
        m.load_state_dict(state)
    return m, c


def load_test_items(converter):
    items = []
    for line in (TEST_DIR / "labels.txt").read_text().splitlines():
        name, label = line.split("\t")
        items.append((TEST_DIR / name, label))
    items = [(p, l) for p, l in items if all(c in converter.dict for c in l)]
    return items


def main() -> None:
    device = torch.device("cpu")
    names = sys.argv[1:] or ["pretrained"]
    m0, c0 = load_ckpt(None)
    items = load_test_items(c0)
    print(f"leak-free Indian test items (charset-covered): {len(items)}")

    out = []
    for name in names:
        if name == "pretrained":
            m, c = m0, c0
        else:
            ckpt = MODELS / name
            if not ckpt.exists():
                print(f"SKIP {name}: not found")
                continue
            m, c = load_ckpt(str(ckpt))
        r = run_eval(name, m, c, items, device)
        out.append(r)
        print(f"{name}: exact {r['exact_pct']}% (CI95 {r['exact_pct_ci95']}) "
              f"char {r['char_pct']}% plate-level {r['plate_level_exact_pct']}%")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "indian_leakfree_eval.json").write_text(json.dumps(out, indent=1))
    print("results written to", RESULTS / "indian_leakfree_eval.json")


if __name__ == "__main__":
    main()
