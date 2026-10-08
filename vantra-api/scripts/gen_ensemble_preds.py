"""Generate per-model predictions (text + confidence) for ensemble evaluation.

Saves results/predictions_{set}_{model}.json for:
  sets:  indian_val (ft_indian/in_val), indian_test (ft_indian/in_test),
         intl_test (ft/test)
  models: r3, inA1, inA2, inB1_sub
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

WORK = Path("/Users/paarth_mendiratta/Desktop/VANTRA/vantra")
sys.path.insert(0, str(WORK / "vantra-api" / "scripts"))
sys.path.insert(0, str(WORK / "scripts"))

import torch

from finetune_ocr import load_pretrained  # noqa
from eval_final import predict  # noqa
from finetune_ocr_ind import load_labels as load_labels_ind  # noqa
from finetune_ocr import load_labels as load_labels_intl  # noqa

MODELS = {
    "r3": WORK / "data/models/ft_r3_alldata_e6/recognizer.pt",
    "inA1": WORK / "data/models/ft_inA1_lr1e4/recognizer.pt",
    "inA2": WORK / "data/models/ft_inA2_lr5e5/recognizer.pt",
    "inB1_sub": WORK / "data/models/ft_inB1_sub_from_r3/recognizer.pt",
}
IND_DATA = WORK / "data" / "realworld" / "ft_indian"


def load_ind(split):
    out = []
    for line in (IND_DATA / f"in_{split}" / "labels.txt").read_text().splitlines():
        name, label = line.split("\t")
        out.append((IND_DATA / f"in_{split}" / name, label))
    return out


SETS = {
    "indian_val": ("ind", "val"),
    "indian_test": ("ind", "test"),
    "intl_test": ("intl", "test"),
}
RESULTS = WORK / "results"
device = torch.device("cpu")


def load_items(which, split):
    if which == "ind":
        return load_ind(split)
    return load_labels_intl(split)


def main() -> None:
    m0, c0, _ = load_pretrained()
    for set_name, (which, split) in SETS.items():
        items = load_items(which, split)
        items = [(p, l) for p, l in items if all(ch in c0.dict for ch in l)]
        for model_name, ckpt in MODELS.items():
            out_f = RESULTS / f"predictions_{set_name}_{model_name}.json"
            if out_f.exists():
                print(f"skip {out_f.name} (exists)")
                continue
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            m0.load_state_dict(state)
            preds = predict(m0, c0, items, device)
            rows = [{"path": str(p), "target": t, "pred": pr, "conf": round(cf, 4)}
                    for (p, t), (pr, cf) in zip(items, preds)]
            out_f.write_text(json.dumps(rows))
            exact = sum(r["pred"] == r["target"] for r in rows)
            print(f"{set_name}/{model_name}: n={len(rows)} exact={100*exact/len(rows):.1f}%")


if __name__ == "__main__":
    main()
