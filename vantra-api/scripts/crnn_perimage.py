"""CRNN per-image evaluation on val + test: predictions, confidences, per-image correctness.

Produces the data needed for:
  - VLM overlap analysis (which images each model gets right)
  - fallback-threshold tuning (CRNN confidence -> combined accuracy on val)
  - final combined-pipeline evaluation on test
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from finetune_ocr import (load_pretrained, load_labels, PlateDataset, collate, lev)

WORK = Path("/Users/paarth_mendiratta/Desktop/VANTRA/vantra")
FT = WORK / "data" / "realworld" / "ft"
RESULTS = WORK / "results"
CKPT = WORK / "data" / "models" / "ft_r3_alldata_e6" / "recognizer.pt"


@torch.no_grad()
def predict_all(model, converter, items, device, bs=32):
    model.eval()
    out = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        ds = PlateDataset(chunk)
        tensors, labels = collate([ds[j] for j in range(len(chunk))])
        tensors = tensors.to(device)
        bml = tensors.size(3) // 10
        tfp = torch.LongTensor(tensors.size(0), bml + 1).fill_(0).to(device)
        preds = model(tensors, tfp).log_softmax(2)
        probs = preds.exp()
        _, idx = probs.max(2)
        idx = idx.cpu().numpy()
        maxp = probs.max(2)[0].cpu().numpy()
        for b in range(tensors.size(0)):
            raw = idx[b]
            seq, prev, confs = [], 0, []
            for t, ix in enumerate(raw):
                if ix != prev and ix != 0:
                    seq.append(ix)
                    confs.append(maxp[b][t])
                prev = ix
            text = "".join(converter.character[c] for c in seq)
            conf = float(np.mean(confs)) if confs else 0.0
            out.append((text, conf))
    return out


def eval_split(model, converter, split: str, out_name: str):
    items = load_labels(split)
    items = [(p, l) for p, l in items if all(c in converter.dict for c in l)]
    device = torch.device("cpu")
    preds = predict_all(model, converter, items, device)
    per_image = {}
    exact = 0
    for (path, tgt), (pred, conf) in zip(items, preds):
        d = lev(pred, tgt)
        ok = d == 0
        exact += ok
        per_image[path.name] = {"pred": pred, "target": tgt, "conf": round(conf, 4),
                                "exact": ok, "ed": d}
    res = {"n": len(items), "exact": exact,
           "exact_pct": round(100 * exact / max(len(items), 1), 1),
           "per_image": per_image}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / out_name).write_text(json.dumps(res, indent=1))
    print(f"[{split}] CRNN: {res['exact_pct']}% exact over {res['n']} -> {out_name}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    args = ap.parse_args()
    model, converter, _ = load_pretrained()
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    for s in args.splits:
        eval_split(model, converter, s, f"crnn_{s}_perimage.json")
