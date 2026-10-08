"""Final held-out evaluation of fine-tuned OCR.

Evaluates on data/realworld/ft/test (NEVER touched during training):
  - pretrained EasyOCR baseline
  - fine-tuned checkpoint(s)
  - with/without Indian-format postprocessing (only applies to 10-char reads)
Reports exact match, char accuracy, and an error breakdown by condition:
  - crop size (small/medium/large)
  - brightness (dark/normal/bright) 
  - aspect ratio (squished/normal)
  - per-source (US/EU vs Taiwan)
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

WORK = Path("/Users/paarth_mendiratta/Desktop/VANTRA/vantra")
sys.path.insert(0, str(WORK / "scripts"))
from finetune_ocr import load_pretrained, load_labels, evaluate, lev  # noqa

import numpy as np
import torch
from PIL import Image

TEST = WORK / "data" / "realworld" / "ft" / "test"
RESULTS = WORK / "results"


_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "T": "7",
             "Z": "2", "S": "5", "B": "8", "G": "6", "A": "4", "U": "0"}
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G",
              "4": "A", "7": "T"}


def indian_postprocess(text):
    if len(text) != 10:
        return text
    out = list(text)
    for i in (2, 3, 6, 7, 8, 9):
        out[i] = _TO_DIGIT.get(out[i], out[i])
    for i in (0, 1, 4, 5):
        out[i] = _TO_LETTER.get(out[i], out[i])
    return "".join(out)


@torch.no_grad()
def predict(model, converter, items, device, bs=32):
    """Returns list of (pred_text, confidence)."""
    from finetune_ocr import PlateDataset, collate
    model.eval()
    preds_out = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        ds = PlateDataset(chunk)
        tensors, labels = collate([ds[j] for j in range(len(chunk))])
        tensors = tensors.to(device)
        bml = tensors.size(3) // 10
        tfp = torch.LongTensor(tensors.size(0), bml + 1).fill_(0).to(device)
        preds = model(tensors, tfp).log_softmax(2)
        probs = preds.exp()
        # greedy decode + mean confidence over non-blank frames
        _, idx = probs.max(2)
        idx = idx.cpu().numpy()
        maxp = probs.max(2)[0].cpu().numpy()
        for b in range(tensors.size(0)):
            raw = idx[b]
            seq = []
            prev = 0
            frame_confs = []
            for t, ix in enumerate(raw):
                if ix != prev and ix != 0:
                    seq.append(ix)
                    frame_confs.append(maxp[b][t])
                prev = ix
            text = "".join(converter.character[c] for c in seq)
            conf = float(np.mean(frame_confs)) if frame_confs else 0.0
            preds_out.append((text, conf))
    return preds_out


def condition_of(path: Path, label: str) -> dict:
    """Classify a test item into condition buckets for the error breakdown."""
    img = Image.open(path)
    w, h = img.size
    g = np.asarray(img.convert("L"))
    mean_b = float(g.mean())
    # size bucket
    if w < 80:
        size = "small(<80px)"
    elif w < 160:
        size = "medium(80-160px)"
    else:
        size = "large(>160px)"
    # brightness
    if mean_b < 70:
        light = "dark"
    elif mean_b > 170:
        light = "bright"
    else:
        light = "normal"
    # aspect (plate-like is 2.5-5.5)
    ar = w / max(h, 1)
    if ar < 2.0:
        aspect = "squished(<2.0)"
    elif ar > 5.5:
        aspect = "wide(>5.5)"
    else:
        aspect = "normal"
    src = "taiwan" if path.name.startswith("bt_") else "us/eu"
    return {"size": size, "light": light, "aspect": aspect, "source": src}


def run_eval(name, model, converter, items, device, use_pp=False):
    preds = predict(model, converter, items, device)
    n = exact = 0
    char_correct = n_chars = 0
    confs_correct, confs_wrong = [], []
    cond_stats = defaultdict(lambda: {"n": 0, "exact": 0, "chars": 0, "tchars": 0})
    examples = []
    for (path, tgt), (pred_raw, conf) in zip(items, preds):
        pred = indian_postprocess(pred_raw) if use_pp else pred_raw
        n += 1
        conds = condition_of(path, tgt)
        key_s, key_l, key_a, key_src = conds["size"], conds["light"], conds["aspect"], conds["source"]
        for k in (key_s, key_l, key_a, key_src):
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
            if len(examples) < 15:
                examples.append({"target": tgt, "pred": pred, "conf": round(conf, 2),
                                 "source": key_src})
        for k in (key_s, key_l, key_a, key_src):
            if ok:
                cond_stats[k]["exact"] += 1
            cond_stats[k]["chars"] += max(0, len(tgt) - d)
    res = {
        "name": name,
        "n": n,
        "exact": exact,
        "exact_pct": round(100 * exact / max(n, 1), 1),
        "char_pct": round(100 * char_correct / max(n_chars, 1), 1),
        "mean_conf_correct": round(float(np.mean(confs_correct)), 3) if confs_correct else None,
        "mean_conf_wrong": round(float(np.mean(confs_wrong)), 3) if confs_wrong else None,
        "conf_p10_correct": round(float(np.percentile(confs_correct, 10)), 3) if confs_correct else None,
        "by_condition": {k: {"n": v["n"], "exact_pct": round(100 * v["exact"] / max(v["n"], 1), 1),
                             "char_pct": round(100 * v["chars"] / max(v["tchars"], 1), 1)}
                         for k, v in sorted(cond_stats.items())},
        "error_examples": examples,
    }
    return res


def main() -> None:
    device = torch.device("cpu")
    RESULTS.mkdir(exist_ok=True)
    items = load_labels("test")
    from finetune_ocr import load_pretrained as lp

    def load_ckpt(path):
        m, c, _ = lp()
        if path:
            state = torch.load(path, map_location="cpu", weights_only=False)
            m.load_state_dict(state)
        return m, c

    model, converter = load_ckpt(None)
    items = [(p, l) for p, l in items if all(c in converter.dict for c in l)]
    print(f"held-out test items (charset-covered): {len(items)}")

    out = []
    # 1. pretrained baseline
    out.append(run_eval("pretrained", model, converter, items, device))
    print(f"pretrained: exact {out[-1]['exact_pct']}% char {out[-1]['char_pct']}%")

    # 2. fine-tuned checkpoints
    for ckpt_name in ("ft_r4_cont_from_r3/recognizer.pt", "ft_r3_alldata_e6/recognizer.pt"):
        ckpt = WORK / "data" / "models" / ckpt_name
        if ckpt.exists():
            m2, c2 = load_ckpt(str(ckpt))
            out.append(run_eval(f"finetuned-{ckpt_name}", m2, c2, items, device))
            print(f"{ckpt_name}: exact {out[-1]['exact_pct']}% char {out[-1]['char_pct']}%")

    # 3. best + indian postprocessing
    best = max(out, key=lambda r: r["exact_pct"])
    if best["name"].startswith("finetuned"):
        import re as _re
        ckpt = WORK / "data" / "models" / _re.sub(r"^finetuned-", "", best["name"])
        m3, c3 = load_ckpt(str(ckpt))
        out.append(run_eval("finetuned+indian-pp", m3, c3, items, device, use_pp=True))
        print(f"+indian-pp: exact {out[-1]['exact_pct']}% char {out[-1]['char_pct']}%")

    (RESULTS / "heldout_eval.json").write_text(json.dumps(out, indent=1))
    print("results written to", RESULTS / "heldout_eval.json")


if __name__ == "__main__":
    main()
