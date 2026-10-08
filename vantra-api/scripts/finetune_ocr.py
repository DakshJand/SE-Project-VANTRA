"""Fine-tune the EasyOCR CRNN recognizer on real license-plate crops.

Training approach (CTC, matching the pretrained objective):
  - input: grayscale crops resized to height 64, aspect-padded to a batch max width
    (same NormalizePAD scheme easyocr uses at inference)
  - target: label text encoded via the CTCLabelConverter charset (blank=0)
  - loss: torch.nn.CTCLoss over the LSTM output sequence
  - model: the pretrained recognizer, unquantized (quantize=False), full fine-tune
    or head-only depending on --freeze

Hyperparameter sweeps are driven by --lr / --epochs / --freeze flags; each run
saves to models/ft_<tag>/ with a val_metrics.json. Pick the best by val exact.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from easyocr.config import imgH as EASYOCR_IMGH
from easyocr.utils import CTCLabelConverter
from easyocr.recognition import NormalizePAD

from app.config import REPO_ROOT
DATA = REPO_ROOT / "data" / "realworld" / "ft"
MODELS = REPO_ROOT / "data" / "models"


def load_pretrained():
    """Unquantized pretrained recognizer + converter (same charset as inference)."""
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False, verbose=False, quantize=False)
    model = reader.recognizer
    converter = CTCLabelConverter(reader.character)
    return model, converter, reader


def load_labels(split: str) -> list[tuple[Path, str]]:
    out = []
    for line in (DATA / split / "labels.txt").read_text().splitlines():
        name, label = line.split("\t")
        out.append((DATA / split / name, label))
    return out


class PlateDataset(torch.utils.data.Dataset):
    def __init__(self, items, imgH=64, max_w=480):
        self.items = items
        self.imgH = imgH
        self.max_w = max_w

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("L")
        w, h = img.size
        ratio = w / float(h)
        resized_w = min(self.max_w, max(16, math.ceil(self.imgH * ratio)))
        resized = img.resize((resized_w, self.imgH), Image.BICUBIC)
        # normalize: easyocr NormalizePAD expects a PIL image and target (C,H,W)
        transform = NormalizePAD((1, self.imgH, self.max_w))
        tensor = transform(resized)
        return tensor, label


def collate(batch):
    tensors = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    return tensors, labels


def encode_labels(converter, labels, device):
    """CTC encode: concatenated indices + per-label lengths."""
    idxs, lengths = [], []
    for lab in labels:
        seq = [converter.dict[ch] for ch in lab if ch in converter.dict]
        # repeat no chars needed (CTC handles repeats via blank insertion at train
        # time only if duplicates — safest: insert blank between repeated chars)
        enc = []
        prev = None
        for c in seq:
            if c == prev:
                enc.append(0)  # blank separator for repeated chars
            enc.append(c)
            prev = c
        idxs.extend(enc)
        lengths.append(len(enc))
    return (torch.IntTensor(idxs).to(device), torch.IntTensor(lengths).to(device))


@torch.no_grad()
def evaluate(model, converter, items, device, imgH=64, bs=32):
    model.eval()
    exact = 0
    char_correct = 0
    n_chars = 0
    confs_correct, confs_wrong = [], []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        ds = PlateDataset(chunk, imgH=imgH)
        tensors, labels = collate([ds[j] for j in range(len(chunk))])
        tensors = tensors.to(device)
        batch_max_length = tensors.size(3) // 10
        text_for_pred = torch.LongTensor(tensors.size(0), batch_max_length + 1).fill_(0).to(device)
        preds = model(tensors, text_for_pred)  # (B, T, num_class)
        preds = preds.log_softmax(2)
        preds_size = torch.IntTensor([preds.size(1)] * tensors.size(0))
        _, preds_index = preds.max(2)
        # greedy CTC decode
        index = preds_index.cpu().numpy()
        for b in range(tensors.size(0)):
            raw = index[b]
            seq = []
            prev = 0
            for t in raw:
                if t != prev and t != 0:
                    seq.append(t)
                prev = t
            pred_text = "".join(converter.character[c] for c in seq)
            tgt = labels[b]
            if pred_text == tgt:
                exact += 1
            d = lev(pred_text, tgt)
            char_correct += max(0, len(tgt) - d)
            n_chars += len(tgt)
    n = len(items)
    return {"exact": exact, "n": n, "exact_pct": 100 * exact / max(n, 1),
            "char_pct": 100 * char_correct / max(n_chars, 1)}


def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def train(tag: str, lr: float, epochs: int, freeze: str, bs: int = 32,
          val_limit: int = 400, log_every: int = 200, args_subsample: int = 0,
          init_checkpoint: str | None = None):
    device = torch.device("cpu")
    model, converter, _ = load_pretrained()
    if init_checkpoint:
        import torch as _t
        state = _t.load(init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        print(f"resumed from {init_checkpoint}")
    model.train()

    if freeze == "head":
        for p in model.FeatureExtraction.parameters():
            p.requires_grad = False
        for p in model.SequenceModeling.parameters():
            p.requires_grad = False
    elif freeze == "lstm":
        for p in model.FeatureExtraction.parameters():
            p.requires_grad = False

    train_items = load_labels("train")
    rng = random.Random(0)
    rng.shuffle(train_items)
    if args_subsample:
        train_items = train_items[:args_subsample]
    val_items = load_labels("val")[:val_limit]
    print(f"[{tag}] train={len(train_items)} val={len(val_items)} "
          f"lr={lr} epochs={epochs} freeze={freeze}")

    # classes per sample are rare; keep only labels fully covered by charset
    train_items = [(p, l) for p, l in train_items
                   if all(c in converter.dict for c in l)]
    val_items = [(p, l) for p, l in val_items if all(c in converter.dict for c in l)]
    print(f"[{tag}] charset-covered: train={len(train_items)} val={len(val_items)}")

    ds = PlateDataset(train_items)
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True,
                                         collate_fn=collate)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    out_dir = MODELS / f"ft_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    history = []
    step = 0
    t0 = time.time()
    for ep in range(epochs):
        tot_loss = 0.0
        nb = 0
        for tensors, labels in loader:
            tensors = tensors.to(device)
            batch_max_length = tensors.size(3) // 10
            text_for_pred = torch.LongTensor(tensors.size(0), batch_max_length + 1).fill_(0).to(device)
            preds = model(tensors, text_for_pred).log_softmax(2)  # (B, T, C)
            B, T, C = preds.size()
            preds = preds.permute(1, 0, 2)  # (T, B, C) for CTCLoss
            tgt, tgt_len = encode_labels(converter, labels, device)
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
            loss = ctc(preds, tgt, input_lengths, tgt_len)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tot_loss += loss.item()
            nb += 1
            step += 1
            if step % log_every == 0:
                print(f"[{tag}] ep{ep+1} step{step} loss={tot_loss/nb:.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        val = evaluate(model, converter, val_items, device)
        val["epoch"] = ep + 1
        val["train_loss"] = tot_loss / max(nb, 1)
        history.append(val)
        print(f"[{tag}] epoch {ep+1}: val exact {val['exact_pct']:.1f}% "
              f"char {val['char_pct']:.1f}% (loss {val['train_loss']:.3f})", flush=True)
        if val["exact_pct"] >= best_val:
            best_val = val["exact_pct"]
            torch.save(model.state_dict(), out_dir / "recognizer.pt")
    (out_dir / "val_metrics.json").write_text(json.dumps(history, indent=1))
    print(f"[{tag}] DONE best val exact {best_val:.1f}% -> {out_dir}")
    return best_val


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--freeze", choices=["none", "head", "lstm"], default="none")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--init", default=None)
    args = ap.parse_args()
    train(args.tag, args.lr, args.epochs, args.freeze, args.bs,
          args_subsample=args.subsample, init_checkpoint=args.init)
