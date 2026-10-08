"""VLM-fallback evaluation harness.

Compares multimodal models (via the latentcode gateway) on plate reading against
the same splits the CRNN was evaluated on. Prompt selection happens on the
VALIDATION set only; the held-out TEST set is evaluated exactly once per final
configuration.

Models available on latentstack.dev: gemini-3.1-pro, gemini-3.5-flash,
gemini-3.7-flash (+ glm-5.2 text-only, unusable for vision).
Prompt strategies: plain, structured (char-by-char reasoning), ranked-candidates.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path

import requests
from PIL import Image

WORK = Path("/Users/paarth_mendiratta/Desktop/VANTRA/vantra")
FT = WORK / "data" / "realworld" / "ft"
RESULTS = WORK / "results"

GATEWAY = "https://latentstack.dev/v1/chat/completions"
KEY = None
def _key():
    global KEY
    if KEY is None:
        import os
        KEY = os.environ.get("LATENTCODE_KEY") or Path("/tmp/latentcode_key").read_text().strip()
    return KEY

PROMPTS = {
    "plain": "Read the license plate text in this image. Reply with ONLY the plate characters, no spaces or dashes, nothing else.",
    "structured": "This image shows a vehicle license plate crop. Read it carefully character by character, left to right. Then output ONLY the final plate string (alphanumeric characters only, no spaces, dashes, or other text).",
    "ranked": "Read the license plate in this image. Some characters may be ambiguous (0/O, 1/I, 8/B, 5/S, 2/Z). Output your best reading as a single line of alphanumeric characters only — no explanation, no spaces.",
}


def load_labels(split: str, limit: int | None = None) -> list[tuple[Path, str]]:
    out = []
    for line in (FT / split / "labels.txt").read_text().splitlines():
        name, label = line.split("\t")
        out.append((FT / split / name, label))
        if limit and len(out) >= limit:
            break
    return out


def b64(path: Path) -> str:
    # downscale large crops to keep payload sane; upscale tiny ones a bit for VLM legibility
    img = Image.open(path).convert("RGB")
    if img.width > 640:
        img = img.resize((640, int(img.height * 640 / img.width)), Image.LANCZOS)
    elif img.width < 96:
        scale = 96 / img.width
        img = img.resize((96, int(img.height * scale)), Image.LANCZOS)
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def call_vlm(model: str, prompt: str, image_b64: str,
             max_tokens: int = 3000) -> tuple[str, float, float]:
    """Returns (text, latency_s, cost_usd)."""
    t0 = time.time()
    r = requests.post(GATEWAY, headers={
        "Authorization": f"Bearer {_key()}",
        "Content-Type": "application/json",
    }, json={
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        }],
        "max_tokens": max_tokens,
    }, timeout=120)
    dt = time.time() - t0
    r.raise_for_status()
    d = r.json()
    return (d["choices"][0]["message"].get("content") or "",
            dt, d.get("usage", {}).get("cost_usd", 0.0))


def clean(text: str) -> str:
    """Extract the alphanumeric plate reading from a VLM response."""
    t = text.strip()
    # take the last non-empty line (reasoning models may emit prose first)
    lines = [l for l in t.splitlines() if l.strip()]
    if lines:
        t = lines[-1]
    return re.sub(r"[^A-Za-z0-9]", "", t).upper()


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


def run_eval(model: str, prompt_name: str, items, tag: str,
             out_prefix: str) -> dict:
    prompt = PROMPTS[prompt_name]
    exact = 0
    char_correct = 0
    n_chars = 0
    latencies = []
    costs = 0.0
    per_image = {}
    for i, (path, tgt) in enumerate(items):
        try:
            raw, dt, cost = call_vlm(model, prompt, b64(path))
            pred = clean(raw)
            latencies.append(dt)
            costs += cost
        except Exception as e:
            pred = ""
            print(f"  [{tag}] item {i} error: {type(e).__name__}", flush=True)
        d = lev(pred, tgt) if pred else len(tgt)
        ok = d == 0
        exact += ok
        char_correct += max(0, len(tgt) - d)
        n_chars += len(tgt)
        per_image[path.name] = {"pred": pred, "target": tgt, "exact": ok, "ed": d}
        if (i + 1) % 25 == 0:
            print(f"  [{tag}] {i+1}/{len(items)} exact {100*exact/(i+1):.1f}% "
                  f"({np.mean(latencies):.1f}s/img)", flush=True)
    n = len(items)
    res = {
        "model": model, "prompt": prompt_name, "n": n,
        "exact": exact, "exact_pct": round(100 * exact / max(n, 1), 1),
        "char_pct": round(100 * char_correct / max(n_chars, 1), 1),
        "mean_latency_s": round(float(np.mean(latencies)), 2),
        "total_cost_usd": round(costs, 4),
        "per_image": per_image,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{out_prefix}_{model.replace('/', '_')}_{prompt_name}.json").write_text(
        json.dumps(res, indent=1))
    print(f"[{tag}] DONE exact {res['exact_pct']}% char {res['char_pct']}% "
          f"latency {res['mean_latency_s']}s/img cost ${res['total_cost_usd']:.3f}")
    return res


if __name__ == "__main__":
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True, choices=list(PROMPTS))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=150,
                    help="number of images (val default 150 for prompt selection)")
    args = ap.parse_args()
    items = load_labels(args.split, args.limit)
    run_eval(args.model, args.prompt, items, f"{args.split[:3]}",
             out_prefix=f"vlm_{args.split}")
