#!/usr/bin/env python3
"""Monitor inA1/inA2/inB1_sub training; enforce kill rules (user directive, 2026-09-09).

Rules:
  - inA1/inA2: kill individually once val plateau confirmed (>=5 epochs done, last
    epoch did not improve on best); hard kill at epoch 8.
  - inB1_sub: kill at hour 8 from start if not finished (3-epoch schedule).
  - absolute: kill everything at 10h from start.
Best-val checkpoints are saved by the trainer itself; nothing is lost by killing.
"""
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

MODELS = Path("/Users/paarth_mendiratta/Desktop/VANTRA/vantra/data/models")
TAGS = ["inA1_lr1e4", "inA2_lr5e5", "inB1_sub_from_r3"]
A_TAGS = {"inA1_lr1e4", "inA2_lr5e5"}

START = time.time()
HOUR8 = START + 8 * 3600
HOUR10 = START + 10 * 3600


def find_pid(tag):
    out = subprocess.run(["ps", "ax", "-o", "pid,command"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if f"--tag {tag} " in line or line.rstrip().endswith(f"--tag {tag}"):
            m = re.match(r"\s*(\d+)", line)
            if m:
                return int(m.group(1))
    return None


def kill_tag(tag, reason):
    pid = find_pid(tag)
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(3)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            pass
        print(f"{ts()} KILLED {tag} ({reason}) — best-val checkpoint retained", flush=True)
    else:
        print(f"{ts()} {tag}: no live pid found for kill ({reason})", flush=True)


def ts():
    return time.strftime("%H:%M")


def history(tag):
    f = MODELS / f"ft_{tag}" / "val_metrics.json"
    if not f.exists():
        return []
    try:
        return json.loads(f.read_text())
    except json.JSONDecodeError:
        return []


def status(tag):
    h = history(tag)
    if not h:
        return 0, None, None, None
    best = max(v["exact_pct"] for v in h)
    last = h[-1]["exact_pct"]
    prev = h[-2]["exact_pct"] if len(h) > 1 else None
    return len(h), best, last, prev


while True:
    now = time.time()
    if now >= HOUR10:
        for t in TAGS:
            kill_tag(t, "10h absolute deadline")
        print(f"{ts()} 10H DEADLINE REACHED — all training stopped", flush=True)
        break

    any_alive = False
    for t in TAGS:
        pid = find_pid(t)
        if pid is None:
            continue
        any_alive = True
        ep, best, last, prev = status(t)
        print(f"{ts()} {t}: pid {pid}, epochs done {ep}, best val exact {best}%, "
              f"last {last}%", flush=True)

        if t in A_TAGS and ep >= 8:
            kill_tag(t, "epoch-8 hard cap")
            continue

        # plateau: >=5 epochs, last epoch did not improve on running best
        if t in A_TAGS and ep >= 5 and last is not None and best is not None:
            if last < best and prev is not None and prev < best:
                kill_tag(t, f"val plateau (epochs {ep-1}-{ep} below best {best})")
                continue

        if t not in A_TAGS and now >= HOUR8:
            kill_tag(t, "8h budget cap")
            continue

    if not any_alive:
        print(f"{ts()} all training jobs finished", flush=True)
        break
    time.sleep(900)

print(f"MONITOR EXIT {ts()}", flush=True)
