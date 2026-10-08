"""Ensemble evaluation: combine per-model predictions via several strategies,
select on Indian val, then evaluate once on both held-out test sets.

Strategies:
  1. conf-weighted-char-vote: per character position, vote weighted by model conf
  2. majority-vote: most common full-string prediction (ties -> highest mean conf)
  3. conf-weighted full-string: pick the string with highest sum of confidences
     among models that predicted it
  4. rank-vote (Borda): full-string Borda count
  5. best-single (baseline reference, not an ensemble)

NOTE on cross-domain models: inA1/inA2 are Indian-specialized, r3/inB1_sub are
international-capable. All four are included per instruction; we also measure
an int'l-only pair (r3+inB1_sub) as a diagnostic.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

RESULTS = Path("/Users/paarth_mendiratta/Desktop/VANTRA/vantra/results")
MODELS = ["r3", "inA1", "inA2", "inB1_sub"]


def load(set_name, model):
    return json.loads((RESULTS / f"predictions_{set_name}_{model}.json").read_text())


def char_vote_weighted(rows_by_model, conf_power=1.0):
    """rows_by_model: list of (pred, conf) per model. Positional weighted vote."""
    max_len = max(len(p) for p, _ in rows_by_model)
    out = []
    for pos in range(max_len):
        votes = defaultdict(float)
        for pred, conf in rows_by_model:
            if pos < len(pred):
                votes[pred[pos]] += conf ** conf_power
        if votes:
            best = max(votes.items(), key=lambda kv: kv[1])
            out.append(best[0])
    return "".join(out)


def majority_vote(rows_by_model):
    strings = Counter(p for p, _ in rows_by_model)
    top = strings.most_common()
    if not top:
        return ""
    best_count = top[0][1]
    finalists = [s for s, c in top if c == best_count]
    if len(finalists) == 1:
        return finalists[0]
    # tie-break by mean conf
    def mean_conf(s):
        cs = [c for p, c in rows_by_model if p == s]
        return sum(cs) / len(cs) if cs else 0.0
    return max(finalists, key=mean_conf)


def conf_weighted_string(rows_by_model):
    scores = defaultdict(float)
    for pred, conf in rows_by_model:
        scores[pred] += conf
    return max(scores.items(), key=lambda kv: kv[1])[0] if scores else ""


def borda(rows_by_model):
    n = len(rows_by_model)
    scores = defaultdict(float)
    for i, (pred, conf) in enumerate(sorted(rows_by_model, key=lambda x: -x[1])):
        scores[pred] += (n - i)  # highest conf gets most points
    return max(scores.items(), key=lambda kv: kv[1])[0] if scores else ""


STRATEGIES = {
    "char_vote_conf": lambda rows: char_vote_weighted(rows),
    "char_vote_conf2": lambda rows: char_vote_weighted(rows, conf_power=2.0),
    "majority_vote": majority_vote,
    "conf_weighted_string": conf_weighted_string,
    "borda": borda,
}


def evaluate(set_name, combine, models=MODELS):
    data = {m: load(set_name, m) for m in models}
    n = len(data[models[0]])
    targets = [r["target"] for r in data[models[0]]]
    exact = 0
    char_correct = 0
    n_chars = 0
    for i in range(n):
        rows = [(data[m][i]["pred"], data[m][i]["conf"]) for m in models]
        pred = combine(rows)
        tgt = targets[i]
        if pred == tgt:
            exact += 1
        # levenshtein char accuracy
        a, b = pred, tgt
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for ii, ca in enumerate(a, 1):
            cur = [ii]
            for jj, cb in enumerate(b, 1):
                cur.append(min(prev[jj] + 1, cur[jj - 1] + 1, prev[jj - 1] + (ca != cb)))
            prev = cur
        d = prev[-1]
        char_correct += max(0, len(tgt) - d)
        n_chars += len(tgt)
    return {"n": n, "exact_pct": round(100 * exact / n, 1),
            "char_pct": round(100 * char_correct / max(n_chars, 1), 1)}


def main() -> None:
    print("=== Strategy selection on Indian val (n=100, 47 plates) ===")
    val_results = {}
    for name, fn in STRATEGIES.items():
        r = evaluate("indian_val", fn)
        val_results[name] = r
        print(f"  {name}: exact {r['exact_pct']}% char {r['char_pct']}%")
    # singles on val for reference
    for m in MODELS:
        d = load("indian_val", m)
        ex = sum(r["pred"] == r["target"] for r in d)
        print(f"  [single {m}]: exact {100*ex/len(d):.1f}%")

    best_strategy = max(val_results, key=lambda k: (val_results[k]["exact_pct"], val_results[k]["char_pct"]))
    print(f"\nbest strategy on val: {best_strategy}")

    print("\n=== One-shot held-out evaluation ===")
    out = {"val_selection": {"strategy": best_strategy, "results": val_results},
           "test": {}}
    for set_name in ("indian_test", "intl_test"):
        for name, fn in STRATEGIES.items():
            r = evaluate(set_name, fn)
            out["test"][f"{set_name}/{name}"] = r
            print(f"  {set_name} {name}: exact {r['exact_pct']}% char {r['char_pct']}%")
        # singles for comparison
        for m in MODELS:
            d = load(set_name, m)
            ex = sum(r["pred"] == r["target"] for r in d)
            ch = evaluate_single_char(d)
            out["test"][f"{set_name}/single_{m}"] = {"n": len(d), "exact_pct": round(100*ex/len(d), 1), "char_pct": ch}
            print(f"  {set_name} [single {m}]: exact {100*ex/len(d):.1f}% char {ch}%")

    (RESULTS / "ensemble_eval.json").write_text(json.dumps(out, indent=1))
    print(f"\nresults -> {RESULTS / 'ensemble_eval.json'}")


def evaluate_single_char(d):
    char_correct = n_chars = 0
    for r in d:
        a, b = r["pred"], r["target"]
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for ii, ca in enumerate(a, 1):
            cur = [ii]
            for jj, cb in enumerate(b, 1):
                cur.append(min(prev[jj] + 1, cur[jj - 1] + 1, prev[jj - 1] + (ca != cb)))
            prev = cur
        char_correct += max(0, len(r["target"]) - prev[-1])
        n_chars += len(r["target"])
    return round(100 * char_correct / max(n_chars, 1), 1)


if __name__ == "__main__":
    main()
