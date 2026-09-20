#!/usr/bin/env python3
"""Fit quorum's calibration against the cloud-Jev teacher + gold labels.

Reads teacher_run.jsonl ({cloud, local_direct, local_cot, label}) and fits,
per local mode, a Platt logistic map  p' = sigmoid(a*logit(p) + b)  on raw
local probs. Two targets, both reported:
  --target cloud : match cloud Jev's probability (distillation / calibration
                   to a trusted scale; this is the RLCD-style teacher signal)
  --target gold  : match the 0/1 label (classic Platt scaling / log-loss fit)

Reports before/after: log-loss + Brier vs cloud, ECE + accuracy vs gold,
and the 0.5-decision accuracy. Emits quorum-loadable calibration JSON.

Held-out honest estimate: 5-fold CV over items (fit on 4, score the 5th).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

EPS = 1e-6


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def platt_fit(pairs, iters=500, lr=0.5):
    """pairs: [(p_raw, y in {0,1})] -> (a, b) minimising log-loss (gradient descent)."""
    if len(pairs) < 8:
        return 1.0, 0.0
    x = [logit(p) for p, _ in pairs]
    y = [t for _, t in pairs]
    a, b = 1.0, 0.0
    n = len(pairs)
    for _ in range(iters):
        ga = gb = 0.0
        for xi, yi in zip(x, y):
            d = sigmoid(a * xi + b) - yi
            ga += d * xi
            gb += d
        a -= lr * ga / n
        b -= lr * gb / n
    return a, b


def platt(p, ab):
    a, b = ab
    return sigmoid(a * logit(p) + b)


def logloss(ps, ts):
    return -sum(t * math.log(max(p, EPS)) + (1 - t) * math.log(max(1 - p, EPS))
                for p, t in zip(ps, ts)) / len(ps)


def brier(ps, ts):
    return sum((p - t) ** 2 for p, t in zip(ps, ts)) / len(ps)


def ece(ps, ts, bins=10):
    buckets = defaultdict(list)
    for p, t in zip(ps, ts):
        buckets[min(int(p * bins), bins - 1)].append((p, t))
    e = 0.0
    for b in buckets.values():
        e += abs(sum(p for p, _ in b) - sum(t for _, t in b)) / len(ps)
    return e


def _num(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def cv_fit_eval(rows, key, target):
    """5-fold CV: return (fitted_on_all, mean metrics raw vs fitted)."""
    tgt = "cloud" if target == "cloud" else "label"
    rows = [dict(r, **{tgt: _num(r.get(tgt))}) for r in rows]
    n = len(rows)
    idx = list(range(n))
    folds = [idx[i::5] for i in range(5)]
    raw_ll = fit_ll = raw_br = fit_br = 0.0
    raw_ece = fit_ece = 0.0
    raw_acc = fit_acc = tot = 0
    for f in folds:
        test = {rows[i]["id"] for i in f}
        train = [r for r in rows if r["id"] not in test]
        te = [r for r in rows if r["id"] in test]
        pairs = [(r[key], 1 if r[tgt] >= 0.5 else 0) for r in train
                 if r.get(key) is not None and r.get(tgt) is not None]
        ab = platt_fit(pairs)
        for r in te:
            p, t = r[key], (r["cloud"] if target == "cloud" else r["label"])
            if p is None or t is None:
                continue
            q = platt(p, ab)
            tot += 1
            raw_ll += logloss([p], [t]); fit_ll += logloss([q], [t])
            raw_br += brier([p], [t]); fit_br += brier([q], [t])
            raw_ece += ece([p], [t]); fit_ece += ece([q], [t])
            raw_acc += (p >= 0.5) == (t >= 0.5); fit_acc += (q >= 0.5) == (t >= 0.5)
    allp = [(r[key], 1 if r["cloud" if target == "cloud" else "label"] >= 0.5 else 0)
            for r in rows if r.get(key) is not None]
    return platt_fit(allp), dict(
        n=tot, raw_logloss=raw_ll / tot, fit_logloss=fit_ll / tot,
        raw_brier=raw_br / tot, fit_brier=fit_br / tot,
        raw_ece=raw_ece / tot, fit_ece=fit_ece / tot,
        raw_acc=raw_acc / tot, fit_acc=fit_acc / tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="teacher_run.jsonl")
    ap.add_argument("--out", default="calibration_teacher.json")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.run)]
    rows = [r for r in rows if r.get("cloud") is not None]
    gold = [r for r in rows if r.get("label") is not None]
    print(f"{len(rows)} rows with cloud probs")

    report = {"n": len(rows), "models": {}}
    for key in ("local_direct", "local_cot"):
        for target in ("cloud", "gold"):
            pool = rows if target == "cloud" else gold
            pool = [r for r in pool if r.get(key) is not None]
            ab, m = cv_fit_eval(pool, key, target)
            name = f"{key}->{target}"
            report["models"][name] = {"platt_a": round(ab[0], 4),
                                      "platt_b": round(ab[1], 4), **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}}
            print(f"\n== {name} (n={m['n']}) a={ab[0]:.3f} b={ab[1]:.3f}")
            print(f"  logloss {m['raw_logloss']:.3f} -> {m['fit_logloss']:.3f}"
                  f" | brier {m['raw_brier']:.3f} -> {m['fit_brier']:.3f}"
                  f" | ECE {m['raw_ece']:.3f} -> {m['fit_ece']:.3f}"
                  f" | acc@0.5 {m['raw_acc']:.3f} -> {m['fit_acc']:.3f}")

    # quorum-loadable shape (temperature-style file, extended with platt)
    out = {"fitted": a.out, "note": "Platt map vs cloud-Jev teacher (a*logit+b)",
           "models": report["models"]}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
