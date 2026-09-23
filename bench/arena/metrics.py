"""Pure scoring for the arena: no model, no I/O, no numpy.

One scored unit is a (item, question) pair:
  gold = {"label": str, "probs": {opt: p} | None, "score": float | None}
  pred = {"label": str | None, "probs": {opt: p} | None, "score": float | None}
`options` is the ordered option list of that question (noul: ["yes", "no"],
score: ["0", "1", ...]). Hard gold has probs None; typed-decisions gold is soft.

Every function takes plain lists so it can be unit-tested in isolation.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the quorum package root

EPS = 1e-6


# ---------------------------------------------------------------------------
# label metrics
# ---------------------------------------------------------------------------

def accuracy(golds: list[str], preds: list[str | None]) -> float:
    if not golds:
        return float("nan")
    return sum(1 for g, p in zip(golds, preds) if p is not None and g == p) / len(golds)


def majority_floor(golds: list[str]) -> float:
    """Accuracy of always answering the most common gold label."""
    if not golds:
        return float("nan")
    return Counter(golds).most_common(1)[0][1] / len(golds)


def macro_f1(golds: list[str], preds: list[str | None], labels: list[str] | None = None) -> float:
    """Unweighted mean F1 over labels that occur in gold (a label never in gold
    and never predicted contributes nothing; a missing prediction counts wrong)."""
    labels = labels or sorted(set(golds))
    f1s = []
    for lab in labels:
        tp = sum(1 for g, p in zip(golds, preds) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(golds, preds) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(golds, preds) if g == lab and p != lab)
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / len(f1s) if f1s else float("nan")


# ---------------------------------------------------------------------------
# distribution metrics
# ---------------------------------------------------------------------------

def normalize(probs: dict[str, float] | None, options: list[str]) -> dict[str, float] | None:
    """Restrict to `options`, renormalise; None if no usable mass."""
    if not probs:
        return None
    v = {o: max(0.0, float(probs.get(o, 0.0) or 0.0)) for o in options}
    s = sum(v.values())
    if s <= 0:
        return None
    return {o: x / s for o, x in v.items()}


def onehot(label: str, options: list[str]) -> dict[str, float]:
    return {o: 1.0 if o == label else 0.0 for o in options}


def brier(gold: dict[str, float], pred: dict[str, float], options: list[str]) -> float:
    """Multi-class Brier: sum over options of (pred - gold)^2. Range [0, 2]."""
    return sum((pred.get(o, 0.0) - gold.get(o, 0.0)) ** 2 for o in options)


def kl(gold: dict[str, float], pred: dict[str, float], options: list[str], eps: float = EPS) -> float:
    """KL(gold || pred), nats. pred is floored at eps and renormalised so a
    confident miss costs a large but finite amount."""
    q = {o: max(pred.get(o, 0.0), eps) for o in options}
    s = sum(q.values())
    total = 0.0
    for o in options:
        g = gold.get(o, 0.0)
        if g > 0:
            total += g * math.log(g / (q[o] / s))
    return total


def tv(gold: dict[str, float], pred: dict[str, float], options: list[str]) -> float:
    """Total variation distance, [0, 1]."""
    return 0.5 * sum(abs(pred.get(o, 0.0) - gold.get(o, 0.0)) for o in options)


def soft_agreement(gold: dict[str, float], pred: dict[str, float], options: list[str]) -> float:
    """Expected agreement sum_k pred_k * gold_k: the chance a draw from the
    system matches a draw from the teacher. (Our definition; the typed-decisions
    card's 'soft accuracy' is not specified precisely enough to copy.)"""
    return sum(pred.get(o, 0.0) * gold.get(o, 0.0) for o in options)


def ece(confidences: list[float], correct: list[bool], n_bins: int = 10) -> float:
    """Top-label expected calibration error with equal-width bins."""
    n = len(confidences)
    if n == 0:
        return float("nan")
    bins: list[list[int]] = [[] for _ in range(n_bins)]
    for i, c in enumerate(confidences):
        b = min(n_bins - 1, max(0, int(c * n_bins)))
        bins[b].append(i)
    total = 0.0
    for idx in bins:
        if not idx:
            continue
        conf = sum(confidences[i] for i in idx) / len(idx)
        acc = sum(1 for i in idx if correct[i]) / len(idx)
        total += len(idx) / n * abs(conf - acc)
    return total


# ---------------------------------------------------------------------------
# binary ranking / selective prediction
# ---------------------------------------------------------------------------

def auc(scores: list[float], positives: list[bool]) -> float:
    """ROC AUC via Mann-Whitney U with average ranks for ties.
    nan when one class is absent."""
    n_pos = sum(1 for p in positives if p)
    n_neg = len(positives) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    r_pos = sum(r for r, p in zip(ranks, positives) if p)
    return (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def coverage_at_risk(confidences: list[float], correct: list[bool], max_err: float = 0.05) -> float:
    """Largest fraction of items you can answer, most-confident first, while
    the error rate among answered items stays <= max_err. Ties in confidence
    are admitted together (you cannot split a tie without extra information)."""
    n = len(confidences)
    if n == 0:
        return float("nan")
    order = sorted(range(n), key=lambda i: -confidences[i])
    best = 0
    wrong = 0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and confidences[order[j + 1]] == confidences[order[i]]:
            j += 1
        wrong += sum(1 for k in range(i, j + 1) if not correct[order[k]])
        answered = j + 1
        if wrong / answered <= max_err:
            best = answered
        i = j + 1
    return best / n


# ---------------------------------------------------------------------------
# score (ordinal) metrics
# ---------------------------------------------------------------------------

def expected_score(probs: dict[str, float] | None, fallback: float | None = None) -> float | None:
    if probs:
        s = sum(probs.values())
        if s > 0:
            return sum(int(k) * v for k, v in probs.items()) / s
    return fallback


def mae(a: list[float], b: list[float]) -> float:
    if not a:
        return float("nan")
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def within_one(gold_levels: list[int], pred_levels: list[int]) -> float:
    if not gold_levels:
        return float("nan")
    return sum(1 for g, p in zip(gold_levels, pred_levels) if abs(g - p) <= 1) / len(gold_levels)


# ---------------------------------------------------------------------------
# uncertainty on the estimate itself
# ---------------------------------------------------------------------------

def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI for the mean of `values` (e.g. per-item 0/1 correctness)."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(values[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    lo = means[int(alpha / 2 * n)]
    hi = means[min(n - 1, int((1 - alpha / 2) * n))]
    return (lo, hi)


def paired_bootstrap(a: list[float], b: list[float], n: int = 2000, seed: int = 0) -> dict:
    """Paired comparison of two systems scored on the SAME items.
    Returns mean(a - b), its 95% CI, and the fraction of resamples where a > b."""
    assert len(a) == len(b), "paired bootstrap needs the same items"
    if not a:
        return {"diff": float("nan"), "ci": (float("nan"), float("nan")), "p_a_better": float("nan")}
    rng = random.Random(seed)
    k = len(a)
    d = [x - y for x, y in zip(a, b)]
    stats = []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        stats.append(sum(d[i] for i in idx) / k)
    stats.sort()
    return {"diff": sum(d) / k,
            "ci": (stats[int(0.025 * n)], stats[min(n - 1, int(0.975 * n))]),
            "p_a_better": sum(1 for s in stats if s > 0) / n}


# ---------------------------------------------------------------------------
# calibration maps (fit on labelled data, applied to raw probabilities)
# ---------------------------------------------------------------------------

# Platt scaling lives in quorum.core (the server applies it); one copy only.
from quorum.core import _logit, _platt_loss, platt_apply, platt_fit, temper  # noqa: E402,F401


def temperature_apply(probs: dict[str, float], t: float) -> dict[str, float]:
    """softmax(log p / t): the server's own function, so scores match what it serves."""
    return temper(probs, t)


def temperature_fit(dists: list[dict[str, float]], golds: list[str],
                    grid: tuple[float, ...] = tuple(0.25 * 1.15 ** i for i in range(40))) -> float:
    """Grid-search the single temperature minimising NLL of the gold label."""
    if not dists:
        return 1.0
    best_t, best_nll = 1.0, float("inf")
    for t in (1.0,) + grid:
        nll = 0.0
        for d, g in zip(dists, golds):
            nll -= math.log(max(temperature_apply(d, t).get(g, 0.0), EPS))
        if nll < best_nll - 1e-12:
            best_t, best_nll = t, nll
    return best_t


def kfold_indices(n: int, k: int = 5, seed: int = 0) -> list[list[int]]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return [idx[i::k] for i in range(k)]
