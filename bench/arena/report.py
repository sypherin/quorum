#!/usr/bin/env python3
"""Arena report: every system on the same items, scored the same way.

  report.py [system ...] [--out _report]     (default: every system under _runs/)

Writes <out>.md and <out>.json. The rules, because a benchmark is only as
honest as its denominators:

- Scope is every item of a task. An item with no ok answer (errored, missing,
  never reached) is WRONG for accuracy and scores the uniform distribution on
  the probabilistic metrics. A cell prints its coverage whenever it is below
  100%. A system with no run file for a task shows "-", never a zero.
- jev never sees private items (the gate heldout), so gate is n/a for jev.
- Calibration is fitted under 5-fold cross-validation per (system, task),
  folds split by ITEM so the questions of one item never straddle a fold:
  temperature per question type, and Platt for noul. The raw columns are what
  a system ships today; the CV columns are what it could ship after fitting on
  that much labelled data. Calibration moves probabilities only, never labels.
- Latency is the median ms per item from the run rows. For laya only rows
  tagged meta.load == "stock" count: the memory-mapped load gives the same
  numbers but a slower code path, and untagged rows predate the tag.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import metrics as M  # noqa: E402
from adapters import options_of  # noqa: E402

DATA, RUNS = HERE / "_data", HERE / "_runs"
EPS = M.EPS
K_FOLDS = 5
# Systems scored against each other in the paired tables: every quorum variant
# against every reference, and every variant against the plain baseline.
REFERENCES = ("jev", "laya", "laya-td")
BASELINE = "quorum-direct"


# ---------------------------------------------------------------------------
# units: one (item, question) pair with its gold and each system's prediction
# ---------------------------------------------------------------------------

def uniform(options: list[str]) -> dict[str, float]:
    return {o: 1.0 / len(options) for o in options}


def build_units(items: list[dict]) -> list[dict]:
    units = []
    for it in items:
        for qid, q in it["questions"].items():
            g = it["gold"][qid]
            opts = options_of(q)
            gold_score = g.get("score")
            if gold_score is None and q["type"] == "score":
                gold_score = float(g["label"])
            units.append({"item": it["id"], "qid": qid, "type": q["type"], "options": opts,
                          "gold": g["label"], "gold_probs": g.get("probs"), "gold_score": gold_score,
                          "private": bool(it.get("private"))})
    return units


def read_run(path: Path) -> dict[str, dict]:
    """Last ok row per item id (a retried error is superseded by its success)."""
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("ok"):
            rows[r["id"]] = r
    return rows


def attach(units: list[dict], rows: dict[str, dict]) -> list[dict]:
    """Per unit: {"label", "probs" (never None: uniform when absent), "answered", "has_dist"}."""
    preds = []
    for u in units:
        row = rows.get(u["item"])
        a = (row or {}).get("answers", {}).get(u["qid"]) if row else None
        label = a.get("label") if a else None
        probs = M.normalize(a.get("probs"), u["options"]) if a else None
        preds.append({"label": label, "probs": probs or uniform(u["options"]),
                      "answered": label is not None, "has_dist": probs is not None,
                      "score": a.get("score") if a else None})
    return preds


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def unit_nll(u: dict, probs: dict[str, float]) -> float:
    if u["gold_probs"]:
        return -sum(g * math.log(max(probs.get(o, 0.0), EPS)) for o, g in u["gold_probs"].items() if g > 0)
    return -math.log(max(probs.get(u["gold"], 0.0), EPS))


def gold_dist(u: dict) -> dict[str, float]:
    return u["gold_probs"] or M.onehot(u["gold"], u["options"])


def conf_of(p: dict, probs: dict[str, float]) -> float:
    """Confidence in the label the system committed to (uniform when it gave none)."""
    return probs.get(p["label"], 0.0) if p["label"] is not None else max(probs.values())


def score_probs(units: list[dict], preds: list[dict], probs_list: list[dict]) -> dict:
    """Probabilistic metrics for one probability assignment (raw or calibrated)."""
    n = len(units)
    correct = [p["label"] == u["gold"] for u, p in zip(units, preds)]
    return {
        "nll": sum(unit_nll(u, pr) for u, pr in zip(units, probs_list)) / n,
        "brier": sum(M.brier(gold_dist(u), pr, u["options"]) for u, pr in zip(units, probs_list)) / n,
        "ece": M.ece([conf_of(p, pr) for p, pr in zip(preds, probs_list)], correct),
    }


def score_cell(units: list[dict], preds: list[dict]) -> dict:
    n = len(units)
    golds = [u["gold"] for u in units]
    labels = [p["label"] for p in preds]
    by_q: dict[str, list[int]] = defaultdict(list)
    for i, u in enumerate(units):
        by_q[u["qid"]].append(i)
    out = {
        "n": n,
        "coverage": sum(p["answered"] for p in preds) / n,
        "no_dist": sum(p["answered"] and not p["has_dist"] for p in preds),
        "acc": M.accuracy(golds, labels),
        "macro_f1": statistics.fmean(
            M.macro_f1([golds[i] for i in ix], [labels[i] for i in ix], units[ix[0]]["options"])
            for ix in by_q.values()),
        "floor": sum(M.majority_floor([golds[i] for i in ix]) * len(ix) for ix in by_q.values()) / n,
    }
    out.update(score_probs(units, preds, [p["probs"] for p in preds]))
    soft = [i for i, u in enumerate(units) if u["gold_probs"]]
    if soft:
        out["soft_agree"] = statistics.fmean(M.soft_agreement(units[i]["gold_probs"], preds[i]["probs"], units[i]["options"]) for i in soft)
        out["tv"] = statistics.fmean(M.tv(units[i]["gold_probs"], preds[i]["probs"], units[i]["options"]) for i in soft)
        out["kl"] = statistics.fmean(M.kl(units[i]["gold_probs"], preds[i]["probs"], units[i]["options"]) for i in soft)
    sc = [i for i, u in enumerate(units) if u["type"] == "score"]
    if sc:
        exp = [M.expected_score(preds[i]["probs"]) for i in sc]
        out["score_mae"] = M.mae([units[i]["gold_score"] for i in sc], exp)
    return out


# ---------------------------------------------------------------------------
# calibration under cross-validation
# ---------------------------------------------------------------------------

def fit_maps(units: list[dict], preds: list[dict], idx: list[int], platt: bool) -> dict:
    """Per question type: temperature; for noul also Platt when `platt`.
    Fitted on answered units with a distribution only."""
    maps: dict[str, dict] = {}
    by_type: dict[str, list[int]] = defaultdict(list)
    for i in idx:
        if preds[i]["answered"] and preds[i]["has_dist"]:
            by_type[units[i]["type"]].append(i)
    for t, ix in by_type.items():
        if platt and t == "noul":
            a, b = M.platt_fit([preds[i]["probs"]["yes"] for i in ix], [units[i]["gold"] == "yes" for i in ix])
            maps[t] = {"platt": (a, b)}
        else:
            maps[t] = {"t": M.temperature_fit([preds[i]["probs"] for i in ix], [units[i]["gold"] for i in ix])}
    return maps


def apply_map(u: dict, p: dict, maps: dict) -> dict[str, float]:
    m = maps.get(u["type"])
    if not m or not (p["answered"] and p["has_dist"]):
        return p["probs"]
    if "platt" in m:
        py = M.platt_apply(p["probs"]["yes"], *m["platt"])
        return {"yes": py, "no": 1 - py}
    return M.temperature_apply(p["probs"], m["t"])


def cv_calibrated(units: list[dict], preds: list[dict], platt: bool, k: int = K_FOLDS, seed: int = 0) -> list[dict]:
    items = sorted({u["item"] for u in units})
    folds = M.kfold_indices(len(items), k, seed)
    fold_of = {items[j]: f for f, js in enumerate(folds) for j in js}
    out: list[dict | None] = [None] * len(units)
    for f in range(k):
        train = [i for i, u in enumerate(units) if fold_of[u["item"]] != f]
        maps = fit_maps(units, preds, train, platt)
        for i, u in enumerate(units):
            if fold_of[u["item"]] == f:
                out[i] = apply_map(u, preds[i], maps)
    return out  # type: ignore[return-value]


def calibration_columns(units: list[dict], preds: list[dict]) -> dict:
    if not any(p["has_dist"] for p in preds):
        return {}
    cols = {"cv_temp": score_probs(units, preds, cv_calibrated(units, preds, platt=False))}
    if any(u["type"] == "noul" for u in units):
        cols["cv_platt"] = score_probs(units, preds, cv_calibrated(units, preds, platt=True))
    best = min(cols, key=lambda c: cols[c]["nll"])
    cols["cv_best"] = {**cols[best], "map": best}
    return cols


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def median_ms(rows: dict[str, dict], system: str) -> float | None:
    ms = [r["ms"] for r in rows.values()
          if not system.startswith("laya") or (r.get("meta") or {}).get("load") == "stock"]
    return statistics.median(ms) if ms else None


def evaluate(systems: list[str], tasks: list[str]) -> dict:
    report: dict = {"cells": {}, "pairs": [], "systems": systems, "tasks": tasks}
    correctness: dict[tuple[str, str], list[float]] = {}
    for task in tasks:
        items = [json.loads(l) for l in open(DATA / f"{task}.jsonl")]
        all_units = build_units(items)
        for s in systems:
            path = RUNS / s / f"{task}.jsonl"
            if not path.exists():
                continue
            units = [u for u in all_units if not (s == "jev" and u["private"])]
            if not units:
                report["cells"][f"{s}|{task}"] = {"na": "private items are never sent to the cloud"}
                continue
            rows = read_run(path)
            preds = attach(units, rows)
            cell = score_cell(units, preds)
            cell.update(calibration_columns(units, preds))
            cell["ms"] = median_ms(rows, s)
            report["cells"][f"{s}|{task}"] = cell
            correctness[(s, task)] = {(u["item"], u["qid"]): float(p["label"] == u["gold"])
                                      for u, p in zip(units, preds)}
    quorums = [s for s in systems if s.startswith("quorum")]
    pairs = [(q, r) for q in quorums for r in systems if r in REFERENCES]
    pairs += [(q, BASELINE) for q in quorums if q != BASELINE and BASELINE in systems]
    pairs += [(s, s[:-3]) for s in systems if s.endswith("+sl") and s[:-3] in systems
              and (s, s[:-3]) not in pairs]
    for a, b in pairs:
        pooled_a, pooled_b = [], []
        for task in tasks:
            ca, cb = correctness.get((a, task)), correctness.get((b, task))
            if ca is None or cb is None:
                continue
            keys = sorted(set(ca) & set(cb))
            va, vb = [ca[k] for k in keys], [cb[k] for k in keys]
            pb = M.paired_bootstrap(va, vb)
            report["pairs"].append({"a": a, "b": b, "task": task, "n": len(keys), **pb})
            pooled_a += va
            pooled_b += vb
        if pooled_a:
            report["pairs"].append({"a": a, "b": b, "task": "ALL (pooled units)", "n": len(pooled_a),
                                    **M.paired_bootstrap(pooled_a, pooled_b)})
    return report


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------

def _fmt(v, nd=3):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def _cell(report, s, t, fn):
    c = report["cells"].get(f"{s}|{t}")
    if c is None:
        return "-"
    if "na" in c:
        return "n/a"
    return fn(c)


def to_markdown(report: dict) -> str:
    S, T = report["systems"], report["tasks"]
    lines = []

    def table(title, fn, extra=None):
        head = ["task"] + ([extra[0]] if extra else []) + S
        lines.append(f"### {title}\n")
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "---|" * len(head))
        for t in T:
            row = [t] + ([extra[1](t)] if extra else []) + [_cell(report, s, t, fn) for s in S]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    def acc(c):
        s = _fmt(c["acc"])
        return s + (f" ({c['coverage']:.0%} cov)" if c["coverage"] < 1 else "")

    def floor(t):
        c = next((report["cells"][f"{s}|{t}"] for s in S if f"{s}|{t}" in report["cells"]
                  and "na" not in report["cells"][f"{s}|{t}"]), None)
        return _fmt(c["floor"]) if c else "-"

    table("Accuracy (unanswered = wrong)", acc, ("majority floor", floor))
    table("Macro-F1", lambda c: _fmt(c["macro_f1"]))
    table("NLL raw -> best 5-fold CV map (lower is better)",
          lambda c: f"{_fmt(c['nll'])} -> {_fmt(c['cv_best']['nll'])} ({c['cv_best']['map'][3:]})" if "cv_best" in c else _fmt(c["nll"]))
    table("ECE raw -> best CV map", lambda c: f"{_fmt(c['ece'])} -> {_fmt(c['cv_best']['ece'])}" if "cv_best" in c else _fmt(c["ece"]))
    table("Median ms per item", lambda c: _fmt(c["ms"], 0))

    soft = [t for t in T if any("soft_agree" in report["cells"].get(f"{s}|{t}", {}) for s in S)]
    if soft:
        lines.append("### Soft-gold tasks (teacher distributions)\n")
        lines.append("| task | system | soft agreement | TV | KL | score MAE |")
        lines.append("|---|---|---|---|---|---|")
        for t in soft:
            for s in S:
                c = report["cells"].get(f"{s}|{t}")
                if c and "soft_agree" in c:
                    lines.append(f"| {t} | {s} | {_fmt(c['soft_agree'])} | {_fmt(c['tv'])} | {_fmt(c['kl'])} | {_fmt(c.get('score_mae'))} |")
        lines.append("")

    if report["pairs"]:
        lines.append("### Paired bootstrap on accuracy (a - b, 95% CI, P(a better))\n")
        lines.append("| a | b | task | n | diff | 95% CI | P(a>b) |")
        lines.append("|---|---|---|---|---|---|---|")
        for p in report["pairs"]:
            lines.append(f"| {p['a']} | {p['b']} | {p['task']} | {p['n']} | {p['diff']:+.3f} | "
                         f"[{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}] | {p['p_a_better']:.2f} |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("systems", nargs="*")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--out", default=str(HERE / "_report"))
    args = ap.parse_args()
    systems = args.systems or sorted(p.name for p in RUNS.iterdir() if p.is_dir())
    tasks = [t for t in args.tasks.split(",") if t] or sorted(p.stem for p in DATA.glob("*.jsonl"))
    report = evaluate(systems, tasks)
    Path(args.out + ".json").write_text(json.dumps(report, indent=1, default=str))
    md = to_markdown(report)
    Path(args.out + ".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
