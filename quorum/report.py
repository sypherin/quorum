"""quorum.report — calibration + confident-error audit over the judgment log.

The README's honest gap is "logprobs good for ranking, not certified
calibrated numbers". This module measures that gap instead of asserting it:
Brier, log-loss and 10-bin ECE per question, plus the two tables that
actually move decisions — reliability (predicted vs observed rate per
confidence bucket) and confident errors (high-prob calls that were wrong).

Borrowed from the kev/Jev eval harnesses: score calibration, not just
accuracy, and always print the majority-class floor next to decision
accuracy so a 95%-one-class question doesn't look like a win.

Input: the serve.py judgment log (or any JSONL with the same shape), where
labeled records carry "labels": {qid: value} (or a legacy single "label"
for one-question records). Unlabeled records are counted and skipped.

Usage:
    python3 -m quorum.report [log/judgments.jsonl] [--qid q] [--min-n 20]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_LOG = Path(__file__).resolve().parent.parent / "log" / "judgments.jsonl"
EPS = 1e-6
BINS = 10


def _num(v):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def prob_of(answer: dict) -> tuple[float, str] | tuple[None, None]:
    """(confidence in the committed answer, the committed value).

    The subject is the answer the model actually committed to (the schema-
    forced enum value), not the argmax of the probability distribution —
    they can disagree, and calibration must describe what the caller sees.
    noul: the contract field is P(yes); the committed value is the answer's
    own yes/no, so confidence is P(yes) or 1-P(yes) accordingly.
    choice: P(chosen). score: P(claimed level)."""
    t = answer.get("type")
    p = answer.get("probabilities")
    if t == "noul":
        pn = answer.get("noul")
        if pn is None:
            return None, None
        pn = float(pn)
        chosen = "no" if str(answer.get("answer", "")).strip().lower() == "no" else "yes"
        conf = pn if chosen == "yes" else 1.0 - pn
        return conf, chosen
    if t == "choice":
        val = answer.get("choice")
        if p is None or val not in p:
            return None, None
        return float(p[val]), str(val)
    if t == "score":
        if not p:
            return None, None
        claimed = str(int(answer.get("score", 0))) if answer.get("score") is not None else None
        if claimed is not None and claimed in p:
            return float(p[claimed]), claimed
        top = max(p.items(), key=lambda kv: kv[1])
        return float(top[1]), str(top[0])
    return None, None


def extract_pairs(records: list[dict]) -> tuple[list[dict], dict]:
    """[(qid, kind, p_chosen, chosen, label_ok)] plus coverage stats.

    label_ok is 1.0/0.0 for categorical, and for score also accepts a graded
    label in [0,1] (self-assessed correctness) so Brier stays meaningful."""
    pairs: list[dict] = []
    stats = {"records": len(records), "labeled_records": 0, "unlabeled": 0,
             "no_prob": 0, "bad_label": 0}
    for r in records:
        labels = r.get("labels")
        if not isinstance(labels, dict):
            labels = {}
            if r.get("label") is not None and len(r.get("answers") or {}) == 1:
                labels = {next(iter(r["answers"])): r["label"]}
        if not labels:
            stats["unlabeled"] += 1
            continue
        stats["labeled_records"] += 1
        for qid, ans in (r.get("answers") or {}).items():
            if qid not in labels:
                continue
            p, chosen = prob_of(ans)
            if p is None:
                stats["no_prob"] += 1
                continue
            lab = labels[qid]
            if ans.get("type") == "score" and _num(lab) is not None:
                v = _num(lab)
                if 0.0 <= v <= 1.0:
                    ok = v
                else:
                    ok = 1.0 if int(v) == int(chosen) else 0.0
            elif isinstance(lab, bool):
                ok = 1.0 if lab == (chosen == "yes") else 0.0
            else:
                ok = 1.0 if str(lab) == chosen else 0.0
            pairs.append({"qid": qid, "kind": ans.get("type", "?"),
                          "p": p, "chosen": chosen, "ok": ok})
    return pairs, stats


def brier(items: list[dict]) -> float:
    return sum((it["p"] - it["ok"]) ** 2 for it in items) / len(items)


def logloss(items: list[dict]) -> float:
    return -sum(it["ok"] * math.log(max(it["p"], EPS))
                + (1 - it["ok"]) * math.log(max(1 - it["p"], EPS))
                for it in items) / len(items)


def ece(items: list[dict], bins: int = BINS) -> float:
    buckets: dict[int, list[dict]] = defaultdict(list)
    for it in items:
        buckets[min(int(it["p"] * bins), bins - 1)].append(it)
    return sum(abs(sum(i["p"] for i in b) - sum(i["ok"] for i in b)) / len(items)
               for b in buckets.values())


def majority_floor(items: list[dict]) -> float:
    """Accuracy of always predicting the more common chosen class — the floor
    decision accuracy must clear to mean anything (kev-harness habit)."""
    if not items:
        return 0.0
    yes = sum(1 for it in items if it["ok"] == 1.0)
    return max(yes, len(items) - yes) / len(items)


def reliability_table(items: list[dict], bins: int = BINS) -> list[tuple]:
    buckets: dict[int, list[dict]] = defaultdict(list)
    for it in items:
        buckets[min(int(it["p"] * bins), bins - 1)].append(it)
    rows = []
    for b in sorted(buckets):
        g = buckets[b]
        rows.append((b / bins, (b + 1) / bins, len(g),
                     sum(i["p"] for i in g) / len(g),
                     sum(i["ok"] for i in g) / len(g)))
    return rows


def confident_errors(items: list[dict], threshold: float = 0.9) -> list[dict]:
    return sorted((it for it in items if it["p"] >= threshold and it["ok"] < 0.5),
                  key=lambda it: -it["p"])


def audit(records: list[dict], qid_filter: str | None = None,
          min_n: int = 1, threshold: float = 0.9) -> dict:
    pairs, stats = extract_pairs(records)
    if qid_filter:
        pairs = [p for p in pairs if p["qid"] == qid_filter]
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        groups[p["qid"]].append(p)
    out = {"stats": stats, "questions": {}}
    for qid, g in sorted(groups.items()):
        if len(g) < min_n:
            continue
        errs = confident_errors(g, threshold)
        out["questions"][qid] = {
            "kind": g[0]["kind"], "n": len(g),
            "brier": round(brier(g), 4), "logloss": round(logloss(g), 4),
            "ece": round(ece(g), 4),
            "acc_at_0.5": round(sum(1 for i in g if (i["p"] >= 0.5) == (i["ok"] >= 0.5)) / len(g), 4),
            "majority_floor": round(majority_floor(g), 4),
            "mean_p": round(sum(i["p"] for i in g) / len(g), 4),
            "base_rate": round(sum(i["ok"] for i in g) / len(g), 4),
            "confident_errors": [
                {"p": round(i["p"], 3), "chosen": i["chosen"]} for i in errs[:10]],
            "n_confident_errors": len(errs),
            "reliability": [[round(v, 4) for v in row] for row in reliability_table(g)],
        }
    return out


def render(rep: dict, threshold: float = 0.9) -> str:
    s = rep["stats"]
    lines = [f"records {s['records']} | labeled {s['labeled_records']} | "
             f"unlabeled {s['unlabeled']} | no-prob {s['no_prob']}"]
    if not rep["questions"]:
        lines.append("no labeled judgments yet — label with: python3 -m quorum.label")
        return "\n".join(lines)
    for qid, q in rep["questions"].items():
        lines.append(f"\n== {qid} ({q['kind']}, n={q['n']})")
        lines.append(f"  brier {q['brier']:.3f}  logloss {q['logloss']:.3f}  "
                     f"ece {q['ece']:.3f}  acc@0.5 {q['acc_at_0.5']:.3f} "
                     f"(floor {q['majority_floor']:.3f})")
        lines.append(f"  mean p {q['mean_p']:.3f} vs base rate {q['base_rate']:.3f}  "
                     f"confident errors (p>={threshold}): {q['n_confident_errors']}")
        lines.append("  bucket   n   mean_p  observed")
        for lo, hi, n, mp, ob in q["reliability"]:
            lines.append(f"  {lo:.1f}-{hi:.1f} {n:4d}   {mp:.3f}    {ob:.3f}")
        for e in q["confident_errors"][:5]:
            lines.append(f"  !! p={e['p']:.3f} chose {e['chosen']!r} but was wrong")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log", nargs="?", default=str(DEFAULT_LOG))
    ap.add_argument("--qid", default=None, help="only this question id")
    ap.add_argument("--min-n", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--json", action="store_true", help="emit machine-readable report")
    args = ap.parse_args()
    records = []
    with open(args.log) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    rep = audit(records, qid_filter=args.qid, min_n=args.min_n,
                threshold=args.threshold)
    if args.json:
        print(json.dumps(rep, indent=1))
    else:
        print(render(rep, threshold=args.threshold))
    return 0


if __name__ == "__main__":
    sys.exit(main())
