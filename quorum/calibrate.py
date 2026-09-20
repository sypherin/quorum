"""quorum.calibrate — fit per-question-type temperatures from labeled judgments.

Input: the judgment log itself (serve.py records carrying a `labels` dict,
as written by quorum.label), or the older sidecar format — one labeled
judgment per line:

    {"qid": "intent", "kind": "choice", "probs": {"interested": 0.7, "not_now": 0.3}, "label": "not_now"}

From the log, each labeled question becomes one record: probs from the
answer's `probabilities`, label from `labels[qid]` (resolved to the value
space — graded 0..1 score labels become a soft two-target split and are
skipped if they name no level; unlabeled questions and questions without a
probability are skipped). Grouping is by primitive type.

- "probs" keys are the answer values as quorum returns them (option keys for
  choice, "yes"/"no" for noul, digit strings for score).
- "label" is the ground-truth answer value.
- "kind" ("choice"|"noul"|"score") is optional; missing kinds group under
  "default". core.load_temperatures looks up T by type, then "default",
  then 1.0.

Method: per group, recover logprobs as log(probs) (softmax is invariant to
the additive constant), then minimize NLL of the true label over temperature
T with scipy.optimize.minimize_scalar (bounded). Output: calibration.json
with {"temperatures": {<group>: T, ...}} — serve.py picks it up
automatically from the package directory (or $QUORUM_CALIBRATION). Ships
with no file => T=1 everywhere.

Usage:
    python3 -m quorum.calibrate labeled.jsonl [-o calibration.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from scipy.optimize import minimize_scalar

T_BOUNDS = (0.05, 20.0)
MIN_N = 30          # question types with fewer labeled samples are omitted
BOUND_SLACK = 1.01  # a fit within 1% of a bound edge counts as a bound-hit
# A weighted NLL below this many nats per unit weight means the soft-target
# weights (graded score labels split into two records summing < 1) collapsed
# the metric — no real fit reaches it (0.9-prob correct labels still cost
# 0.105 nats/sample; this catches the ~1e-18 collapse).
COLLAPSE_NLL_PER_SAMPLE = 1e-6


def load_records(path: str) -> list[dict]:
    """Accepts the judgment log (records with `answers` + `labels`) and/or the
    sidecar format (lines with `probs` + `label`). Log lines are expanded into
    one sidecar record per labeled question."""
    records = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "probs" in rec and "label" in rec:
                records.append(rec)
                continue
            if rec.get("answers"):
                records.extend(expand_log_record(rec, lineno))
                continue
            raise ValueError(f"line {lineno}: needs 'probs'+'label' or 'answers'+'labels'")
    if not records:
        raise ValueError("no labeled records found")
    return records


def expand_log_record(rec: dict, lineno: int = 0) -> list[dict]:
    out = []
    labels = rec.get("labels")
    if not isinstance(labels, dict):
        return out
    for qid, ans in (rec.get("answers") or {}).items():
        if qid not in labels:
            continue
        probs = ans.get("probabilities")
        if not probs:
            continue
        kind = ans.get("type")
        if kind not in ("noul", "choice", "score"):
            continue
        label = labels[qid]
        if kind == "noul":
            lab = str(label).strip().lower()
            if lab in ("true", "1", "1.0"):
                lab = "yes"
            elif lab in ("false", "0", "0.0"):
                lab = "no"
            if lab not in probs:
                continue
            out.append({"qid": qid, "kind": kind, "probs": probs, "label": lab})
        elif kind == "choice":
            if str(label) not in probs:
                continue
            out.append({"qid": qid, "kind": kind, "probs": probs, "label": str(label)})
        else:  # score: exact level, or graded 0..1 correctness
            lab = label
            if isinstance(lab, (int, float)) and not float(lab).is_integer():
                g = float(lab)
                claimed = str(int(round(float(ans.get("score", 0)))))
                other = next((k for k in probs if k != claimed), None)
                if other is None:
                    continue
                out.append({"qid": qid, "kind": kind, "probs": probs,
                            "label": claimed, "label_weight": g})
                out.append({"qid": qid, "kind": kind, "probs": probs,
                            "label": other, "label_weight": 1.0 - g})
                continue
            lab = str(int(float(lab))) if not isinstance(lab, str) else lab
            if lab not in probs:
                continue
            out.append({"qid": qid, "kind": kind, "probs": probs, "label": lab})
    return out


def nll_for_temperature(T: float, group: list[dict]) -> float:
    """Negative log-likelihood of labels after temperature scaling.
    Records may carry `label_weight` (graded score labels split into two
    soft-target records); default weight 1."""
    total = 0.0
    for rec in group:
        logps = {k: math.log(max(float(p), 1e-12)) for k, p in rec["probs"].items()}
        scaled = {k: v / T for k, v in logps.items()}
        m = max(scaled.values())
        log_z = m + math.log(sum(math.exp(v - m) for v in scaled.values()))
        label = str(rec["label"])
        if label not in scaled:
            raise ValueError(f"label {label!r} not among probs keys {sorted(scaled)}")
        total += rec.get("label_weight", 1.0) * (log_z - scaled[label])
    return total


def fit_temperature(group: list[dict]) -> float:
    res = minimize_scalar(
        lambda T: nll_for_temperature(T, group),
        bounds=T_BOUNDS,
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(res.x)


def fit_guarded(name: str, group: list[dict]) -> tuple[float | None, str]:
    """Fit T for one question type, refusing degenerate fits.

    Returns (T, note). T is None — and a loud warning is printed — when the
    group is below MIN_N samples, the fitted T sits on a bound (the model
    wants to run away: the score fit at low n pins to 0.05), or the NLL at
    the fit is at/below the NLL at T=1 (no calibration signal; with
    soft-target weights summing below 1 this region also covers negative
    NLL, where the metric is meaningless).
    """
    n = len(group)
    if n < MIN_N:
        msg = f"{name}: n={n} < MIN_N={MIN_N} -> omitted from temperatures"
        print(f"WARN {msg}", file=sys.stderr)
        return None, msg
    T = fit_temperature(group)
    lo, hi = T_BOUNDS
    if T <= lo * BOUND_SLACK or T >= hi / BOUND_SLACK:
        msg = (f"{name}: fitted T={T:.4f} is at a bound ({lo}..{hi}) "
               f"-> degenerate at n={n}, refusing to ship")
        print(f"WARN {msg}", file=sys.stderr)
        return None, msg
    nll = nll_for_temperature(T, group)
    wsum = sum(r.get("label_weight", 1.0) for r in group)
    if nll <= 0.0 or nll >= nll_for_temperature(1.0, group):
        msg = (f"{name}: fitted NLL={nll:.3f} gives no improvement over T=1 "
               f"(or is negative from soft-target weighting) -> refusing to ship")
        print(f"WARN {msg}", file=sys.stderr)
        return None, msg
    if wsum > 0 and nll / wsum < COLLAPSE_NLL_PER_SAMPLE:
        msg = (f"{name}: fitted NLL={nll:.3f} over weight {wsum:.3f} collapses "
               f"to {nll / wsum:.2e} nats/sample (< {COLLAPSE_NLL_PER_SAMPLE}) — "
               f"soft-target weights near zero make the metric meaningless, "
               f"refusing to ship")
        print(f"WARN {msg}", file=sys.stderr)
        return None, msg
    return T, f"{name}: n={n} -> T={T:.6f} (NLL {nll:.3f} vs T=1 {nll_for_temperature(1.0, group):.3f})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jsonl", help="labeled judgments JSONL")
    ap.add_argument("-o", "--out", default=str(Path(__file__).with_name("calibration.json")))
    args = ap.parse_args()

    records = load_records(args.jsonl)
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        groups[str(rec.get("kind", "default"))].append(rec)

    temperatures = {}
    notes = []
    for name, group in sorted(groups.items()):
        T, note = fit_guarded(name, group)
        notes.append(note)
        if T is not None:
            temperatures[name] = round(T, 6)
            print(note)

    out = {"temperatures": temperatures,
           "note": "; ".join(notes) + f" | MIN_N={MIN_N}; omitted types fall back to T=1"}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
