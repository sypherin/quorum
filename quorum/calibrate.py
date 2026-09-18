"""quorum.calibrate — fit per-question-type temperatures from labeled judgments.

Input: a JSONL file where each line is one labeled judgment, e.g. derived
from the judgment log (serve.py) once outcomes are known:

    {"qid": "intent", "kind": "choice", "probs": {"interested": 0.7, "not_now": 0.3}, "label": "not_now"}

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


def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "probs" not in rec or "label" not in rec:
                raise ValueError(f"line {lineno}: needs 'probs' and 'label' keys")
            records.append(rec)
    if not records:
        raise ValueError("no records found")
    return records


def nll_for_temperature(T: float, group: list[dict]) -> float:
    """Negative log-likelihood of labels after temperature scaling."""
    total = 0.0
    for rec in group:
        logps = {k: math.log(max(float(p), 1e-12)) for k, p in rec["probs"].items()}
        scaled = {k: v / T for k, v in logps.items()}
        m = max(scaled.values())
        log_z = m + math.log(sum(math.exp(v - m) for v in scaled.values()))
        label = str(rec["label"])
        if label not in scaled:
            raise ValueError(f"label {label!r} not among probs keys {sorted(scaled)}")
        total += log_z - scaled[label]
    return total


def fit_temperature(group: list[dict]) -> float:
    res = minimize_scalar(
        lambda T: nll_for_temperature(T, group),
        bounds=T_BOUNDS,
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(res.x)


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
    for name, group in sorted(groups.items()):
        T = fit_temperature(group)
        temperatures[name] = round(T, 6)
        print(f"{name}: {len(group)} samples -> T = {T:.4f} "
              f"(NLL {nll_for_temperature(T, group):.3f} vs T=1 NLL {nll_for_temperature(1.0, group):.3f})")

    out = {"temperatures": temperatures}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
