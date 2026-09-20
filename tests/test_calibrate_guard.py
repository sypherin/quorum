"""Tests for the calibrate.py shipping guard (MIN_N, bound-hits, negative NLL)."""
import json
import math
import subprocess
import sys
from pathlib import Path

from quorum.calibrate import MIN_N, fit_guarded

REPO = Path(__file__).resolve().parents[1]


def _noul(i, p_yes, label):
    return {"qid": f"q{i}", "kind": "noul",
            "probs": {"yes": p_yes, "no": 1.0 - p_yes}, "label": label}


def _group(n, maker):
    return [maker(i) for i in range(n)]


def _well_behaved(i):
    # label agrees with the majority prob, mild confidence -> interior T
    return _noul(i, 0.7, "yes" if i % 4 else "no")


def test_below_min_n_is_omitted():
    group = _group(MIN_N - 1, _well_behaved)
    T, note = fit_guarded("tiny", group)
    assert T is None
    assert f"n={MIN_N - 1}" in note and "MIN_N" in note


def test_min_n_and_above_ships():
    T, note = fit_guarded("ok", _group(MIN_N, _well_behaved))
    assert T is not None and 0.05 < T < 20.0


def test_bound_hit_is_refused():
    # probs the label never supports: NLL keeps improving as T -> the upper
    # bound -> the fit runs away and must not ship
    group = _group(MIN_N, lambda i: _noul(i, 1e-9, "yes"))
    T, note = fit_guarded("runaway", group)
    assert T is None
    assert "bound" in note


def test_collapsed_weight_nll_is_refused():
    # the score-fit artifact at low n: soft-target weights (label_weight)
    # summing well below 1 collapse the weighted NLL to ~0, so the fit looks
    # perfect (T=0.06, NLL=6e-18) while the metric is meaningless. The guard
    # must refuse the type instead of shipping a fake T.
    group = [_noul(i, 0.9, "yes") for i in range(MIN_N)]
    for r in group:
        r["label_weight"] = 0.001
    T, note = fit_guarded("soft", group)
    assert T is None
    assert "NLL" in note


def test_cli_omits_small_types_and_keeps_note(tmp_path):
    log = tmp_path / "judgments.jsonl"
    lines = []
    for i in range(MIN_N):
        lines.append(json.dumps({
            "state": f"s{i}",
            "questions": {"q": {"type": "noul", "criteria": {"yes": "y", "no": "n"}}},
            "answers": {"q": {"type": "noul",
                              "probabilities": {"yes": 0.7, "no": 0.3}}},
            "labels": {"q": "yes" if i % 4 else "no"},
        }))
    # one small "choice" type -> must be omitted
    lines.append(json.dumps({
        "state": "c0",
        "questions": {"q": {"type": "choice", "criteria": {"a": "a", "b": "b"}}},
        "answers": {"q": {"type": "choice", "probabilities": {"a": 0.7, "b": 0.3}}},
        "labels": {"q": "a"},
    }))
    log.write_text("\n".join(lines) + "\n")
    out = tmp_path / "cal.json"
    res = subprocess.run(
        [sys.executable, "-m", "quorum.calibrate", str(log), "-o", str(out)],
        cwd=REPO, capture_output=True, text=True, check=True)
    cal = json.loads(out.read_text())
    assert "noul" in cal["temperatures"]
    assert "choice" not in cal["temperatures"]
    assert f"n=1 < MIN_N={MIN_N}" in cal["note"]
    assert "choice" in cal["note"]
    assert "WARN" in res.stderr and "choice" in res.stderr
