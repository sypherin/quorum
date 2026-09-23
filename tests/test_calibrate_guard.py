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
            "raw_answers": {"q": {"type": "noul",
                                  "probabilities": {"yes": 0.7, "no": 0.3}}},
            "answers": {"q": {"type": "noul"}},
            "labels": {"q": "yes" if i % 4 else "no"},
        }))
    # one small "choice" type -> must be omitted
    lines.append(json.dumps({
        "state": "c0",
        "questions": {"q": {"type": "choice", "criteria": {"a": "a", "b": "b"}}},
        "raw_answers": {"q": {"type": "choice", "probabilities": {"a": 0.7, "b": 0.3}}},
        "answers": {"q": {"type": "choice"}},
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


# ---------- raw vs served answers ----------

from quorum.calibrate import LEGACY_RAW_BEFORE, fit_platt_guarded, load_records  # noqa: E402


def _logline(ts, served, raw=None, label="yes"):
    rec = {"ts": ts, "state": "s", "questions": {"q": {"type": "noul"}},
           "answers": {"q": {"type": "noul", "probabilities": {"yes": served, "no": 1 - served}}},
           "labels": {"q": label}}
    if raw is not None:
        rec["raw_answers"] = {"q": {"type": "noul", "probabilities": {"yes": raw, "no": 1 - raw}}}
    return json.dumps(rec)


def test_fit_reads_raw_answers_never_served(tmp_path):
    f = tmp_path / "j.jsonl"
    f.write_text(_logline("2026-09-23T00:00:00+00:00", served=0.6, raw=0.95) + "\n")
    [rec] = load_records(str(f))
    assert rec["probs"]["yes"] == 0.95


def test_legacy_records_trusted_only_before_the_cutoff(tmp_path):
    f = tmp_path / "j.jsonl"
    f.write_text("\n".join([
        _logline("2026-09-18T00:50:43.472146+00:00", served=0.9),   # before: served == raw
        _logline(LEGACY_RAW_BEFORE, served=0.8),                    # the cutoff itself: still raw
        _logline("2026-09-20T02:00:00+00:00", served=0.7),          # after: may be calibrated
        _logline("2026-09-20T02:00:00+00:00", served=0.7, raw=0.99),  # after, but carries raw
    ]) + "\n")
    stats = {}
    recs = load_records(str(f), stats)
    assert [r["probs"]["yes"] for r in recs] == [0.9, 0.8, 0.99]
    assert stats["skipped_calibrated"] == 1


def test_all_skipped_says_why(tmp_path):
    f = tmp_path / "j.jsonl"
    f.write_text(_logline("2026-09-21T00:00:00+00:00", served=0.7) + "\n")
    try:
        load_records(str(f))
    except ValueError as e:
        assert "1 labeled records skipped" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_platt_guard_refuses_small_and_no_gain():
    fit, note = fit_platt_guarded("noul", [_noul(i, 0.7, "yes") for i in range(MIN_N - 1)])
    assert fit is None and "MIN_N" in note
    # a biased judge: says 0.5 for everything, right on "yes" 90% of the time
    biased = [_noul(i, 0.5, "yes" if i % 10 else "no") for i in range(100)]
    fit, note = fit_platt_guarded("noul", biased)
    assert fit is not None and fit["b"] > 1.5  # the midpoint moves; temperature cannot do this
    # already perfectly calibrated at 0.5/0.5 labels: no gain -> refuse
    fair = [_noul(i, 0.5, "yes" if i % 2 else "no") for i in range(100)]
    fit, note = fit_platt_guarded("noul", fair)
    assert fit is None and "no gain" in note


def test_cli_platt_writes_platt_not_temperature(tmp_path):
    log = tmp_path / "j.jsonl"
    lines = [json.dumps({"ts": "2026-09-23T00:00:00+00:00", "state": f"s{i}",
                         "questions": {"q": {"type": "noul"}},
                         "raw_answers": {"q": {"type": "noul", "probabilities": {"yes": 0.5, "no": 0.5}}},
                         "answers": {}, "labels": {"q": "yes" if i % 10 else "no"}}) for i in range(60)]
    log.write_text("\n".join(lines) + "\n")
    out = tmp_path / "cal.json"
    subprocess.run([sys.executable, "-m", "quorum.calibrate", str(log), "-o", str(out), "--noul-method", "platt"],
                   cwd=REPO, capture_output=True, text=True, check=True)
    cal = json.loads(out.read_text())
    assert "noul" not in cal["temperatures"] and set(cal["platt"]["noul"]) == {"a", "b"}
