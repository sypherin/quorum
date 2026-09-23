"""apply_calibration (post-hoc) must equal the old in-extraction temperature
path exactly, keep committed answers, and add Platt for noul."""
import copy
import json
import math

import pytest

from quorum import core


def _lp(token, logprob, tops):
    return {"token": token, "logprob": logprob,
            "top_logprobs": [{"token": t, "logprob": lp} for t, lp in tops]}


QS = {
    "urgent": {"type": "noul", "instructions": "urgent?"},
    "team": {"type": "choice", "instructions": "route",
             "criteria": {"billing": "b", "technical": "t", "sales": "s"}},
    "mood": {"type": "score", "instructions": "mood", "criteria": ["low", "mid", "high"]},
}
RAW = {"urgent": "no", "team": "billing", "mood": 1}
STREAM = [
    _lp('{"urgent":"', -0.001, []),
    _lp("no", -0.2, [("no", math.log(0.8)), ("yes", math.log(0.2))]),
    _lp('","team":"', -0.001, []),
    _lp("billing", -0.1, [("billing", math.log(0.7)), ("technical", math.log(0.25)), ("sales", math.log(0.05))]),
    _lp('","mood":', -0.001, []),
    _lp("1", -0.3, [("0", math.log(0.2)), ("1", math.log(0.5)), ("2", math.log(0.3))]),
    _lp("}", -0.001, []),
]
TEMPS = {"noul": 3.7, "choice": 1.8, "score": 0.6}


def test_post_hoc_temperature_equals_in_extraction_temperature():
    old = core.extract_answers(RAW, QS, STREAM, TEMPS)
    new = core.apply_calibration(core.extract_answers(RAW, QS, STREAM), {"temperatures": TEMPS})
    for qid in QS:
        assert new[qid].keys() == old[qid].keys(), qid
        for k, v in old[qid].items():
            if isinstance(v, dict) and k == "probabilities":
                assert new[qid][k] == pytest.approx(v, abs=1e-12)
            elif isinstance(v, float):
                assert new[qid][k] == pytest.approx(v, abs=1e-12), (qid, k)
            else:
                assert new[qid][k] == v, (qid, k)


def test_committed_answers_never_move_and_raw_is_untouched():
    raw = core.extract_answers(RAW, QS, STREAM)
    before = copy.deepcopy(raw)
    cal = core.apply_calibration(raw, {"temperatures": {"default": 50.0}})
    assert raw == before
    assert cal["urgent"]["answer"] == "no" and cal["team"]["choice"] == "billing" and cal["mood"]["level"] == 1
    # a huge T flattens toward uniform over the options that had mass
    assert cal["team"]["probabilities"]["billing"] < 0.4


def test_platt_supersedes_temperature_for_noul_only():
    raw = core.extract_answers(RAW, QS, STREAM)
    cal = core.apply_calibration(raw, {"temperatures": {"noul": 9.0, "choice": 2.0},
                                       "platt": {"noul": {"a": 0.5, "b": 1.0}}})
    want = core.platt_apply(0.2, 0.5, 1.0)
    assert cal["urgent"]["noul"] == pytest.approx(want)
    assert cal["urgent"]["probabilities"] == pytest.approx({"yes": want, "no": 1 - want})
    assert cal["urgent"]["confidence"] == pytest.approx(max(want, 1 - want))
    # choice still gets its temperature
    assert cal["team"]["probabilities"] == pytest.approx(
        core.apply_calibration(raw, {"temperatures": {"choice": 2.0}})["team"]["probabilities"])


def test_no_calibration_is_identity_and_degraded_answers_pass_through():
    raw = core.extract_answers(RAW, QS, STREAM)
    assert core.apply_calibration(raw, None) == raw
    blind = core.extract_answers({"urgent": "yes"}, {"urgent": QS["urgent"]}, None)
    assert core.apply_calibration(blind, {"temperatures": TEMPS}) == blind
    missing = core.extract_answers({}, {"urgent": QS["urgent"]}, STREAM)
    assert core.apply_calibration(missing, {"temperatures": TEMPS}) == missing


def test_score_keeps_committed_level_and_expectation():
    raw = core.extract_answers(RAW, QS, STREAM)["mood"]
    assert raw["level"] == 1 and raw["score"] == pytest.approx(0 * 0.2 + 1 * 0.5 + 2 * 0.3)


def test_load_calibration_reads_platt_and_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("QUORUM_CALIBRATION", str(tmp_path / "absent.json"))
    assert core.load_calibration() == {"temperatures": {}, "platt": {}}
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"temperatures": {"choice": 2}, "platt": {"noul": {"a": "0.5", "b": -1}}}))
    monkeypatch.setenv("QUORUM_CALIBRATION", str(f))
    assert core.load_calibration() == {"temperatures": {"choice": 2.0}, "platt": {"noul": {"a": 0.5, "b": -1.0}}}
    assert core.load_temperatures() == {"choice": 2.0}


def test_platt_fit_recovers_bias_and_overconfidence():
    p = [0.5] * 100
    y = [True] * 90 + [False] * 10
    a, b = core.platt_fit(p, y)
    assert core.platt_apply(0.5, a, b) == pytest.approx(0.9, abs=0.01)
    p2 = [0.99] * 50 + [0.01] * 50
    y2 = [True] * 35 + [False] * 15 + [False] * 35 + [True] * 15
    a2, b2 = core.platt_fit(p2, y2)
    assert 0 < a2 < 1 and core.platt_apply(0.99, a2, b2) == pytest.approx(0.7, abs=0.02)
    # saturated inputs must not blow up the Newton step
    a3, b3 = core.platt_fit([1.0] * 20 + [0.0] * 20, [True] * 15 + [False] * 5 + [False] * 15 + [True] * 5)
    assert math.isfinite(a3) and math.isfinite(b3)


# ---------- prose state ----------

def test_render_state_keeps_every_key_and_value():
    state = {"ticket": {"id": 42, "vip": True, "note": None, "tags": ["billing", "late"],
                        "body": "line one\nline two"},
             "history": [{"from": "agent", "text": "hi"}, []], "empty": {}}
    out = core.render_state(state)
    assert out.splitlines() == [
        "ticket:",
        "  id: 42",
        "  vip: true",
        "  note: none",
        "  tags:",
        "    - billing",
        "    - late",
        "  body: line one",
        "    line two",
        "history:",
        "  -",
        "    from: agent",
        "    text: hi",
        "  - (empty)",
        "empty: (empty)",
    ]


def test_validate_state_format():
    q = {"a": {"type": "noul", "instructions": "x"}}
    s_json, _ = core.validate_request({"state": {"k": "v"}, "questions": q})
    s_prose, _ = core.validate_request({"state": {"k": "v"}, "questions": q}, state_format="prose")
    assert s_json == '{"k": "v"}' and s_prose == "k: v"
    s_str, _ = core.validate_request({"state": '{"k": "v"}', "questions": q}, state_format="prose")
    assert s_str == '{"k": "v"}'  # a string state is never re-rendered
    with pytest.raises(core.BadRequest):
        core.validate_request({"state": "s", "questions": q}, state_format="yaml")


def test_temper_survives_extreme_temperatures():
    p = {"a": 0.6, "b": 0.3, "c": 0.1, "d": 0.0}
    sharp = core.temper(p, 0.05)
    assert sharp["a"] == pytest.approx(1.0) and sharp["d"] == 0.0
    tiny = core.temper({"a": 1e-300, "b": 1e-310}, 0.05)
    assert tiny["a"] == pytest.approx(1.0)  # direct powers give 0/0 here
    flat = core.temper(p, 1e6)
    assert flat["a"] == pytest.approx(1 / 3, abs=1e-5) and flat["d"] == 0.0
