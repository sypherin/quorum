import json
import math

import pytest

from quorum import core


# ---------- validate_request ----------

def test_validate_rejects_missing_state():
    with pytest.raises(core.BadRequest):
        core.validate_request({"questions": {"a": {"type": "noul", "instructions": "x"}}})


def test_validate_rejects_bad_type():
    with pytest.raises(core.BadRequest):
        core.validate_request({"state": "s", "questions": {"a": {"type": "poem", "instructions": "x"}}})


def test_validate_rejects_object_score_criteria():
    # typesafe itself rejects object criteria for score — we mirror that
    with pytest.raises(core.BadRequest):
        core.validate_request({
            "state": "s",
            "questions": {"a": {"type": "score", "instructions": "x", "criteria": {"lo": "a", "hi": "b"}}},
        })


def test_validate_normalizes_dict_state():
    state, qs = core.validate_request({
        "state": {"msg": "help"},
        "questions": {"a": {"type": "noul", "instructions": "urgent?"}},
    })
    assert json.loads(state) == {"msg": "help"}
    assert qs["a"]["type"] == "noul"


# ---------- build_schema ----------

def test_schema_enums_in_order():
    qs = {
        "urgency": {"type": "noul", "instructions": "urgent?"},
        "team": {"type": "choice", "instructions": "route", "criteria": {"billing": "b", "tech": "t"}},
        "mood": {"type": "score", "instructions": "mood", "criteria": ["low", "mid", "high"]},
    }
    schema = core.build_schema(qs)
    props = schema["schema"]["properties"]
    assert props["urgency"]["enum"] == ["yes", "no"]
    assert props["team"]["enum"] == ["billing", "tech"]
    assert props["mood"]["maximum"] == 2
    assert set(schema["schema"]["required"]) == set(qs)


# ---------- chain-of-thought mode (opt-in) ----------

def test_cot_schema_interleaves_bounded_reasoning_before_answers():
    qs = {
        "urgency": {"type": "noul", "instructions": "urgent?"},
        "team": {"type": "choice", "instructions": "route", "criteria": {"billing": "b", "tech": "t"}},
    }
    schema = core.build_schema(qs, cot=True)
    props = schema["schema"]["properties"]
    # a bounded reasoning field exists per question
    assert props["urgency__why"]["type"] == "string"
    assert props["urgency__why"]["maxLength"] == core.REASON_MAXLEN
    assert props["team__why"]["maxLength"] == core.REASON_MAXLEN
    # answer fields are unchanged (extraction still finds them)
    assert props["urgency"]["enum"] == ["yes", "no"]
    assert props["team"]["enum"] == ["billing", "tech"]
    # order: each '__why' comes immediately BEFORE its answer (model reasons then commits)
    req = schema["schema"]["required"]
    assert req.index("urgency__why") < req.index("urgency")
    assert req.index("team__why") < req.index("team")


def test_cot_off_is_byte_identical_to_default():
    qs = {"a": {"type": "noul", "instructions": "x"}}
    assert core.build_schema(qs, cot=False) == core.build_schema(qs)
    assert "__why" not in json.dumps(core.build_schema(qs))
    assert core.build_messages("s", qs, cot=False) == core.build_messages("s", qs)


def test_cot_system_prompt_permits_reasoning_only_in_why():
    qs = {"a": {"type": "noul", "instructions": "x"}}
    sysmsg = core.build_messages("s", qs, cot=True)[0]["content"]
    assert "__why" in sysmsg
    # default prompt forbids reasoning; CoT prompt must not
    assert "do not reason" not in sysmsg.lower()


def test_cot_answer_key_not_shadowed_by_why_key_on_rfind():
    # '"a__why":' must NOT satisfy a search for '"a":' — the value token walk
    # relies on the answer key being findable and last
    stream = [
        _lp('{"a__why":"', -0.001, []),
        _lp("brief reason", -0.01, []),
        _lp('","a":"', -0.001, []),
        _lp("yes", -0.02, [("yes", math.log(0.9)), ("no", math.log(0.1))]),
    ]
    idx = core.find_value_token(stream, "a")
    p = core.probs_at_position(stream, idx, ["yes", "no"])
    assert p == {"yes": pytest.approx(0.9), "no": pytest.approx(0.1)}


# ---------- logprob extraction ----------

def _lp(token, logprob, tops):
    return {"token": token, "logprob": logprob,
            "top_logprobs": [{"token": t, "logprob": lp} for t, lp in tops]}


def test_noul_probs_from_logprobs():
    # stream: think tokens then {"urgency":"yes"} — value token carries the signal
    stream = [
        _lp("<th", -0.01, []),
        _lp('{"urgency":"', -0.001, []),
        _lp('yes', -0.05, [("yes", math.log(0.95)), ('no', math.log(0.05))]),
        _lp('"}', -0.001, []),
    ]
    idx = core.find_value_token(stream, "urgency")
    assert idx == 2
    p = core.probs_at_position(stream, idx, ["yes", "no"])
    assert p == {"yes": pytest.approx(0.95), "no": pytest.approx(0.05)}


def test_choice_probs_and_confidence():
    stream = [
        _lp('{"team":"', -0.001, []),
        _lp('billing', -0.02, [("billing", math.log(0.9)), ("technical", math.log(0.1))]),
    ]
    raw = {"team": "billing"}
    qs = {"team": {"type": "choice", "instructions": "route",
                   "criteria": {"billing": "b", "technical": "t", "sales": "s"}}}
    out = core.extract_answers(raw, qs, stream)["team"]
    assert out["choice"] == "billing"
    assert out["confidence"] == pytest.approx(0.9)
    assert out["probabilities"]["sales"] == 0.0


def test_score_weighted_mean():
    stream = [
        _lp('{"mood":', -0.001, []),
        _lp("1", -0.05, [("0", math.log(0.1)), ("1", math.log(0.6)), ("2", math.log(0.3))]),
    ]
    qs = {"mood": {"type": "score", "instructions": "mood", "criteria": ["low", "mid", "high"]}}
    out = core.extract_answers({"mood": 1}, qs, stream)["mood"]
    assert out["score"] == pytest.approx(0 * 0.1 + 1 * 0.6 + 2 * 0.3)
    assert out["probabilities"]["1"] == pytest.approx(0.6)


def test_missing_logprobs_degrades_explicitly():
    qs = {"a": {"type": "noul", "instructions": "x"}}
    out = core.extract_answers({"a": "yes"}, qs, None)["a"]
    assert out["noul"] is None and out["probabilities"] is None


def test_rfind_skips_think_block_mentions():
    # the key name appears inside a think block BEFORE the real JSON — rfind must win
    stream = [
        _lp("<think> maybe \"urgency\":\"no\"? </think>", -0.01, []),
        _lp('{"urgency":"', -0.001, []),
        _lp("yes", -0.001, [("yes", math.log(0.8)), ("no", math.log(0.2))]),
    ]
    assert core.find_value_token(stream, "urgency") == 2


def test_pretty_printed_json_spacer_tokens():
    # real llama-server output shape: {"team": "billing"} — spacer ' "' token
    # between colon and value; probs_for_key must walk forward
    stream = [
        _lp('{"team":', -0.001, []),
        _lp(' "', -0.001, [('"', -9.0), (' x', -9.5)]),
        _lp('billing', -0.02, [("billing", math.log(0.9)), ("technical", math.log(0.1))]),
    ]
    qs = {"team": {"type": "choice", "instructions": "route",
                   "criteria": {"billing": "b", "technical": "t"}}}
    out = core.extract_answers({"team": "billing"}, qs, stream)["team"]
    assert out["probabilities"] == {"billing": pytest.approx(0.9), "technical": pytest.approx(0.1)}
    assert out["confidence"] == pytest.approx(0.9)


# ---------- cloud-Jev contract regressions (2026-09-17 review) ----------

def test_noul_field_is_always_p_yes():
    # model clearly answers "no" — cloud Jev's noul field is P(yes), i.e. LOW
    stream = [
        _lp('{"wants_meeting":"', -0.001, []),
        _lp('no', -0.05, [("no", math.log(0.95)), ("yes", math.log(0.05))]),
    ]
    qs = {"wants_meeting": {"type": "noul", "instructions": "wants a meeting?"}}
    out = core.extract_answers({"wants_meeting": "no"}, qs, stream)["wants_meeting"]
    assert out["noul"] == pytest.approx(0.05)
    assert out["probabilities"]["no"] == pytest.approx(0.95)


def test_noul_criteria_reach_the_prompt():
    qs = {"safety": {"type": "noul", "instructions": "risky?",
                     "criteria": {"yes": "creates risk of data loss",
                                  "no": "no such risk"}}}
    msgs = core.build_messages("state text", qs)
    user = msgs[1]["content"]
    assert "creates risk of data loss" in user
    assert "no such risk" in user


def test_ambiguous_prefix_token_not_misattributed():
    # token 'not' is the start of BOTH options — its logprob must not be
    # assigned to either; with no other signal the position yields nothing
    stream = [
        _lp('{"intent":"', -0.001, []),
        _lp('not', -0.1, [("not", math.log(0.9)), ("interested", math.log(0.1))]),
        _lp('_now', -0.01, [("_now", -0.01)]),
    ]
    qs = {"intent": {"type": "choice", "instructions": "intent?",
                     "criteria": {"interested": "i", "not_now": "n", "not_a_fit": "f"}}}
    out = core.extract_answers({"intent": "not_now"}, qs, stream)["intent"]
    # no unambiguous signal anywhere near the value slot => no distribution,
    # rather than a wrongly-attributed one
    assert out["probabilities"] is None
    assert out["choice"] == "not_now"


def test_unambiguous_prefix_token_still_matches():
    # ' bill' is a prefix of only one option — keep the partial-token tolerance
    stream = [
        _lp('{"team":"', -0.001, []),
        _lp(' bill', -0.02, [(" bill", math.log(0.8)), ("technical", math.log(0.2))]),
    ]
    qs = {"team": {"type": "choice", "instructions": "route",
                   "criteria": {"billing": "b", "technical": "t"}}}
    out = core.extract_answers({"team": "billing"}, qs, stream)["team"]
    assert out["probabilities"]["billing"] == pytest.approx(0.8)
