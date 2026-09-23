"""serve: fan-out, the judgment cache, raw+calibrated logging, prose state."""
import asyncio
import json
import math

import pytest

from quorum import serve

QS = {
    "a": {"type": "noul", "instructions": "a?"},
    "b": {"type": "noul", "instructions": "b?"},
    "c": {"type": "choice", "instructions": "c?", "criteria": {"x": "x", "y": "y"}},
}


def _resp(answers, p=0.8, cache_n=None):
    """A llama-server style response: constrained JSON + a logprob at each value."""
    content = json.dumps(answers, separators=(",", ":"))
    lps = []
    for k, v in answers.items():
        lps.append({"token": f'"{k}":"', "logprob": 0.0, "top_logprobs": []})
        other = {"yes": "no", "no": "yes", "x": "y", "y": "x"}[v]
        lps.append({"token": v, "logprob": math.log(p), "top_logprobs": [
            {"token": v, "logprob": math.log(p)}, {"token": other, "logprob": math.log(1 - p)}]})
        lps.append({"token": '",', "logprob": 0.0, "top_logprobs": []})
    r = {"choices": [{"message": {"content": content}, "logprobs": {"content": lps}}],
         "usage": {"prompt_tokens": 100, "completion_tokens": 5}}
    if cache_n is not None:
        r["timings"] = {"cache_n": cache_n}
    return r


ANS = {"a": "yes", "b": "no", "c": "x"}


def _fake(calls, broken_cot=()):
    async def fake(state, questions, cot=False):
        calls.append({"state": state, "qids": list(questions), "cot": cot})
        if cot and set(questions) & set(broken_cot):
            return {"choices": [{"message": {"content": '{"a__why": "never clo'}}], "usage": {}}
        return _resp({q: ANS[q] for q in questions}, cache_n=90 if len(calls) > 1 else 0)
    return fake


@pytest.fixture
def logged(monkeypatch):
    recs = []
    monkeypatch.setattr(serve, "log_record", lambda *a, **k: recs.append((a, k)))
    monkeypatch.setenv("QUORUM_CALIBRATION", "/nonexistent/calibration.json")
    return recs


def run(body):
    return asyncio.run(serve.handle_systemone(body))


def test_fanout_one_call_per_question_same_state(monkeypatch, logged):
    calls = []
    monkeypatch.setattr(serve, "call_upstream", _fake(calls))
    status, out = run({"state": "S", "questions": QS, "fanout": True})
    assert status == 200
    assert [c["qids"] for c in calls] == [["a"], ["b"], ["c"]]
    assert {c["state"] for c in calls} == {"S"}
    assert {k: v.get("answer", v.get("choice")) for k, v in out["answers"].items()} == ANS
    assert out["fanout"] is True and out["calls"] == 3 and out["mode"] == "direct"
    assert out["usage"] == {"input_tokens": 300, "output_tokens": 15, "cached_tokens": 180}
    assert out["answers"]["a"]["noul"] == pytest.approx(0.8)
    assert out["answers"]["b"]["noul"] == pytest.approx(0.2)  # P(yes) even when the answer is no


def test_default_is_one_call_for_all_questions(monkeypatch, logged):
    calls = []
    monkeypatch.setattr(serve, "call_upstream", _fake(calls))
    status, out = run({"state": "S", "questions": QS})
    assert status == 200 and len(calls) == 1 and calls[0]["qids"] == ["a", "b", "c"]
    assert out["fanout"] is False and out["calls"] == 1
    assert out["usage"]["cached_tokens"] == 0


def test_cached_tokens_unknown_when_upstream_does_not_report(monkeypatch, logged):
    async def fake(state, questions, cot=False):
        return _resp({q: ANS[q] for q in questions})  # no timings, no prompt_tokens_details

    monkeypatch.setattr(serve, "call_upstream", fake)
    _, out = run({"state": "S", "questions": QS, "fanout": True})
    assert out["usage"]["cached_tokens"] is None  # unknown, never a made-up 0
    assert out["usage"]["input_tokens"] == 300


def test_fanout_cot_falls_back_per_question_and_names_it(monkeypatch, logged):
    calls = []
    monkeypatch.setattr(serve, "call_upstream", _fake(calls, broken_cot=("b",)))
    status, out = run({"state": "S", "questions": QS, "fanout": True, "reasoning": True})
    assert status == 200
    assert [(c["qids"], c["cot"]) for c in calls] == [(["a"], True), (["b"], True), (["b"], False), (["c"], True)]
    assert out["mode"] == "direct-fallback" and "b" in out["degraded"] and "a," not in out["degraded"]
    assert out["calls"] == 4


def test_fanout_concurrency_primes_the_cache_with_the_first_question(monkeypatch, logged):
    events = []

    async def fake(state, questions, cot=False):
        q = next(iter(questions))
        events.append(("start", q))
        await asyncio.sleep(0.01)
        events.append(("end", q))
        return _resp({q: ANS[q]})

    monkeypatch.setattr(serve, "call_upstream", fake)
    monkeypatch.setattr(serve, "FANOUT_CONCURRENCY", 3)
    status, _ = run({"state": "S", "questions": QS, "fanout": True})
    assert status == 200
    assert events[:2] == [("start", "a"), ("end", "a")]
    assert set(events[2:4]) == {("start", "b"), ("start", "c")}  # the rest overlap


def test_upstream_failure_in_any_fanout_call_is_a_502(monkeypatch, logged):
    async def fake(state, questions, cot=False):
        if "c" in questions:
            raise ConnectionError("down")
        return _resp({q: ANS[q] for q in questions})

    monkeypatch.setattr(serve, "call_upstream", fake)
    status, out = run({"state": "S", "questions": QS, "fanout": True})
    assert status == 502 and "down" in out["detail"]
    assert logged == []


def test_cache_hit_skips_upstream_and_misses_on_any_option(monkeypatch, logged):
    calls = []
    monkeypatch.setattr(serve, "call_upstream", _fake(calls))
    s1, o1 = run({"state": "S", "questions": QS})
    s2, o2 = run({"state": "S", "questions": QS})
    assert s1 == s2 == 200 and len(calls) == 1
    assert o2["cached"] is True and o2["calls"] == 0 and o2["answers"] == o1["answers"]
    run({"state": "S", "questions": QS, "reasoning": True})
    run({"state": "S", "questions": QS, "fanout": True})
    run({"state": "S2", "questions": QS})
    assert len(calls) == 1 + 1 + 3 + 1


def test_cache_evicts_least_recent(monkeypatch, logged):
    calls = []
    monkeypatch.setattr(serve, "call_upstream", _fake(calls))
    monkeypatch.setattr(serve, "CACHE_SIZE", 2)
    for s in ("1", "2", "1", "3", "1", "2"):
        run({"state": s, "questions": QS})
    # 1 miss, 2 miss, 1 hit, 3 miss (evicts 2), 1 hit, 2 miss
    assert [c["state"] for c in calls] == ["1", "2", "3", "2"]


def test_cache_off(monkeypatch, logged):
    calls = []
    monkeypatch.setattr(serve, "call_upstream", _fake(calls))
    monkeypatch.setattr(serve, "CACHE_SIZE", 0)
    run({"state": "S", "questions": QS})
    run({"state": "S", "questions": QS})
    assert len(calls) == 2 and serve._cache == {}


def test_calibration_applies_on_hits_and_log_keeps_raw(monkeypatch, logged, tmp_path):
    calls = []
    monkeypatch.setattr(serve, "call_upstream", _fake(calls))
    _, o1 = run({"state": "S", "questions": QS})
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"platt": {"noul": {"a": 1.0, "b": 2.0}}}))
    monkeypatch.setenv("QUORUM_CALIBRATION", str(cal))
    _, o2 = run({"state": "S", "questions": QS})
    assert len(calls) == 1 and o2["cached"]
    want = 1 / (1 + math.exp(-(math.log(0.8 / 0.2) + 2.0)))
    assert o2["answers"]["a"]["noul"] == pytest.approx(want)
    (args, kw) = logged[-1]
    assert kw["raw_answers"]["a"]["noul"] == pytest.approx(0.8)  # the log keeps what calibrate needs
    assert args[2]["a"]["noul"] == pytest.approx(want)
    assert kw["calibration"]["platt"] == {"noul": {"a": 1.0, "b": 2.0}}
    assert kw["extra"]["cached"] is True


def test_prose_state_reaches_upstream_and_bad_format_is_400(monkeypatch, logged):
    calls = []
    monkeypatch.setattr(serve, "call_upstream", _fake(calls))
    run({"state": {"user": {"plan": "pro"}}, "questions": QS, "state_format": "prose"})
    assert calls[-1]["state"] == "user:\n  plan: pro"
    run({"state": {"user": {"plan": "pro"}}, "questions": QS})
    assert calls[-1]["state"] == '{"user": {"plan": "pro"}}'
    status, out = run({"state": {"k": 1}, "questions": QS, "state_format": "xml"})
    assert status == 400 and "state_format" in out["detail"]


def test_log_record_writes_raw_and_calibrated(tmp_path, monkeypatch):
    path = tmp_path / "j.jsonl"
    monkeypatch.setenv("QUORUM_LOG", str(path))
    serve.log_record("S", QS, {"a": {"noul": 0.6}}, raw_answers={"a": {"noul": 0.9}},
                     calibration={"temperatures": {"noul": 3.0}}, extra={"mode": "direct"})
    rec = json.loads(path.read_text())
    assert rec["answers"]["a"]["noul"] == 0.6 and rec["raw_answers"]["a"]["noul"] == 0.9
    assert rec["calibration"] == {"temperatures": {"noul": 3.0}} and rec["mode"] == "direct"
    assert rec["label"] is None
