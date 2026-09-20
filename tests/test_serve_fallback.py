import asyncio
import json

from quorum import serve

QS = {"q": {"type": "noul", "instructions": "x?"}}


def _resp(content):
    return {"choices": [{"message": {"content": content}, "logprobs": None}], "usage": {}}


def test_truncated_cot_degrades_to_direct_and_says_so(monkeypatch):
    calls = []

    async def fake(state, questions, cot=False):
        calls.append(cot)
        return _resp('{"q__why": "runaway rationale never clo') if cot else _resp('{"q": "yes"}')

    monkeypatch.setattr(serve, "call_upstream", fake)
    monkeypatch.setattr(serve, "log_record", lambda *a, **k: None)
    status, out = asyncio.run(serve.handle_systemone({"state": "s", "questions": QS, "reasoning": True}))
    assert status == 200 and calls == [True, False]
    assert out["mode"] == "direct-fallback" and "degraded" in out
    assert out["answers"]["q"]["answer"] == "yes"


def test_unparseable_direct_is_still_a_502(monkeypatch):
    async def fake(state, questions, cot=False):
        return _resp("not json")

    monkeypatch.setattr(serve, "call_upstream", fake)
    status, out = asyncio.run(serve.handle_systemone({"state": "s", "questions": QS}))
    assert status == 502
