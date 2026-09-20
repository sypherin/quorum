import pytest

pytest.importorskip("playwright")
from play import JudgmentCache          # noqa: E402


class FakeClient:
    host, port, path = "127.0.0.1", 8017, "/v1/systemone"

    def __init__(self):
        self.calls = 0

    def ask(self, payload, questions):
        self.calls += 1
        return {"answers": {"jump": {"noul": 0.8}}, "model": "m", "usage": {}, "latency_ms": 1700, "request_id": "x"}


def test_cache_replays_the_models_answer_and_counts_hits_and_misses(tmp_path):
    client = FakeClient()
    c = JudgmentCache(client, tmp_path / "cache.jsonl")
    a = c.ask({"situation": "s"}, {"jump": {"type": "noul"}})
    b = c.ask({"situation": "s"}, {"jump": {"type": "noul"}})
    assert client.calls == 1 and (c.misses, c.hits) == (1, 1)
    assert b["answers"] == a["answers"] and b["latency_ms"] == 0 and b["request_id"] == "cache"
    assert c.path == "/v1/systemone"                    # the wrapper must not shadow the client's attributes


def test_cache_survives_a_restart_and_a_new_question_is_a_miss(tmp_path):
    c1 = JudgmentCache(FakeClient(), tmp_path / "cache.jsonl")
    c1.ask({"situation": "s"}, {"jump": {"type": "noul"}})
    client2 = FakeClient()
    c2 = JudgmentCache(client2, tmp_path / "cache.jsonl")
    c2.ask({"situation": "s"}, {"jump": {"type": "noul"}})
    assert client2.calls == 0
    c2.ask({"situation": "s"}, {"big": {"type": "noul"}})
    assert client2.calls == 1
