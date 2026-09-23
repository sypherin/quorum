import json

import pytest

import run as R


def test_breaker_trips_on_streak_and_success_resets():
    b = R.Breaker(3)
    assert [b.record(ok) for ok in (False, False, True, False, False)] == [False] * 5
    assert b.record(False) is True and b.tripped
    assert R.Breaker(0).record(False) is False  # 0 = never trips


class _Down:
    calls = 0

    def ask(self, item):
        _Down.calls += 1
        raise RuntimeError("upstream error: All connection attempts failed")


def test_dead_upstream_stops_after_the_streak(tmp_path, monkeypatch):
    data, runs = tmp_path / "_data", tmp_path / "_runs"
    data.mkdir()
    (data / "t.jsonl").write_text("".join(
        json.dumps({"id": str(i), "state": "s", "questions": {"q": {"type": "noul"}}}) + "\n" for i in range(100)))
    monkeypatch.setattr(R, "DATA", data)
    monkeypatch.setattr(R, "RUNS", runs)
    monkeypatch.setattr(R.adapters, "make", lambda system: _Down())
    monkeypatch.setattr("sys.argv", ["run.py", "quorum-direct", "t", "--max-consecutive-errors", "7"])
    with pytest.raises(SystemExit) as e:
        R.main()
    assert e.value.code == 4
    rows = (runs / "quorum-direct" / "t.jsonl").read_text().splitlines()
    assert len(rows) == 7 and _Down.calls == 7          # not 100
    assert all(not json.loads(r)["ok"] for r in rows)
