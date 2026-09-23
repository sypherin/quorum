import json
import math

import pytest

import report as R


def _item(iid, gold, qtype="choice", options=("a", "b", "c"), private=False, qid="q", probs=None):
    q = {"type": qtype, "instructions": "x"}
    if qtype in ("choice", "score"):
        q["criteria"] = {o: f"means {o}" for o in options}
    return {"id": iid, "state": "s", "questions": {qid: q},
            "gold": {qid: {"label": gold, "probs": probs, "score": None}}, "private": private}


def _row(iid, label, probs, qid="q", ms=10, meta=None):
    return {"id": iid, "ok": True, "answers": {qid: {"label": label, "probs": probs, "score": None}},
            "ms": ms, "meta": meta or {}}


# ---------- units and attachment ----------

def test_score_gold_falls_back_to_label_level():
    it = _item("1", "2", qtype="score", options=("0", "1", "2", "3"))
    (u,) = R.build_units([it])
    assert u["options"] == ["0", "1", "2", "3"]
    assert u["gold_score"] == 2.0


def test_unanswered_gets_uniform_and_counts_wrong():
    units = R.build_units([_item("1", "a"), _item("2", "b")])
    preds = R.attach(units, {"1": _row("1", "a", {"a": 0.9, "b": 0.05, "c": 0.05})})
    assert preds[1] == {"label": None, "probs": {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3},
                        "answered": False, "has_dist": False, "score": None}
    cell = R.score_cell(units, preds)
    assert cell["acc"] == 0.5
    assert cell["coverage"] == 0.5
    # the unanswered unit pays log(3), never zero
    assert cell["nll"] == pytest.approx((-math.log(0.9) + math.log(3)) / 2)


def test_label_without_distribution_is_counted():
    units = R.build_units([_item("1", "a")])
    preds = R.attach(units, {"1": _row("1", "a", None)})
    assert preds[0]["answered"] and not preds[0]["has_dist"]
    cell = R.score_cell(units, preds)
    assert cell["acc"] == 1.0 and cell["no_dist"] == 1


def test_read_run_keeps_last_ok_and_ignores_errors(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join([
        json.dumps({"id": "1", "ok": False, "error": "boom"}),
        "{not json",
        json.dumps(_row("1", "b", {"b": 1.0})),
        json.dumps(_row("1", "a", {"a": 1.0})),
        json.dumps({"id": "2", "ok": False, "error": "boom"}),
    ]) + "\n")
    rows = R.read_run(p)
    assert set(rows) == {"1"} and rows["1"]["answers"]["q"]["label"] == "a"
    assert R.read_run(tmp_path / "missing.jsonl") == {}


def test_soft_gold_nll_is_cross_entropy():
    it = _item("1", "a", probs={"a": 0.75, "b": 0.25, "c": 0.0})
    (u,) = R.build_units([it])
    p = {"a": 0.5, "b": 0.5, "c": 0.0}
    assert R.unit_nll(u, p) == pytest.approx(-(0.75 * math.log(0.5) + 0.25 * math.log(0.5)))


# ---------- cross-validated calibration ----------

def test_cv_folds_never_split_an_item(monkeypatch):
    items = []
    for i in range(20):
        it = _item(str(i), "a", qid="q1")
        it["questions"]["q2"] = dict(it["questions"]["q1"])
        it["gold"]["q2"] = {"label": "b", "probs": None, "score": None}
        items.append(it)
    units = R.build_units(items)
    rows = {str(i): {"id": str(i), "ok": True, "ms": 1, "meta": {},
                     "answers": {q: {"label": "a", "probs": {"a": 0.6, "b": 0.3, "c": 0.1}} for q in ("q1", "q2")}}
            for i in range(20)}
    preds = R.attach(units, rows)
    seen = []
    real_fit = R.fit_maps

    def spy(units_, preds_, idx, platt):
        seen.append({units_[i]["item"] for i in idx})
        return real_fit(units_, preds_, idx, platt)

    monkeypatch.setattr(R, "fit_maps", spy)
    out = R.cv_calibrated(units, preds, platt=False, k=5)
    assert len(seen) == 5
    all_items = {u["item"] for u in units}
    held = [all_items - s for s in seen]
    assert all(len(h) == 4 for h in held)
    assert set().union(*held) == all_items          # every item held out exactly once
    assert all(o is not None for o in out)


def test_cv_leaves_unanswered_units_uniform():
    items = [_item(str(i), "a") for i in range(10)]
    units = R.build_units(items)
    rows = {str(i): _row(str(i), "a", {"a": 0.99, "b": 0.005, "c": 0.005}) for i in range(9)}
    preds = R.attach(units, rows)
    out = R.cv_calibrated(units, preds, platt=False, k=5)
    assert out[9] == {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}


def test_cv_platt_repairs_an_overconfident_noul_system():
    # says 0.95 yes on everything, but only 60% are yes: calibration must pull toward 0.6
    items = [_item(str(i), "yes" if i % 5 < 3 else "no", qtype="noul") for i in range(100)]
    units = R.build_units(items)
    rows = {str(i): _row(str(i), "yes", {"yes": 0.95, "no": 0.05}) for i in range(100)}
    preds = R.attach(units, rows)
    raw = R.score_probs(units, preds, [p["probs"] for p in preds])
    cols = R.calibration_columns(units, preds)
    assert cols["cv_platt"]["nll"] < raw["nll"] - 0.3
    assert cols["cv_best"]["nll"] == min(cols["cv_temp"]["nll"], cols["cv_platt"]["nll"])
    # labels never move: accuracy is identical before and after
    assert [p["label"] for p in preds] == ["yes"] * 100


def test_cv_platt_redecides_a_biased_noul_system_and_keeps_raw_accuracy():
    # ranks perfectly but says "no" to everything: P(yes) 0.3 on yes items, 0.1 on no items
    items = [_item(str(i), "yes" if i % 2 else "no", qtype="noul") for i in range(100)]
    units = R.build_units(items)
    rows = {str(i): _row(str(i), "no", {"yes": 0.3 if i % 2 else 0.1, "no": 0.7 if i % 2 else 0.9})
            for i in range(100)}
    preds = R.attach(units, rows)
    cell = R.score_cell(units, preds)
    cols = R.calibration_columns(units, preds)
    assert cell["acc"] == 0.5                       # raw column never moves
    assert cols["cv_platt"]["acc"] == 1.0           # the fitted threshold separates them
    assert "acc" not in cols["cv_temp"]


def test_decide_moves_only_answered_noul_labels():
    items = [_item("c", "a"), _item("n", "yes", qtype="noul"), _item("u", "yes", qtype="noul")]
    units = R.build_units(items)
    rows = {"c": _row("c", "b", {"a": 0.4, "b": 0.5, "c": 0.1}), "n": _row("n", "no", {"yes": 0.2, "no": 0.8})}
    preds = R.attach(units, rows)
    mapped = [{"a": 0.9, "b": 0.05, "c": 0.05}, {"yes": 0.7, "no": 0.3}, {"yes": 0.9, "no": 0.1}]
    assert R.decide(units, preds, mapped) == ["b", "yes", None]


def test_cv_platt_labels_none_without_noul_distributions():
    units = R.build_units([_item(str(i), "a") for i in range(5)])
    preds = R.attach(units, {str(i): _row(str(i), "a", {"a": 0.8, "b": 0.1, "c": 0.1}) for i in range(5)})
    assert R.cv_platt_labels(units, preds) is None


def test_no_distributions_means_no_calibration_columns():
    units = R.build_units([_item("1", "a")])
    preds = R.attach(units, {"1": _row("1", "a", None)})
    assert R.calibration_columns(units, preds) == {}


# ---------- assembly ----------

def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_evaluate_private_scope_missing_runs_and_pairs(tmp_path, monkeypatch):
    data, runs = tmp_path / "_data", tmp_path / "_runs"
    monkeypatch.setattr(R, "DATA", data)
    monkeypatch.setattr(R, "RUNS", runs)
    _write(data / "pub.jsonl", [_item(str(i), "a") for i in range(6)])
    _write(data / "priv.jsonl", [_item(str(i), "a", private=True) for i in range(4)])
    _write(data / "mixed.jsonl", [_item("p", "a", private=True), _item("q", "a")])
    good = {"a": 0.8, "b": 0.1, "c": 0.1}
    _write(runs / "quorum-direct" / "pub.jsonl", [_row(str(i), "a", good) for i in range(6)])
    _write(runs / "quorum-direct" / "priv.jsonl", [_row(str(i), "a", good) for i in range(4)])
    _write(runs / "jev" / "pub.jsonl", [_row(str(i), "a" if i < 3 else "b", good) for i in range(6)])
    _write(runs / "jev" / "priv.jsonl", [])
    _write(runs / "jev" / "mixed.jsonl", [_row("q", "a", good)])

    rep = R.evaluate(["jev", "quorum-direct"], ["pub", "priv", "mixed"])
    assert rep["cells"]["jev|priv"] == {"na": "private items are never sent to the cloud"}
    assert rep["cells"]["jev|mixed"]["n"] == 1 and rep["cells"]["jev|mixed"]["acc"] == 1.0
    assert "quorum-direct|mixed" not in rep["cells"]            # no run file: absent, not zero
    pairs = {(p["a"], p["b"], p["task"]): p for p in rep["pairs"]}
    assert set(pairs) == {("quorum-direct", "jev", "pub"), ("quorum-direct", "jev", "ALL (pooled units)")}
    assert pairs[("quorum-direct", "jev", "pub")]["diff"] == pytest.approx(0.5)
    assert "| priv | " in R.to_markdown(rep) and "n/a" in R.to_markdown(rep)


def test_evaluate_pairs_cv_only_on_tasks_with_noul(tmp_path, monkeypatch):
    data, runs = tmp_path / "_data", tmp_path / "_runs"
    monkeypatch.setattr(R, "DATA", data)
    monkeypatch.setattr(R, "RUNS", runs)
    _write(data / "yn.jsonl", [_item(str(i), "yes" if i % 2 else "no", qtype="noul") for i in range(20)])
    _write(data / "ch.jsonl", [_item(str(i), "a") for i in range(6)])
    biased = lambda i: _row(str(i), "no", {"yes": 0.3 if i % 2 else 0.1, "no": 0.7 if i % 2 else 0.9})
    for s in ("quorum-direct", "laya"):
        _write(runs / s / "yn.jsonl", [biased(i) for i in range(20)])
        _write(runs / s / "ch.jsonl", [_row(str(i), "a", {"a": 0.8, "b": 0.1, "c": 0.1}) for i in range(6)])
    rep = R.evaluate(["laya", "quorum-direct"], ["yn", "ch"])
    assert {p["task"] for p in rep["pairs_cv"]} == {"yn", "ALL (pooled units)"}
    assert {p["task"] for p in rep["pairs"]} == {"yn", "ch", "ALL (pooled units)"}
    assert rep["cells"]["quorum-direct|yn"]["acc"] == 0.5
    assert rep["cells"]["quorum-direct|yn"]["cv_platt"]["acc"] == 1.0
    md = R.to_markdown(rep)
    assert "| yn | 0.500 -> 1.000 | 0.500 -> 1.000 |" in md and "after CV Platt" in md


def test_median_ms_counts_only_stock_laya_rows():
    rows = {"1": _row("1", "a", None, ms=100, meta={"load": "stock"}),
            "2": _row("2", "a", None, ms=900, meta={"load": "mmap"}),
            "3": _row("3", "a", None, ms=300, meta={"load": "stock"}),
            "4": _row("4", "a", None, ms=700, meta={})}
    assert R.median_ms(rows, "laya") == 200
    assert R.median_ms(rows, "quorum-direct") == 500
    assert R.median_ms({"4": rows["4"]}, "laya-td") is None


def test_limit_scores_the_run_py_slice_only(tmp_path, monkeypatch):
    data, runs = tmp_path / "_data", tmp_path / "_runs"
    monkeypatch.setattr(R, "DATA", data)
    monkeypatch.setattr(R, "RUNS", runs)
    _write(data / "t.jsonl", [_item(str(i), "a") for i in range(10)])
    good = {"a": 0.8, "b": 0.1, "c": 0.1}
    # a --limit 4 run: first 4 items answered (slow ones), nothing after
    _write(runs / "quorum-cot" / "t.jsonl", [_row(str(i), "a", good, ms=900) for i in range(4)])
    # a full run whose items past the slice are fast and wrong
    _write(runs / "quorum-direct" / "t.jsonl",
           [_row(str(i), "a" if i < 4 else "b", good, ms=100 if i < 4 else 5) for i in range(10)])

    full = R.evaluate(["quorum-direct", "quorum-cot"], ["t"])
    assert full["cells"]["quorum-cot|t"]["coverage"] == 0.4          # unsliced: charged for 6 misses

    sl = R.evaluate(["quorum-direct", "quorum-cot"], ["t"], limit=4)
    assert sl["cells"]["quorum-cot|t"]["coverage"] == 1.0
    assert sl["cells"]["quorum-cot|t"]["acc"] == 1.0
    assert sl["cells"]["quorum-direct|t"]["acc"] == 1.0              # its wrong tail is out of scope
    assert sl["cells"]["quorum-direct|t"]["ms"] == 100               # latency from the slice's rows only
    (pair,) = [p for p in sl["pairs"] if p["task"] == "t"]
    assert pair["n"] == 4
