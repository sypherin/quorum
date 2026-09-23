import json

import pytest
import torch
from safetensors.torch import load_file, save_file

import adapters as A


def test_mmap_safetensors_matches_load_file(tmp_path):
    p = str(tmp_path / "w.safetensors")
    save_file({"a": torch.randn(3, 5).to(torch.bfloat16), "b": torch.arange(7, dtype=torch.int64),
               "c": torch.randn(2, 2, 2), "e": torch.zeros(0)}, p)
    ref, got = load_file(p), A.mmap_safetensors(p)
    assert set(ref) == set(got)
    for k in ref:
        assert got[k].dtype == ref[k].dtype and got[k].shape == ref[k].shape
        assert torch.equal(got[k], ref[k])


def test_normalize_answer_noul_uses_p_yes():
    q = {"type": "noul", "instructions": "x"}
    r = A.normalize_answer({"noul": 0.2}, q)
    assert r["label"] == "no" and abs(r["probs"]["yes"] - 0.2) < 1e-9 and abs(r["conf"] - 0.8) < 1e-9


def test_normalize_answer_score_expected_and_argmax():
    q = {"type": "score", "instructions": "x", "criteria": ["a", "b", "c"]}
    r = A.normalize_answer({"probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}, q)
    assert r["label"] == "2" and abs(r["score"] - 1.6) < 1e-9


def test_normalize_answer_error_is_explicit():
    r = A.normalize_answer({"error": "boom"}, {"type": "noul", "instructions": "x"})
    assert r["label"] is None and r["error"] == "boom"


class _Rot(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        self.ln = torch.nn.LayerNorm(4)
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, 4, 2).float() / 4)), persistent=False)

    def forward(self, x):
        return self.ln(self.lin(x)) * self.inv_freq.repeat(2)


def test_params_on_meta_then_assign_reproduces_model(tmp_path):
    torch.manual_seed(0)
    ref = _Rot()
    p = str(tmp_path / "m.safetensors")
    save_file(ref.state_dict(), p)
    with A.params_on_meta():
        m = _Rot()
    assert all(t.is_meta for t in m.parameters())
    assert not m.inv_freq.is_meta  # computed buffer survives
    m.load_state_dict(A.mmap_safetensors(p), strict=True, assign=True)
    assert not any(t.is_meta for t in m.parameters())
    x = torch.randn(3, 4)
    assert torch.equal(m(x), ref(x))
    # the patch is undone on exit
    assert not any(t.is_meta for t in torch.nn.Linear(2, 2).parameters())


class _Enc(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(10, 8)
        self.lin = torch.nn.Linear(8, 8)
        self.ln = torch.nn.LayerNorm(8, bias=False)

    def forward(self, ids):
        return self.ln(self.lin(self.emb(ids)))


def test_upcast_on_use_equals_fp32_model_exactly():
    torch.manual_seed(1)
    stored = _Enc().to(torch.bfloat16)
    ref = _Enc()
    ref.load_state_dict({k: v.float() for k, v in stored.state_dict().items()})
    assert A.upcast_on_use(stored) == 3
    ids = torch.tensor([[1, 2, 3], [4, 5, 9]])
    out = stored(ids)
    assert out.dtype == torch.float32
    assert torch.equal(out, ref(ids))
    assert all(p.dtype == torch.bfloat16 for p in stored.parameters())  # storage untouched


def test_upcast_on_use_refuses_unknown_parameter_owner():
    m = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Conv1d(1, 1, 1))
    import pytest
    with pytest.raises(TypeError):
        A.upcast_on_use(m)


# ---------- shortlist wrapper ----------

def test_reduce_questions_keeps_original_order_and_passes_others():
    qs = {"intent": {"type": "choice", "criteria": {"a": "A", "b": "B", "c": "C", "d": "D"}},
          "ok": {"type": "noul", "instructions": "x"}}
    out = A.reduce_questions(qs, {"intent": ["d", "b"]})
    assert list(out["intent"]["criteria"]) == ["b", "d"]          # question order, not rank order
    assert out["ok"] is qs["ok"]
    assert list(qs["intent"]["criteria"]) == ["a", "b", "c", "d"]  # caller's dict untouched
    assert A.reduce_questions(qs, {}) == qs


def test_reduce_questions_refuses_unknown_labels():
    qs = {"intent": {"type": "choice", "criteria": {"a": "A"}}}
    with pytest.raises(ValueError):
        A.reduce_questions(qs, {"intent": ["zz"]})


def test_shortlisted_charges_dropped_gold_as_a_miss(tmp_path, monkeypatch):
    import shortlist as SL
    monkeypatch.setattr(SL, "DATA", tmp_path)
    (tmp_path / "t.sl.json").write_text(json.dumps({"k": 2, "model": "m", "lists": {"1": {"intent": ["b", "c"]}}}))
    seen = {}

    class Inner:
        def ask(self, item):
            seen["criteria"] = list(item["questions"]["intent"]["criteria"])
            return {"intent": {"choice": "b", "probabilities": {"b": 0.7, "c": 0.3}}}, {"load": "stock"}

    item = {"id": "1", "task": "t", "state": "s",
            "questions": {"intent": {"type": "choice", "criteria": {"a": "A", "b": "B", "c": "C"}}}}
    raw, meta = A.Shortlisted(Inner()).ask(item)
    assert seen["criteria"] == ["b", "c"]
    assert meta == {"load": "stock", "shortlist_k": 2, "shortlisted": ["intent"]}
    norm = A.normalize_answer(raw["intent"], item["questions"]["intent"])   # full question
    assert norm["probs"]["a"] == 0.0 and norm["label"] == "b"


def test_recall_at_k():
    import shortlist as SL
    assert SL.recall_at_k([["a", "b", "c"], ["c", "a", "b"]], ["b", "b"], 2) == 0.5
    assert SL.recall_at_k([["a", "b", "c"], ["c", "a", "b"]], ["b", "b"], 3) == 1.0
