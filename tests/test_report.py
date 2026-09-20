import json

import pytest

from quorum import report


def _rec(qid, ans, labels=None):
    r = {
        "state": "s",
        "questions": {qid: {"type": ans["type"], "instructions": "x"}},
        "answers": {qid: ans},
    }
    if labels is not None:
        r["labels"] = {qid: labels}
    return r


# ---------- prob_of: the committed answer, not the distribution argmax ----------

def test_prob_of_noul_yes():
    ans = {"type": "noul", "noul": 0.8, "answer": "yes",
           "probabilities": {"yes": 0.8, "no": 0.2}}
    assert report.prob_of(ans) == pytest.approx((0.8, "yes"))


def test_prob_of_noul_no_uses_complement():
    # contract field stays P(yes)=0.1; the committed answer is "no" -> conf 0.9
    ans = {"type": "noul", "noul": 0.1, "answer": "no",
           "probabilities": {"yes": 0.1, "no": 0.9}}
    p, chosen = report.prob_of(ans)
    assert chosen == "no"
    assert p == pytest.approx(0.9)


def test_prob_of_noul_legacy_record_without_answer_field():
    # old log lines have no "answer" — fall back to the yes-side reading
    ans = {"type": "noul", "noul": 0.7, "probabilities": {"yes": 0.7, "no": 0.3}}
    assert report.prob_of(ans) == pytest.approx((0.7, "yes"))


def test_prob_of_choice_uses_committed_option():
    ans = {"type": "choice", "choice": "tech",
           "probabilities": {"billing": 0.7, "tech": 0.3}}
    assert report.prob_of(ans) == pytest.approx((0.3, "tech"))


def test_prob_of_score_uses_claimed_level():
    ans = {"type": "score", "score": 1.0,
           "probabilities": {"0": 0.6, "1": 0.3, "2": 0.1}}
    assert report.prob_of(ans) == pytest.approx((0.3, "1"))


def test_prob_of_missing_probs():
    assert report.prob_of({"type": "noul", "noul": None}) == (None, None)


# ---------- extract_pairs: label resolution ----------

def test_extract_pairs_agreement_and_error():
    recs = [
        _rec("q", {"type": "noul", "noul": 0.9, "answer": "yes",
                   "probabilities": {"yes": 0.9, "no": 0.1}}, "yes"),
        _rec("q", {"type": "noul", "noul": 0.95, "answer": "yes",
                   "probabilities": {"yes": 0.95, "no": 0.05}}, "no"),
    ]
    pairs, stats = report.extract_pairs(recs)
    assert [p["ok"] for p in pairs] == [1.0, 0.0]
    assert stats["labeled_records"] == 2


def test_extract_pairs_bool_label_matches_no():
    recs = [_rec("q", {"type": "noul", "noul": 0.05, "answer": "no",
                       "probabilities": {"yes": 0.05, "no": 0.95}}, False)]
    pairs, _ = report.extract_pairs(recs)
    assert pairs[0]["ok"] == 1.0


def test_extract_pairs_graded_score_label():
    recs = [_rec("m", {"type": "score", "score": 2.0,
                       "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}, 0.4)]
    pairs, _ = report.extract_pairs(recs)
    assert pairs[0]["ok"] == pytest.approx(0.4)


def test_extract_pairs_skips_unlabeled_and_legacy_single_label():
    legacy = {"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}},
              "answers": {"q": {"type": "noul", "noul": 0.6, "answer": "yes",
                                "probabilities": {"yes": 0.6, "no": 0.4}}},
              "label": "yes"}
    unl = _rec("q", {"type": "noul", "noul": 0.6, "answer": "yes",
                     "probabilities": {"yes": 0.6, "no": 0.4}})
    pairs, stats = report.extract_pairs([legacy, unl])
    assert len(pairs) == 1
    assert stats["unlabeled"] == 1


def test_extract_pairs_multi_question_record():
    rec = {"state": "s",
           "questions": {"a": {"type": "noul", "instructions": "x"},
                         "b": {"type": "noul", "instructions": "y"}},
           "answers": {"a": {"type": "noul", "noul": 0.8, "answer": "yes",
                             "probabilities": {"yes": 0.8, "no": 0.2}},
                       "b": {"type": "noul", "noul": 0.3, "answer": "no",
                             "probabilities": {"yes": 0.3, "no": 0.7}}},
           "labels": {"a": "yes", "b": "yes"}}
    pairs, _ = report.extract_pairs([rec])
    by = {p["qid"]: p for p in pairs}
    assert by["a"]["ok"] == 1.0 and by["b"]["ok"] == 0.0


# ---------- metrics ----------

def test_perfectly_calibrated_group_low_brier_ece():
    # 100 binary items: bucket b (p in [b/10,(b+1)/10)) has exactly b+1 yeses
    # out of 10 — mean p == observed rate per bucket => ECE ~ 0.
    recs = []
    for b in range(10):
        for i in range(10):
            p = (b + 0.5) / 10
            ok = "yes" if i < b + 1 else "no"
            recs.append(_rec("q", {"type": "noul", "noul": p, "answer": "yes",
                                   "probabilities": {"yes": p, "no": 1 - p}}, ok))
    rep = report.audit(recs)
    q = rep["questions"]["q"]
    assert q["brier"] < 0.25
    # bucket midpoints deviate ±0.05 from bucket edges -> ECE floor is 0.05
    assert q["ece"] < 0.06


def test_majority_floor_and_confident_errors():
    recs = [_rec("q", {"type": "noul", "noul": 0.99, "answer": "yes",
                       "probabilities": {"yes": 0.99, "no": 0.01}}, "no")] * 3
    recs += [_rec("q", {"type": "noul", "noul": 0.99, "answer": "yes",
                        "probabilities": {"yes": 0.99, "no": 0.01}}, "yes")]
    rep = report.audit(recs, threshold=0.9)
    q = rep["questions"]["q"]
    assert q["n_confident_errors"] == 3
    assert q["majority_floor"] == 0.75
    assert q["acc_at_0.5"] == 0.25  # always-yes decision loses to the floor


def test_audit_qid_filter_and_min_n():
    recs = [_rec("a", {"type": "noul", "noul": 0.9, "answer": "yes",
                       "probabilities": {"yes": 0.9, "no": 0.1}}, "yes")] * 5
    recs += [_rec("b", {"type": "noul", "noul": 0.9, "answer": "yes",
                        "probabilities": {"yes": 0.9, "no": 0.1}}, "yes")]
    assert set(report.audit(recs)["questions"]) == {"a", "b"}
    assert set(report.audit(recs, qid_filter="b")["questions"]) == {"b"}
    assert set(report.audit(recs, min_n=5)["questions"]) == {"a"}


def test_render_smoke():
    recs = [_rec("q", {"type": "noul", "noul": 0.9, "answer": "yes",
                       "probabilities": {"yes": 0.9, "no": 0.1}}, "yes")]
    text = report.render(report.audit(recs))
    assert "== q" in text and "brier" in text
