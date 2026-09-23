import math

import pytest

import metrics as M


def test_accuracy_counts_missing_pred_as_wrong():
    assert M.accuracy(["a", "b", "a", "c"], ["a", None, "b", "c"]) == 0.5


def test_majority_floor():
    assert M.majority_floor(["x", "x", "y", "x"]) == 0.75


def test_macro_f1_hand_computed():
    g = ["a", "a", "b", "b"]
    p = ["a", "b", "b", "b"]
    # a: tp1 fp0 fn1 -> 2/3 ; b: tp2 fp1 fn0 -> 4/5
    assert M.macro_f1(g, p) == pytest.approx((2 / 3 + 4 / 5) / 2)


def test_brier_perfect_and_worst():
    opts = ["a", "b"]
    assert M.brier(M.onehot("a", opts), {"a": 1.0, "b": 0.0}, opts) == 0
    assert M.brier(M.onehot("a", opts), {"a": 0.0, "b": 1.0}, opts) == 2


def test_kl_zero_when_equal_and_positive_otherwise():
    opts = ["a", "b", "c"]
    g = {"a": 0.5, "b": 0.3, "c": 0.2}
    assert M.kl(g, g, opts) == pytest.approx(0, abs=1e-9)
    assert M.kl(g, {"a": 1.0, "b": 0.0, "c": 0.0}, opts) > 5  # confident miss is expensive but finite


def test_tv_and_soft_agreement():
    opts = ["a", "b"]
    g = {"a": 0.7, "b": 0.3}
    p = {"a": 0.4, "b": 0.6}
    assert M.tv(g, p, opts) == pytest.approx(0.3)
    assert M.soft_agreement(g, p, opts) == pytest.approx(0.7 * 0.4 + 0.3 * 0.6)


def test_normalize_restricts_and_renormalises():
    assert M.normalize({"a": 2, "b": 2, "zzz": 4}, ["a", "b"]) == {"a": 0.5, "b": 0.5}
    assert M.normalize(None, ["a"]) is None
    assert M.normalize({"a": 0}, ["a"]) is None


def test_ece_perfect_calibration_is_zero():
    conf = [0.8] * 10
    correct = [True] * 8 + [False] * 2
    assert M.ece(conf, correct) == pytest.approx(0)
    assert M.ece([1.0] * 4, [False] * 4) == pytest.approx(1.0)


def test_auc_known_values_and_ties():
    assert M.auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert M.auc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0
    assert M.auc([0.5, 0.5], [True, False]) == 0.5
    assert math.isnan(M.auc([0.1, 0.2], [True, True]))


def test_coverage_at_risk():
    # most confident 4 are right, then a wrong one: 4/5 answered at 0% err; 5/5 has 20% err
    conf = [0.99, 0.98, 0.97, 0.96, 0.5]
    corr = [True, True, True, True, False]
    assert M.coverage_at_risk(conf, corr, 0.05) == 0.8
    # a tie between a right and a wrong answer must be admitted together
    assert M.coverage_at_risk([0.9, 0.9], [True, False], 0.05) == 0.0


def test_expected_score_and_within_one():
    assert M.expected_score({"0": 0.5, "2": 0.5}) == 1.0
    assert M.expected_score(None, 3.0) == 3.0
    assert M.within_one([0, 2, 4], [1, 4, 4]) == pytest.approx(2 / 3)
    assert M.mae([1, 2], [2, 4]) == 1.5


def test_bootstrap_ci_brackets_mean():
    vals = [1.0] * 70 + [0.0] * 30
    lo, hi = M.bootstrap_ci(vals, n=500, seed=1)
    assert lo < 0.7 < hi and hi - lo < 0.25


def test_paired_bootstrap_detects_clear_winner():
    a = [1.0] * 90 + [0.0] * 10
    b = [1.0] * 50 + [0.0] * 50
    r = M.paired_bootstrap(a, b, n=500)
    assert r["diff"] == pytest.approx(0.4) and r["p_a_better"] > 0.99 and r["ci"][0] > 0


def test_platt_identity_on_calibrated_data_and_fixes_bias():
    # a model that says 0.5 for everything but is right 90% on yes: platt must lift P(yes)
    p = [0.5] * 100
    y = [True] * 90 + [False] * 10
    a, b = M.platt_fit(p, y)
    assert M.platt_apply(0.5, a, b) == pytest.approx(0.9, abs=0.01)
    # an overconfident model gets its slope pulled below 1
    p2 = [0.99] * 50 + [0.01] * 50
    y2 = [True] * 35 + [False] * 15 + [False] * 35 + [True] * 15
    a2, _ = M.platt_fit(p2, y2)
    assert 0 < a2 < 1
    assert M.platt_apply(0.99, a2, 0) == pytest.approx(0.7, abs=0.02)


def test_temperature_fit_softens_overconfidence():
    dists = [{"a": 0.99, "b": 0.01}] * 10
    golds = ["a"] * 7 + ["b"] * 3
    t = M.temperature_fit(dists, golds)
    assert t > 1
    assert M.temperature_apply({"a": 0.99, "b": 0.01}, t)["a"] == pytest.approx(0.7, abs=0.05)


def test_kfold_partitions_everything_once():
    folds = M.kfold_indices(23, 5, seed=3)
    flat = sorted(i for f in folds for i in f)
    assert flat == list(range(23)) and len(folds) == 5
