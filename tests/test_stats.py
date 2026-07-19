"""stats(bootstrap_ci / mcnemar_test / error_factor_regression)の単体テスト。"""

import numpy as np
import pytest

from src.analysis.stats import bootstrap_ci, error_factor_regression, mcnemar_test


def _single_item_set(specs):
    """specs: list of (gt, pred_a[, pred_b])。画像1枚=1項目(dog)で組み立てる。"""
    eval_set, res_a, res_b = [], [], []
    for i, spec in enumerate(specs):
        gt = spec[0]
        eval_set.append({"image_id": i, "checklist": ["dog"],
                         "ground_truth": {"dog": gt}})
        res_a.append({"image_id": i, "judgements": [{"category": "dog", "judgement": spec[1]}]})
        if len(spec) > 2:
            res_b.append({"image_id": i, "judgements": [{"category": "dog", "judgement": spec[2]}]})
    return eval_set, res_a, res_b


def test_bootstrap_ci_perfect_is_degenerate():
    """全問正解なら macro_f1=1.0 で CI は (1.0, 1.0)。"""
    eval_set, res_a, _ = _single_item_set([("present", "present")] * 8)
    lo, hi = bootstrap_ci(res_a, eval_set, n_boot=300, seed=1)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


def test_bootstrap_ci_bounds():
    """混在ケースでは 0<=lo<=hi<=1。"""
    specs = [("present", "present"), ("present", "absent"),
             ("absent", "absent"), ("absent", "present")] * 5
    eval_set, res_a, _ = _single_item_set(specs)
    lo, hi = bootstrap_ci(res_a, eval_set, n_boot=300, seed=2)
    assert 0.0 <= lo <= hi <= 1.0


def test_mcnemar_counts_and_exact():
    """n01 / n10 が設計どおり数えられ、exact 経路で p 値が出る。"""
    specs = (
        [("present", "present", "present")] * 10  # both correct
        + [("present", "present", "absent")] * 8   # a正解 b不正解 → n01
        + [("present", "absent", "present")] * 2   # a不正解 b正解 → n10
        + [("present", "absent", "absent")] * 10   # both wrong
    )
    eval_set, res_a, res_b = _single_item_set(specs)
    out = mcnemar_test(res_a, res_b, eval_set)
    assert out["n01"] == 8
    assert out["n10"] == 2
    assert 0.0 <= out["p_value"] <= 1.0


def test_error_factor_regression_recovers_sign():
    """小さい bbox 面積比ほど誤りが増えるよう作ると、係数が負になる。"""
    rng = np.random.default_rng(0)
    eval_set, results = [], []
    for i in range(400):
        area = float(rng.uniform(0.0, 1.0))
        crowd = int(rng.integers(0, 2))
        # 面積が小さいほど error 確率が高い
        p_err = 0.85 if area < 0.5 else 0.15
        err = rng.random() < p_err
        gt = "present"
        pred = "absent" if err else "present"  # gt=present なので absent が誤り
        eval_set.append({"image_id": i, "checklist": ["dog"],
                         "ground_truth": {"dog": gt},
                         "meta": {"min_bbox_area_ratio": area, "has_crowd": crowd}})
        results.append({"image_id": i, "judgements": [{"category": "dog", "judgement": pred}]})

    out = error_factor_regression(results, eval_set)
    assert out["n"] == 400
    assert "min_bbox_area_ratio" in out["features"]
    # 面積↑ → 誤り↓ なので係数は負
    assert out["features"]["min_bbox_area_ratio"]["coef"] < 0
    if out["converged"]:
        assert out["features"]["min_bbox_area_ratio"]["p_value"] < 0.05
    # 2 特徴(area, crowd)あるので VIF が出る
    assert "min_bbox_area_ratio" in out["vif"]
