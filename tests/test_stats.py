"""stats(bootstrap_ci / mcnemar_test / error_factor_regression)の単体テスト。"""

import numpy as np
import pytest

from src.analysis.stats import (
    bootstrap_ci,
    error_factor_regression,
    mcnemar_test,
    paired_bootstrap_diff,
)


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


def test_error_factor_regression_cluster_matches_coef_but_not_se():
    """クラスタ頑健版は係数(オッズ比)は通常版と同一、SE/p値だけ動く。

    1 画像に複数項目(カテゴリ)を持たせ、画像レベル定数(has_crowd)の効果を
    人工的に強く効かせる。クラスタ化で SE が動くことを、通常版との比較で確認する。
    """
    rng = np.random.default_rng(1)
    eval_set, results = [], []
    for i in range(150):
        crowd = int(rng.integers(0, 2))
        checklist = ["dog", "cat", "cow", "sheep", "bird"]  # 1 画像 5 項目 → クラスタ
        gt = {}
        judgements = []
        for cat in checklist:
            p_err = 0.8 if crowd else 0.2
            err = rng.random() < p_err
            gt[cat] = "present"
            judgements.append({"category": cat, "judgement": "absent" if err else "present"})
        eval_set.append({"image_id": i, "checklist": checklist, "ground_truth": gt,
                         "meta": {"min_bbox_area_ratio": 0.5, "has_crowd": crowd}})
        results.append({"image_id": i, "judgements": judgements})

    out = error_factor_regression(results, eval_set, cluster=True)
    assert out["n"] == 150 * 5
    assert out["n_clusters"] == 150
    assert "features_cluster" in out
    # 係数(オッズ比)は共分散の推定方法に依存しないので通常版と一致する
    naive_or = out["features"]["has_crowd"]["odds_ratio"]
    cluster_or = out["features_cluster"]["has_crowd"]["odds_ratio"]
    assert naive_or == pytest.approx(cluster_or, rel=1e-6)
    # クラスタ化により有効自由度が 750→150 に落ちるので SE(≒CI幅)は広がるはず
    naive_width = np.log(out["features"]["has_crowd"]["ci_high"]) - np.log(out["features"]["has_crowd"]["ci_low"])
    cluster_width = (np.log(out["features_cluster"]["has_crowd"]["ci_high"])
                     - np.log(out["features_cluster"]["has_crowd"]["ci_low"]))
    assert cluster_width > naive_width

    # cluster=False では非クラスタ版のみ(既定と同じ結果)で追加計算を省ける
    out_no_cluster = error_factor_regression(results, eval_set, cluster=False)
    assert "features_cluster" not in out_no_cluster
    assert out_no_cluster["features"]["has_crowd"]["coef"] == pytest.approx(
        out["features"]["has_crowd"]["coef"])


def test_paired_bootstrap_diff_identical_results_is_null():
    """A と B が同一予測なら diff 分布は退化して 0、CI も (0,0)、p は非有意(1.0 付近)。"""
    specs = [("present", "present"), ("present", "absent"),
             ("absent", "absent"), ("absent", "present")] * 5
    eval_set, res_a, _ = _single_item_set(specs)
    out = paired_bootstrap_diff(res_a, res_a, eval_set, n_boot=300, seed=3)
    for m in ("macro_f1", "micro_f1", "item_accuracy"):
        assert out[m]["point_diff"] == pytest.approx(0.0)
        assert out[m]["ci95"][0] == pytest.approx(0.0)
        assert out[m]["ci95"][1] == pytest.approx(0.0)
        assert out[m]["p_value_boot"] == pytest.approx(1.0)


def test_paired_bootstrap_diff_matches_marginal_point_estimates():
    """diff の point 推定値は、各モデルを compute_metrics で別々に計算した値の差と一致する。"""
    from src.analysis.metrics import compute_metrics

    specs = [("present", "present", "present"), ("present", "present", "absent"),
             ("absent", "absent", "present"), ("absent", "absent", "absent")] * 5
    eval_set, res_a, res_b = _single_item_set(specs)
    out = paired_bootstrap_diff(res_a, res_b, eval_set, n_boot=50, seed=4)

    rep_a = compute_metrics(res_a, eval_set, mode="strict")
    rep_b = compute_metrics(res_b, eval_set, mode="strict")
    assert out["macro_f1"]["point_diff"] == pytest.approx(rep_b.macro_f1 - rep_a.macro_f1)
    assert out["micro_f1"]["point_diff"] == pytest.approx(rep_b.micro_f1 - rep_a.micro_f1)
    assert out["item_accuracy"]["point_diff"] == pytest.approx(rep_b.item_accuracy - rep_a.item_accuracy)


def test_paired_bootstrap_diff_rejects_mismatched_image_ids():
    eval_set, res_a, _ = _single_item_set([("present", "present")] * 4)
    bad_eval_set = eval_set[:-1]  # image_id 集合を意図的に不一致にする
    with pytest.raises(ValueError):
        paired_bootstrap_diff(res_a, res_a, bad_eval_set, n_boot=10, seed=1)
