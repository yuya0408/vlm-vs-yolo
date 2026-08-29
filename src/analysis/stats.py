"""統計分析: ブートストラップ CI / McNemar 検定 / 誤り要因回帰。"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .metrics import _pred_lookup, _prf, confusion_counts


def _per_image_counts(results: list[dict], eval_set: list[dict]) -> list[dict[str, dict[str, int]]]:
    """画像 1 枚ごとのカテゴリ別 confusion セルを返す(ブートストラップの単位)。"""
    eval_by_id = {e["image_id"]: e for e in eval_set}
    per_image = []
    for r in results:
        e = eval_by_id[r["image_id"]]
        per_image.append(confusion_counts([r], [e]))
    return per_image


def _f1_from_counts(total: dict[str, dict[str, int]], metric: str, mode: str = "strict") -> float:
    cats = sorted(total)
    micro_tp = micro_fp = micro_fn = 0
    f1s = []
    for cat in cats:
        c = total[cat]
        tp, fp = c["pp"], c["ap"]
        fn = c["pa"] + c["pu"] if mode == "strict" else c["pa"]
        _, _, f1 = _prf(tp, fp, fn)
        support = c["pp"] + c["pa"] + c["pu"]
        if support > 0:
            f1s.append(f1)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
    if metric == "macro_f1":
        return sum(f1s) / len(f1s) if f1s else 0.0
    if metric == "micro_f1":
        return _prf(micro_tp, micro_fp, micro_fn)[2]
    raise ValueError(f"未対応の metric: {metric!r}")


def bootstrap_ci(results: list[dict], eval_set: list[dict], metric: str = "macro_f1",
                 n_boot: int = 10_000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    """画像単位のブートストラップで指標の percentile 信頼区間を返す。

    リサンプリング単位は「画像」(項目単位だと同一画像内の相関を無視するため)。
    """
    per_image = _per_image_counts(results, eval_set)
    n = len(per_image)
    if n == 0:
        raise ValueError("results が空")
    rng = np.random.default_rng(seed)

    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        total: dict[str, dict[str, int]] = defaultdict(
            lambda: {k: 0 for k in ("pp", "pa", "pu", "ap", "aa", "au")})
        for i in idx:
            for cat, cells in per_image[i].items():
                for k, v in cells.items():
                    total[cat][k] += v
        boots[b] = _f1_from_counts(dict(total), metric)

    lower = float(np.percentile(boots, 100 * alpha / 2))
    upper = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lower, upper


_CELLS = ("pp", "pa", "pu", "ap", "aa", "au")


def _per_image_counts_by_id(results: list[dict], eval_set: list[dict]) -> dict[int, dict[str, dict[str, int]]]:
    per_image = _per_image_counts(results, eval_set)
    return {r["image_id"]: cells for r, cells in zip(results, per_image)}


def _stack_counts(results: list[dict], eval_set: list[dict],
                  image_ids: list[int], cats: list[str]) -> np.ndarray:
    """(n_images, n_cats, 6) の confusion セル配列を image_ids/cats の順序で作る。"""
    by_id = _per_image_counts_by_id(results, eval_set)
    arr = np.zeros((len(image_ids), len(cats), 6), dtype=np.float64)
    for i, img_id in enumerate(image_ids):
        cat_counts = by_id[img_id]
        for j, cat in enumerate(cats):
            c = cat_counts.get(cat)
            if c is not None:
                arr[i, j, :] = [c[k] for k in _CELLS]
    return arr


def _metrics_from_totals(total: np.ndarray) -> dict[str, np.ndarray]:
    """total: (..., n_cats, 6) → macro_f1 / micro_f1 / item_accuracy を同じ先頭shapeで返す(strict)。

    compute_metrics(mode="strict") / _f1_from_counts と同じ定義(0除算は 0 扱い)。
    """
    pp, pa, pu, ap, aa, au = (total[..., k] for k in range(6))
    tp, fp, fn = pp, ap, pa + pu
    support = pp + pa + pu

    def _safe_div(num, den):
        return np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den > 0)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    scored = support > 0
    n_scored = scored.sum(axis=-1)
    macro_f1 = _safe_div(np.where(scored, f1, 0.0).sum(axis=-1), n_scored)

    micro_tp, micro_fp, micro_fn = tp.sum(axis=-1), fp.sum(axis=-1), fn.sum(axis=-1)
    micro_p = _safe_div(micro_tp, micro_tp + micro_fp)
    micro_r = _safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1 = _safe_div(2 * micro_p * micro_r, micro_p + micro_r)

    correct_items = (pp + aa).sum(axis=-1)
    total_items = total.sum(axis=(-1, -2))
    item_accuracy = _safe_div(correct_items, total_items)

    return {"macro_f1": macro_f1, "micro_f1": micro_f1, "item_accuracy": item_accuracy}


def paired_bootstrap_diff(results_a: list[dict], results_b: list[dict], eval_set: list[dict],
                          metrics: tuple[str, ...] = ("macro_f1", "micro_f1", "item_accuracy"),
                          n_boot: int = 10_000, alpha: float = 0.05, seed: int = 42) -> dict:
    """同一リサンプル添字(画像単位)で diff = B の指標 − A の指標 の分布を作る。

    marginal な bootstrap_ci を2回別々に取って CI の重複を見るのは、差の有意性の判定として
    誤り(重複していても差が有意でない保証にも、有意である保証にもならない)。ペアで同じ
    リサンプルを使うことで、画像単位の相関を保ったまま差そのものの分布を得る。

    呼び出し規約: diff = B − A。例: paired_bootstrap_diff(yolo, vlm, ...) なら diff = VLM − YOLO。

    返り値: {metric: {"point_diff", "ci95": (lo,hi), "ci90": (lo,hi), "p_value_boot"}}
    p_value_boot は両側 p 値(片側の一方が 0 件のときは floor 2/(n_boot+1) を使う。タイ
    (diff==0)は両側に算入するため保守的)。
    """
    ids_a = {r["image_id"] for r in results_a}
    ids_b = {r["image_id"] for r in results_b}
    eval_ids = {e["image_id"] for e in eval_set}
    if ids_a != eval_ids or ids_b != eval_ids:
        raise ValueError("results_a / results_b / eval_set の image_id 集合が一致しない")

    image_ids = sorted(eval_ids)
    cats = sorted({cat for e in eval_set for cat in e["checklist"]})

    arr_a = _stack_counts(results_a, eval_set, image_ids, cats)
    arr_b = _stack_counts(results_b, eval_set, image_ids, cats)

    n = len(image_ids)
    point_a = _metrics_from_totals(arr_a.sum(axis=0))
    point_b = _metrics_from_totals(arr_b.sum(axis=0))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    weights = np.empty((n_boot, n), dtype=np.float64)
    for b in range(n_boot):
        weights[b] = np.bincount(idx[b], minlength=n)

    flat_a = arr_a.reshape(n, -1)
    flat_b = arr_b.reshape(n, -1)
    total_a = (weights @ flat_a).reshape(n_boot, len(cats), 6)
    total_b = (weights @ flat_b).reshape(n_boot, len(cats), 6)

    boot_a = _metrics_from_totals(total_a)
    boot_b = _metrics_from_totals(total_b)

    out = {}
    for m in metrics:
        diffs = boot_b[m] - boot_a[m]
        point_diff = float(point_b[m] - point_a[m])

        n_le0 = int(np.sum(diffs <= 0))
        n_ge0 = int(np.sum(diffs >= 0))
        p_boot = min(1.0, 2 * (min(n_le0, n_ge0) + 1) / (n_boot + 1))

        ci95 = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
        ci90 = (float(np.percentile(diffs, 5.0)), float(np.percentile(diffs, 95.0)))

        out[m] = {
            "point_diff": point_diff,
            "ci95": ci95,
            "ci90": ci90,
            "p_value_boot": p_boot,
        }
    return out


def _correct_map(results: list[dict], eval_set: list[dict]) -> dict[tuple[int, str], bool]:
    """(image_id, category) → 正解かどうか(strict: pred==gt、uncertain は不正解)。"""
    preds = _pred_lookup(results, eval_set)
    eval_by_id = {e["image_id"]: e for e in eval_set}
    out: dict[tuple[int, str], bool] = {}
    for image_id, pmap in preds.items():
        gt = eval_by_id[image_id]["ground_truth"]
        for cat, pred in pmap.items():
            out[(image_id, cat)] = (pred == gt[cat])
    return out


def mcnemar_test(results_a: list[dict], results_b: list[dict], eval_set: list[dict]) -> dict:
    """2 つのラン(同一評価セット)の対応ありの差を McNemar 検定で評価する。

    項目単位で n01 = a正解/b不正解、n10 = a不正解/b正解 を数える。
    n01 + n10 < 25 のとき exact(二項検定)、それ以外は近似(継続補正あり)。
    """
    from statsmodels.stats.contingency_tables import mcnemar

    ca = _correct_map(results_a, eval_set)
    cb = _correct_map(results_b, eval_set)
    if set(ca) != set(cb):
        raise ValueError("results_a と results_b の (image_id, category) 集合が一致しない")

    both_correct = n01 = n10 = both_wrong = 0
    for key in ca:
        a, b = ca[key], cb[key]
        if a and b:
            both_correct += 1
        elif a and not b:
            n01 += 1
        elif not a and b:
            n10 += 1
        else:
            both_wrong += 1

    table = [[both_correct, n01], [n10, both_wrong]]
    discordant = n01 + n10
    res = mcnemar(table, exact=discordant < 25, correction=True)
    return {
        "n01": n01,
        "n10": n10,
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
    }


def _fit_logit(y, X, cov_type: str | None, groups=None):
    """Logit を1回フィットし、(features辞書の元になる要素, converged) を返す。

    完全分離・特異などで通常の MLE が失敗した場合は L2 正則化にフォールバックする
    (この場合 p 値・CI は出せない = converged=False)。
    """
    import statsmodels.api as sm

    converged = True
    conf = None
    try:
        if cov_type == "cluster":
            fit = sm.Logit(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups}, disp=0)
        else:
            fit = sm.Logit(y, X).fit(disp=0)
        params, pvalues = fit.params, fit.pvalues
        conf = fit.conf_int()
    except Exception:  # 完全分離・特異など → L2 正則化にフォールバック(p 値は出ない)
        converged = False
        fit = sm.Logit(y, X).fit_regularized(disp=0)
        params = fit.params
        pvalues = {k: float("nan") for k in params.index}

    features = {}
    for f in X.columns:
        if f == "const":
            continue
        coef = float(params[f])
        lo = float(conf.loc[f, 0]) if conf is not None else float("nan")
        hi = float(conf.loc[f, 1]) if conf is not None else float("nan")
        features[f] = {
            "coef": coef,
            "odds_ratio": float(np.exp(coef)),
            "ci_low": float(np.exp(lo)) if conf is not None else float("nan"),
            "ci_high": float(np.exp(hi)) if conf is not None else float("nan"),
            "p_value": float(pvalues[f]),
        }
    return features, converged


def error_factor_regression(results: list[dict], eval_set: list[dict], cluster: bool = True) -> dict:
    """誤答 ~ bbox面積比 + 項目数 + crowd + (positive) のロジスティック回帰。

    1 行 = 1 判定項目。目的変数 error(0/1, strict)。説明変数は eval_set の meta から構築。
    係数の解釈が目的なので sklearn でなく statsmodels Logit を使い、オッズ比・CI・p 値を返す。
    多重共線性チェックのため VIF も返す。

    観測単位は項目だが、n_items / min_bbox_area_ratio / has_crowd は画像レベルの定数で
    同一画像内の行が独立でないため、既定(cluster=True)ではクラスタ(image_id)頑健標準誤差の
    版も並行して計算し "features_cluster" / "converged_cluster" に返す(補正の効果を見るため
    通常版 "features" はそのまま残す)。cluster=False で通常版のみ計算しコストを省ける。
    """
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    preds = _pred_lookup(results, eval_set)
    eval_by_id = {e["image_id"]: e for e in eval_set}

    rows = []
    for image_id, pmap in preds.items():
        e = eval_by_id[image_id]
        gt = e["ground_truth"]
        meta = e.get("meta", {})
        n_items = len(e["checklist"])
        area = float(meta.get("min_bbox_area_ratio", 0.0))
        crowd = int(bool(meta.get("has_crowd", False)))
        for cat, pred in pmap.items():
            rows.append({
                "image_id": image_id,
                "error": int(pred != gt[cat]),
                "n_items": n_items,
                "min_bbox_area_ratio": area,
                "has_crowd": crowd,
                "gt_present": int(gt[cat] == "present"),
            })

    df = pd.DataFrame(rows)
    candidate = ["n_items", "min_bbox_area_ratio", "has_crowd", "gt_present"]
    feats = [f for f in candidate if df[f].nunique() > 1]  # 分散ゼロの列は落とす
    if not feats:
        raise ValueError("説明変数に分散が無く回帰できない")

    X = sm.add_constant(df[feats], has_constant="add")
    y = df["error"]
    groups = df["image_id"].values

    features, converged = _fit_logit(y, X, cov_type=None)

    out = {"n": int(len(df)), "converged": converged, "features": features}

    if cluster:
        features_cluster, converged_cluster = _fit_logit(y, X, cov_type="cluster", groups=groups)
        out["features_cluster"] = features_cluster
        out["converged_cluster"] = converged_cluster
        out["n_clusters"] = int(df["image_id"].nunique())

    vif = {}
    if len(feats) >= 2:
        Xv = X.values
        for i, name in enumerate(X.columns):
            if name == "const":
                continue
            vif[name] = float(variance_inflation_factor(Xv, i))
    out["vif"] = vif

    return out
