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


def error_factor_regression(results: list[dict], eval_set: list[dict]) -> dict:
    """誤答 ~ bbox面積比 + 項目数 + crowd + (positive) のロジスティック回帰。

    1 行 = 1 判定項目。目的変数 error(0/1, strict)。説明変数は eval_set の meta から構築。
    係数の解釈が目的なので sklearn でなく statsmodels Logit を使い、オッズ比・CI・p 値を返す。
    多重共線性チェックのため VIF も返す。
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

    converged = True
    conf = None
    try:
        fit = sm.Logit(y, X).fit(disp=0)
        params, pvalues = fit.params, fit.pvalues
        conf = fit.conf_int()
    except Exception:  # 完全分離・特異など → L2 正則化にフォールバック(p 値は出ない)
        converged = False
        fit = sm.Logit(y, X).fit_regularized(disp=0)
        params = fit.params
        pvalues = {k: float("nan") for k in params.index}

    features = {}
    for f in feats:
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

    vif = {}
    if len(feats) >= 2:
        Xv = X.values
        for i, name in enumerate(X.columns):
            if name == "const":
                continue
            vif[name] = float(variance_inflation_factor(Xv, i))

    return {"n": int(len(df)), "converged": converged, "features": features, "vif": vif}
