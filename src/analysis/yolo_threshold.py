"""YOLO の信頼度しきい値チューニング(公平性のための steelman)。

既定 conf=0.25 は YOLO に最良とは限らない。低 conf で 1 回だけ全検出を保存し、後処理で
しきい値をスイープする。再推論は不要(無料)なので、グリッドの追加・変更のコストがゼロになる。

**グリッドが低域に密なのは意図的**: 有無判定への還元では bbox 精度が悪い低信頼検出にも
IoU 由来の FP 罰則が無いため、既定 0.25 よりずっと低い側が最適になりうる。0.05 刻みだけの
粗いグリッドでは最適点を跨いでしまう(report/REPORT.md §2)。

**操作点は VLM を一切参照せずに選ぶ**(「テストで選んでテストで報告した」批判を封じる):
1. 標準慣行 — 検出器の運用しきい値は F1 のピークを採る(既定の選定指標は micro-F1)
2. リークなし分割 — tune で選び test で評価する
3. argmax 一致 — 独立な第 2 基準(item accuracy)の argmax が同じ点に収束するかを機械的に確認

macro-F1 の argmax は退化点(低 conf 側の recall 振り切り)に落ちやすいので既定の選定指標に
しない。3 基準がどこに落ちたかは analyze() の "operating_point" にそのまま出す。

実行例:
    # 1) 全検出を保存(YOLO をローカル推論。数分・無料)
    python -m src.analysis.yolo_threshold capture \
        --eval-set data/eval_set.json --weights yolo26x.pt \
        --out results/yolo_raw_detections.json
    # 2) しきい値スイープ + 操作点選定
    python -m src.analysis.yolo_threshold sweep \
        --raw results/yolo_raw_detections.json --eval-set data/eval_set.json \
        --out results/yolo_threshold.json
    # 3) 選んだ操作点で「ランと同じ形式」の結果を書き出す
    #    (compare / error_taxonomy がそのまま食える = 主比較 tuned YOLO の入力)
    python -m src.analysis.yolo_threshold export \
        --raw results/yolo_raw_detections.json --eval-set data/eval_set.json \
        --threshold 0.075 --out results/yolo-tuned-0.075.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .metrics import compute_metrics, count_recall, item_accuracy

# 低域を密に取るグリッド(0.005〜0.75)。既定 0.25 は比較用に必ず含める。
DEFAULT_THRESHOLDS = [0.005, 0.01, 0.025, 0.05, 0.075, 0.10,
                      0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75]

# 「妥当なしきい値域」= 結論の頑健性をバンドで示す区間(REPORT §1 のバンド表)。
DEFAULT_BAND = (0.01, 0.10)

# 操作点の選定指標。macro-F1 は低 conf の退化点に張り付くため既定にしない(モジュール docstring)。
SELECTION_METRIC = "micro_f1"
# 独立な第 2 基準。これの argmax が SELECTION_METRIC と一致するかで「収束」を判定する。
AGREEMENT_METRIC = "item_accuracy"

CURVE_METRICS = ("macro_f1", "micro_f1", "item_accuracy", "count_recall")


def capture_detections(eval_set: list[dict], weights: str, images_dir: str,
                       base_conf: float = 0.001) -> dict:
    """各画像をローカル推論し {image_id: {category: [confidences]}} を返す。

    併せて画像ごとの推論レイテンシも記録する。export() が「ランと同じ形式」を書き出す際に
    実測値を載せるため(レイテンシを後から捏造しないための実測記録)。
    """
    import time

    from ..providers.yolo import YoloProvider
    provider = YoloProvider(weights=weights, conf_threshold=base_conf)
    raw: dict[str, dict[str, list[float]]] = {}
    latencies: dict[str, float] = {}
    for e in eval_set:
        path = Path(images_dir) / e["file_name"]
        t0 = time.time()
        raw[str(e["image_id"])] = provider.detect_confidences(path, base_conf=base_conf)
        latencies[str(e["image_id"])] = round(time.time() - t0, 4)
    return {"weights": weights, "base_conf": base_conf,
            "detections": raw, "latencies": latencies}


def _records_at_threshold(raw: dict, eval_set: list[dict], t: float) -> list[dict]:
    """しきい値 t で present/absent + count を再構成した records を作る。"""
    det = raw["detections"]
    records = []
    for e in eval_set:
        confs_by_cat = det.get(str(e["image_id"]), {})
        judgements = []
        for cat in e["checklist"]:
            n = sum(1 for c in confs_by_cat.get(cat, []) if c >= t)
            judgements.append({"category": cat,
                               "judgement": "present" if n > 0 else "absent",
                               "count": n})
        records.append({"image_id": e["image_id"], "judgements": judgements})
    return records


def _safe_count_recall(records: list[dict], eval_set: list[dict]) -> float | None:
    """個数 recall。gt_counts が無い評価セットでは None(スイープを止めない)。"""
    try:
        cr = count_recall(records, eval_set)
    except ValueError:
        return None
    return cr["recall"] if cr["n_gt_instances"] > 0 else None


def sweep(raw: dict, eval_set: list[dict], thresholds: list[float]) -> list[dict]:
    """各しきい値での指標カーブを返す。

    REPORT §2 のスイープ表と同じ 4 指標を出す: macro-F1 / micro-F1 / item accuracy /
    個数 recall(副軸)。閾値選定を F1 系だけに依存させないため item accuracy を含める。
    """
    out = []
    for t in thresholds:
        records = _records_at_threshold(raw, eval_set, t)
        rep = compute_metrics(records, eval_set, mode="strict")
        cr = _safe_count_recall(records, eval_set)
        out.append({
            "threshold": round(t, 4),
            "macro_f1": round(rep.macro_f1, 4),
            "micro_f1": round(rep.micro_f1, 4),
            "item_accuracy": round(item_accuracy(records, eval_set), 4),
            "count_recall": None if cr is None else round(cr, 4),
        })
    return out


def _argmax(curve: list[dict], metric: str = SELECTION_METRIC) -> dict:
    """metric を最大化する点。同値なら低いしきい値を採る(決定的)。"""
    if metric not in CURVE_METRICS:
        raise ValueError(f"未対応の metric: {metric!r}")
    scored = [c for c in curve if c.get(metric) is not None]
    if not scored:
        raise ValueError(f"{metric} が全点で欠損しており argmax が取れない")
    return max(scored, key=lambda d: (d[metric], -d["threshold"]))


def band_summary(curve: list[dict], band: tuple[float, float] = DEFAULT_BAND) -> dict:
    """バンド内での各指標の最小〜最大。結論を点推定に依存させないための材料(REPORT §1)。"""
    lo, hi = band
    inside = [c for c in curve if lo <= c["threshold"] <= hi]
    if not inside:
        raise ValueError(f"バンド {band} に含まれるしきい値がグリッドに無い")
    out: dict = {"low": lo, "high": hi, "n_points": len(inside)}
    for m in CURVE_METRICS:
        vals = [c[m] for c in inside if c.get(m) is not None]
        out[m] = [min(vals), max(vals)] if vals else None
    return out


def tune_test_split(eval_set: list[dict], test_frac: float = 0.6, seed: int = 42):
    """評価セットを tune/test に決定的分割する。"""
    idx = np.arange(len(eval_set))
    np.random.default_rng(seed).shuffle(idx)
    n_test = int(round(len(eval_set) * test_frac))
    test_idx = set(idx[:n_test].tolist())
    tune = [e for i, e in enumerate(eval_set) if i not in test_idx]
    test = [e for i, e in enumerate(eval_set) if i in test_idx]
    return tune, test


def analyze(raw: dict, eval_set: list[dict], thresholds: list[float] | None = None,
            test_frac: float = 0.6, seed: int = 42,
            band: tuple[float, float] = DEFAULT_BAND,
            selection_metric: str = SELECTION_METRIC) -> dict:
    """スイープ + 操作点選定 + バンド要約。VLM の結果は一切参照しない。"""
    thresholds = list(thresholds if thresholds is not None else DEFAULT_THRESHOLDS)

    full_curve = sweep(raw, eval_set, thresholds)

    # 全データでの各指標の argmax(「3 本の線がどこに落ちるか」の材料)
    argmax_full = {m: _argmax(full_curve, m) for m in CURVE_METRICS
                   if any(c.get(m) is not None for c in full_curve)}

    # リーク回避: tune で選び test で評価する
    tune, test = tune_test_split(eval_set, test_frac, seed)
    tune_curve = sweep(raw, tune, thresholds)
    best_on_tune = _argmax(tune_curve, selection_metric)
    tuned_t = best_on_tune["threshold"]
    test_at_tuned = sweep(raw, test, [tuned_t])[0]
    test_default = sweep(raw, test, [0.25])[0]
    test_curve = sweep(raw, test, thresholds)
    test_optimal = _argmax(test_curve, selection_metric)

    # 操作点: 3 つの独立な線が収束するか(収束しない場合もそのまま出す)
    line_standard = argmax_full[selection_metric]["threshold"]     # 1. 標準慣行(F1 ピーク)
    line_split = tuned_t                                          # 2. リークなし分割
    line_agreement = (argmax_full[AGREEMENT_METRIC]["threshold"]   # 3. 独立な第 2 基準
                      if AGREEMENT_METRIC in argmax_full else None)
    converged = (line_standard == line_split == line_agreement)

    return {
        "thresholds": thresholds,
        "selection_metric": selection_metric,
        "full_curve": full_curve,
        "argmax_full": argmax_full,
        "default_0.25_full": sweep(raw, eval_set, [0.25])[0],
        "band": band_summary(full_curve, band),
        "operating_point": {
            "threshold": line_split,
            "converged": converged,
            "lines": {
                "standard_practice_argmax": line_standard,
                "leak_free_tune_split": line_split,
                f"agreement_{AGREEMENT_METRIC}_argmax": line_agreement,
            },
        },
        "split": {
            "test_frac": test_frac, "seed": seed,
            "n_tune": len(tune), "n_test": len(test),
            "best_threshold_on_tune": tuned_t,
            "test_at_tuned": test_at_tuned,          # リークなしの正当な値
            "test_at_default_0.25": test_default,
            "test_optimal": test_optimal,            # test 上の到達可能上限(参考)
        },
    }


def export_records(raw: dict, eval_set: list[dict], threshold: float,
                   eval_set_path: str, eval_set_sha256: str) -> dict:
    """指定しきい値の判定を runner と同じ結果形式で返す。

    これが tuned YOLO を主比較に載せるための入口。src/analysis/compare.py と
    src/analysis/error_taxonomy.py は runner の出力形式しか受け取らないため、
    スイープ結果をその形式に変換して初めて「tuned YOLO vs VLM」が計算できる。
    """
    records = _records_at_threshold(raw, eval_set, threshold)
    latencies = raw.get("latencies", {})
    weights = raw.get("weights", "yolo")

    out_records = []
    for r in records:
        out_records.append({
            "image_id": r["image_id"],
            "judgements": r["judgements"],
            # ローカル推論なのでトークン消費は無い(コスト 0 の根拠)
            "usage": {"input_tokens": 0, "output_tokens": 0},
            # capture 時の実測レイテンシ。しきい値は後処理なので推論時間は変わらない。
            "latency_sec": float(latencies.get(str(r["image_id"]), 0.0)),
            "cached": False,
        })

    slug = str(threshold).replace(".", "_")
    return {
        "run_id": f"yolo-tuned-{slug}-{eval_set_sha256[:8]}",
        "provider": "yolo",
        "model_version": f"{Path(weights).stem}@conf={threshold}",
        "eval_set_path": eval_set_path,
        "eval_set_sha256": eval_set_sha256,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "threshold": threshold,
        "derived_from": "yolo_threshold.export(raw detections)",
        "records": out_records,
    }


def _print_sweep_summary(res: dict) -> None:
    d = res["default_0.25_full"]
    s = res["split"]
    op = res["operating_point"]
    b = res["band"]
    sm = res["selection_metric"]

    print(f"既定 conf=0.25 (全データ): macro-F1={d['macro_f1']} micro-F1={d['micro_f1']} "
          f"item-acc={d['item_accuracy']}")
    print(f"操作点 conf={op['threshold']} (選定指標={sm}, 3 基準の収束={op['converged']})")
    for name, t in op["lines"].items():
        print(f"    {name}: conf={t}")
    print(f"リークなし: tune({s['n_tune']}) で conf={s['best_threshold_on_tune']} を選び "
          f"test({s['n_test']}) で {sm}={s['test_at_tuned'][sm]} "
          f"(test の既定 0.25 = {s['test_at_default_0.25'][sm]})")
    print(f"バンド conf {b['low']}〜{b['high']} ({b['n_points']} 点): "
          f"macro-F1 {b['macro_f1']} / micro-F1 {b['micro_f1']} / item-acc {b['item_accuracy']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_cap = sub.add_parser("capture", help="低 conf で 1 回だけ推論し全検出を保存する")
    p_cap.add_argument("--eval-set", required=True)
    p_cap.add_argument("--weights", default="yolo26x.pt")
    p_cap.add_argument("--images-dir", default="data/coco/val2017")
    p_cap.add_argument("--base-conf", type=float, default=0.001)
    p_cap.add_argument("--out", default="results/yolo_raw_detections.json")

    p_sw = sub.add_parser("sweep", help="しきい値スイープと操作点選定(再推論なし)")
    p_sw.add_argument("--raw", default="results/yolo_raw_detections.json")
    p_sw.add_argument("--eval-set", required=True)
    p_sw.add_argument("--out", default="results/yolo_threshold.json")
    p_sw.add_argument("--test-frac", type=float, default=0.6)
    p_sw.add_argument("--seed", type=int, default=42)
    p_sw.add_argument("--selection-metric", default=SELECTION_METRIC, choices=CURVE_METRICS)
    p_sw.add_argument("--band", type=float, nargs=2, default=list(DEFAULT_BAND),
                      metavar=("LOW", "HIGH"))

    p_ex = sub.add_parser("export", help="指定しきい値の判定をラン形式で書き出す")
    p_ex.add_argument("--raw", default="results/yolo_raw_detections.json")
    p_ex.add_argument("--eval-set", required=True)
    p_ex.add_argument("--threshold", type=float, required=True)
    p_ex.add_argument("--out", required=True)

    args = parser.parse_args()
    eval_set = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))

    if args.command == "capture":
        raw = capture_detections(eval_set, args.weights, args.images_dir, args.base_conf)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        print(f"全検出を保存: {args.out}（{len(raw['detections'])} 画像, "
              f"base_conf={args.base_conf}）")
        return

    raw = json.loads(Path(args.raw).read_text(encoding="utf-8"))

    if args.command == "export":
        from ..runner import _sha256_file
        obj = export_records(raw, eval_set, args.threshold, args.eval_set,
                             _sha256_file(args.eval_set))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        if not raw.get("latencies"):
            print("警告: raw にレイテンシ記録が無い(古い capture)。latency_sec=0 で書き出した。"
                  " 正しい値が要るなら capture からやり直すこと。")
        print(f"tuned YOLO(conf={args.threshold})をラン形式で保存: {args.out}")
        return

    res = analyze(raw, eval_set, DEFAULT_THRESHOLDS, args.test_frac, args.seed,
                  band=tuple(args.band), selection_metric=args.selection_metric)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_sweep_summary(res)
    print(f"保存: {args.out}")


if __name__ == "__main__":
    main()
