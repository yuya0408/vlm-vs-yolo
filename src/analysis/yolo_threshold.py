"""YOLO の信頼度しきい値チューニング(公平性のための steelman)。

既定 conf=0.25 は YOLO に最良とは限らない。低 conf で 1 回だけ全検出を保存し、後処理で
しきい値をスイープして macro-F1 を最大化する最適しきい値を求める。再推論は不要(無料)。

リーク回避: 評価セットを tune/test に分割し、tune で選んだしきい値を test で評価する。
併せて「全データで選んだ楽観的上限(optimistic)」も出し、VLM がその上限すら上回るかを見る。

実行例:
    # 1) 全検出を保存(YOLO をローカル推論。数分・無料)
    python -m src.analysis.yolo_threshold capture \
        --eval-set data/eval_set.json --weights yolo26x.pt \
        --out results/yolo_raw_detections.json
    # 2) しきい値スイープ
    python -m src.analysis.yolo_threshold sweep \
        --raw results/yolo_raw_detections.json --eval-set data/eval_set.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .metrics import compute_metrics, count_recall as _count_recall


def capture_detections(eval_set: list[dict], weights: str, images_dir: str,
                       base_conf: float = 0.001) -> dict:
    """各画像をローカル推論し {image_id: {category: [confidences]}} を返す。"""
    from ..providers.yolo import YoloProvider
    provider = YoloProvider(weights=weights, conf_threshold=base_conf)
    raw: dict[str, dict[str, list[float]]] = {}
    for e in eval_set:
        path = Path(images_dir) / e["file_name"]
        raw[str(e["image_id"])] = provider.detect_confidences(path, base_conf=base_conf)
    return {"weights": weights, "base_conf": base_conf, "detections": raw}


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


def sweep(raw: dict, eval_set: list[dict], thresholds: list[float]) -> list[dict]:
    """各しきい値での macro-F1・micro-F1・item accuracy・個数recall(strict)を返す。"""
    out = []
    for t in thresholds:
        records = _records_at_threshold(raw, eval_set, t)
        rep = compute_metrics(records, eval_set, mode="strict")
        row = {"threshold": round(t, 4), "macro_f1": round(rep.macro_f1, 4),
               "micro_f1": round(rep.micro_f1, 4),
               "item_accuracy": round(rep.item_accuracy, 4)}
        try:
            row["count_recall"] = round(_count_recall(records, eval_set)["recall"], 4)
        except ValueError:
            pass  # count が無いデータでは副軸は計算しない
        out.append(row)
    return out


def _argmax(curve: list[dict]) -> dict:
    """macro-F1 最大点(全データで貪欲に選ぶ楽観的上限。IoU 罰則が無い有無還元では recall に
    振れた退化点になりうるため、あくまで参考値。運用しきい値の選定には _argmax_by を使う)。"""
    return max(curve, key=lambda d: d["macro_f1"])


def _argmax_by(curve: list[dict], key: str) -> dict:
    """指定した指標(micro_f1 / item_accuracy 等)の最大点。しきい値昇順の先頭を優先(タイ時に低い側)。"""
    return max(curve, key=lambda d: d[key])


def tune_test_split(eval_set: list[dict], test_frac: float = 0.6, seed: int = 42):
    """評価セットを tune/test に決定的分割する。"""
    idx = np.arange(len(eval_set))
    np.random.default_rng(seed).shuffle(idx)
    n_test = int(round(len(eval_set) * test_frac))
    test_idx = set(idx[:n_test].tolist())
    tune = [e for i, e in enumerate(eval_set) if i not in test_idx]
    test = [e for i, e in enumerate(eval_set) if i in test_idx]
    return tune, test


def analyze(raw: dict, eval_set: list[dict], thresholds: list[float],
            test_frac: float = 0.6, seed: int = 42) -> dict:
    # 1) 全データでの楽観的上限(macro-F1 を eval で直接選ぶ → IoU 罰則の無い有無還元では
    #    recall に振れた退化点になりうる。運用しきい値の根拠には使わない、参考値)。
    full_curve = sweep(raw, eval_set, thresholds)
    optimistic = _argmax(full_curve)

    # 2) 運用しきい値の選定は VLM 非依存の3基準の収束で行う(docs/DESIGN.md, report/REPORT.md §2):
    #    (a) 標準慣行: 全データの micro-F1 / item accuracy のピーク
    #    (b) リークなし分割: tune 上で micro-F1 を最大化する閾値を選び、test で評価
    #    (c) argmax一致: (a)の argmax が (b)の選定と一致するか
    operating_by_micro_f1_full = _argmax_by(full_curve, "micro_f1")
    operating_by_item_accuracy_full = _argmax_by(full_curve, "item_accuracy")

    tune, test = tune_test_split(eval_set, test_frac, seed)
    tune_curve = sweep(raw, tune, thresholds)
    best_on_tune = _argmax_by(tune_curve, "micro_f1")
    test_at_best = sweep(raw, test, [best_on_tune["threshold"]])[0]
    # 比較用: test での既定 0.25 と test 上の真の最適(micro-F1 基準)
    test_default = sweep(raw, test, [0.25])[0]
    test_curve = sweep(raw, test, thresholds)
    test_optimal = _argmax_by(test_curve, "micro_f1")

    return {
        "thresholds": thresholds,
        "full_curve": full_curve,
        "optimistic_best": optimistic,            # 全データ macro-F1 最適(上限・楽観、退化点になりうる)
        "default_0.25_full": sweep(raw, eval_set, [0.25])[0],
        "operating_point": {
            # 3基準が収束していれば、その閾値が VLM 非依存の運用しきい値として頑健。
            "standard_practice_micro_f1": operating_by_micro_f1_full["threshold"],
            "standard_practice_item_accuracy": operating_by_item_accuracy_full["threshold"],
            "leak_free_tune_selected": best_on_tune["threshold"],
            "converged": len({operating_by_micro_f1_full["threshold"],
                              operating_by_item_accuracy_full["threshold"],
                              best_on_tune["threshold"]}) == 1,
        },
        "split": {
            "test_frac": test_frac, "seed": seed,
            "n_tune": len(tune), "n_test": len(test),
            "best_threshold_on_tune": best_on_tune["threshold"],
            "test_micro_f1_at_tuned": test_at_best["micro_f1"],    # リークなしの正当な値
            "test_macro_f1_at_tuned": test_at_best["macro_f1"],
            "test_macro_f1_at_default_0.25": test_default["macro_f1"],
            "test_optimal": test_optimal,         # test 上の到達可能上限(micro-F1 基準、参考)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_cap = sub.add_parser("capture")
    p_cap.add_argument("--eval-set", required=True)
    p_cap.add_argument("--weights", default="yolo26x.pt")
    p_cap.add_argument("--images-dir", default="data/coco/val2017")
    p_cap.add_argument("--base-conf", type=float, default=0.001)
    p_cap.add_argument("--out", default="results/yolo_raw_detections.json")

    p_sw = sub.add_parser("sweep")
    p_sw.add_argument("--raw", default="results/yolo_raw_detections.json")
    p_sw.add_argument("--eval-set", required=True)
    p_sw.add_argument("--out", default="results/yolo_threshold.json")
    p_sw.add_argument("--test-frac", type=float, default=0.6)
    p_sw.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "capture":
        eval_set = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
        raw = capture_detections(eval_set, args.weights, args.images_dir, args.base_conf)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        print(f"全検出を保存: {args.out}（{len(raw['detections'])} 画像, base_conf={args.base_conf}）")
        return

    raw = json.loads(Path(args.raw).read_text(encoding="utf-8"))
    eval_set = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    # 低 conf 側を粗いグリッドのままにすると運用しきい値の真のピーク(0.075 近辺)を見逃す
    # (report/REPORT.md §2)。0.01〜0.25 は密に、それ以降は既定の 0.05 刻みのまま。
    thresholds = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25] + \
        [round(0.05 * k, 2) for k in range(6, 16)]  # 0.3〜0.75
    res = analyze(raw, eval_set, thresholds, args.test_frac, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    d = res["default_0.25_full"]
    o = res["optimistic_best"]
    op = res["operating_point"]
    s = res["split"]
    print(f"既定 conf=0.25 (全300): macro-F1={d['macro_f1']}")
    print(f"楽観的最適(macro-F1, 退化点の可能性あり, 全300で選択): "
          f"conf={o['threshold']} macro-F1={o['macro_f1']}")
    print(f"運用しきい値: 標準慣行(micro-F1)={op['standard_practice_micro_f1']} / "
          f"標準慣行(item accuracy)={op['standard_practice_item_accuracy']} / "
          f"リークなしtune選定={op['leak_free_tune_selected']} "
          f"→ 収束={'YES' if op['converged'] else 'NO'}")
    print(f"リークなし: tune({s['n_tune']})で conf={s['best_threshold_on_tune']} を選び "
          f"test({s['n_test']})で micro-F1={s['test_micro_f1_at_tuned']} "
          f"macro-F1={s['test_macro_f1_at_tuned']} "
          f"(test の既定0.25 macro-F1={s['test_macro_f1_at_default_0.25']})")
    print(f"保存: {args.out}")


if __name__ == "__main__":
    main()
