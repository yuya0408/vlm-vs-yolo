"""M4: YOLO vs VLM 比較分析ドライバ。

2 つのラン結果(YOLO / VLM)を同一評価セットで突き合わせ、比較に必要な統計を一括算出する。

算出する内容:
- 各モデル: strict/excl 指標(P/R/F1・coverage)、macro/micro-F1 のブートストラップ CI、
  個数 recall(副軸)、コスト(円)、レイテンシ(中央値/p95)、uncertain 率
- 両モデル間: McNemar 検定(対応あり)
- 各モデル: 誤り要因のロジスティック回帰(誤答 ~ 項目数 + bbox面積比 + crowd + gt_present)
- パレート: 精度(macro-F1 strict)× コスト × レイテンシ

実行例:
    python -m src.analysis.compare \
        --yolo results/yolo-yolo26x-289d6c0e.json \
        --vlm  results/gemini-gemini-3.5-flash-289d6c0e.json \
        --eval-set data/eval_set.json \
        --out results/comparison.json

出力: results/comparison.json(構造化)+ 標準出力に Markdown 要約。
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import yaml

from ..gate import _result_cost_jpy
from .metrics import compute_metrics
from .stats import bootstrap_ci, error_factor_regression, mcnemar_test, paired_bootstrap_diff


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _latency_stats(records: list[dict]) -> dict:
    """レイテンシの代表値。外れ値に強い中央値・p95 を主に使う。"""
    lat = sorted(r["latency_sec"] for r in records)
    if not lat:
        return {"median": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    p95 = lat[min(len(lat) - 1, int(round(0.95 * len(lat))) - 1)]
    return {
        "median": round(statistics.median(lat), 4),
        "p95": round(float(p95), 4),
        "max": round(max(lat), 4),
        "mean": round(statistics.mean(lat), 4),
    }


def _uncertain_stats(records: list[dict]) -> dict:
    """uncertain 率(VLM だけが表現できる不確実性の観察用)。"""
    total = uncertain = 0
    for r in records:
        for j in r["judgements"]:
            total += 1
            if j["judgement"] == "uncertain":
                uncertain += 1
    rate = uncertain / total if total else 0.0
    return {"uncertain": uncertain, "total_judgements": total, "rate": round(rate, 4)}


def _safe_count_recall(records: list[dict], eval_set: list[dict]):
    """個数 recall(副軸)。count を出さないモデルでは None。"""
    from .metrics import count_recall
    try:
        cr = count_recall(records, eval_set)
        return {"recall": round(cr["recall"], 4), "n_gt_instances": cr["n_gt_instances"]}
    except ValueError:
        return None


def _model_summary(result: dict, eval_set: list[dict], pricing: dict,
                   n_boot: int, seed: int) -> dict:
    records = result["records"]
    strict = compute_metrics(records, eval_set, mode="strict")
    excl = compute_metrics(records, eval_set, mode="excl")

    macro_ci = bootstrap_ci(records, eval_set, metric="macro_f1", n_boot=n_boot, seed=seed)
    micro_ci = bootstrap_ci(records, eval_set, metric="micro_f1", n_boot=n_boot, seed=seed)

    try:
        regression = error_factor_regression(records, eval_set)
    except Exception as exc:  # noqa: BLE001 — レポートには算出可否を残す
        regression = {"error": str(exc)}

    return {
        "run_id": result.get("run_id"),
        "provider": result.get("provider"),
        "model_version": result.get("model_version"),
        "strict": {
            "macro_f1": round(strict.macro_f1, 4),
            "micro_f1": round(strict.micro_f1, 4),
            "coverage": round(strict.coverage, 4),
            "item_accuracy": round(strict.item_accuracy, 4),
        },
        "excl": {
            "macro_f1": round(excl.macro_f1, 4),
            "micro_f1": round(excl.micro_f1, 4),
            "coverage": round(excl.coverage, 4),
            "item_accuracy": round(excl.item_accuracy, 4),
        },
        "macro_f1_ci95": [round(macro_ci[0], 4), round(macro_ci[1], 4)],
        "micro_f1_ci95": [round(micro_ci[0], 4), round(micro_ci[1], 4)],
        "count_recall": _safe_count_recall(records, eval_set),
        "uncertain": _uncertain_stats(records),
        "cost_jpy": round(_result_cost_jpy(result, pricing), 2),
        "latency_sec": _latency_stats(records),
        "error_factor_regression": regression,
    }


def compare(yolo_path: str, vlm_path: str, eval_set_path: str,
            pricing_path: str = "conf/pricing.yaml",
            n_boot: int = 10_000, seed: int = 42) -> dict:
    """YOLO と VLM の比較サマリ(dict)を返す。"""
    yolo = _load(yolo_path)
    vlm = _load(vlm_path)
    eval_set = _load(eval_set_path)
    pricing = yaml.safe_load(Path(pricing_path).read_text(encoding="utf-8"))

    # 同一評価セットでの比較であることを保証(McNemar は対応ありが前提)
    if yolo.get("eval_set_sha256") != vlm.get("eval_set_sha256"):
        raise ValueError(
            f"評価セットが不一致: yolo={yolo.get('eval_set_sha256')} "
            f"vlm={vlm.get('eval_set_sha256')}。同一 eval_set のランで比較すること。")

    yolo_sum = _model_summary(yolo, eval_set, pricing, n_boot, seed)
    vlm_sum = _model_summary(vlm, eval_set, pricing, n_boot, seed)

    # McNemar(YOLO vs VLM、項目単位・strict 正誤)。主推論ではなく補助的なクロスチェック
    # (主推論は下の paired_diff。docs/DESIGN.md §4)。
    mc = mcnemar_test(yolo["records"], vlm["records"], eval_set)

    # 主推論: 画像単位ペア差ブートストラップ(diff = VLM − YOLO)。2つの周辺CIの重複は
    # 差の非有意を意味しないため、同一リサンプル添字で差そのものの分布を作る。
    paired_diff = paired_bootstrap_diff(
        yolo["records"], vlm["records"], eval_set,
        metrics=("macro_f1", "micro_f1", "item_accuracy"), n_boot=n_boot, seed=seed)

    # パレート: 精度(macro-F1 strict) × コスト × レイテンシ(中央値)
    pareto = {
        "yolo": {"macro_f1": yolo_sum["strict"]["macro_f1"],
                 "cost_jpy": yolo_sum["cost_jpy"],
                 "latency_median": yolo_sum["latency_sec"]["median"]},
        "vlm": {"macro_f1": vlm_sum["strict"]["macro_f1"],
                "cost_jpy": vlm_sum["cost_jpy"],
                "latency_median": vlm_sum["latency_sec"]["median"]},
    }

    return {
        "eval_set_path": eval_set_path,
        "eval_set_sha256": yolo.get("eval_set_sha256"),
        "n_images": len(eval_set),
        "models": {"yolo": yolo_sum, "vlm": vlm_sum},
        "mcnemar": mc,
        "paired_diff": paired_diff,
        "pareto": pareto,
        "params": {"n_boot": n_boot, "seed": seed},
    }


def to_markdown(summary: dict) -> str:
    y = summary["models"]["yolo"]
    v = summary["models"]["vlm"]
    mc = summary["mcnemar"]
    pd = summary["paired_diff"]

    def row(label, yv, vv):
        return f"| {label} | {yv} | {vv} |"

    def diff_row(label, metric):
        d = pd[metric]
        return (f"| {label} | {d['point_diff']*100:+.2f}pt | "
                f"[{d['ci95'][0]*100:+.2f}, {d['ci95'][1]*100:+.2f}]pt |")

    lines = [
        f"## YOLO vs VLM 比較(N={summary['n_images']}, eval {summary['eval_set_sha256'][:8]})",
        "",
        f"- YOLO: `{y['model_version']}` / VLM: `{v['model_version']}`",
        "",
        "| 指標 | YOLO | VLM |",
        "|---|---|---|",
        row("macro-F1 (strict)",
            f"{y['strict']['macro_f1']} [{y['macro_f1_ci95'][0]}, {y['macro_f1_ci95'][1]}]",
            f"{v['strict']['macro_f1']} [{v['macro_f1_ci95'][0]}, {v['macro_f1_ci95'][1]}]"),
        row("micro-F1 (strict)", y["strict"]["micro_f1"], v["strict"]["micro_f1"]),
        row("item accuracy (strict)", y["strict"]["item_accuracy"], v["strict"]["item_accuracy"]),
        row("macro-F1 (excl)", y["excl"]["macro_f1"], v["excl"]["macro_f1"]),
        row("coverage (excl)", y["excl"]["coverage"], v["excl"]["coverage"]),
        row("個数 recall(副軸)",
            (y["count_recall"] or {}).get("recall", "—"),
            (v["count_recall"] or {}).get("recall", "—")),
        row("uncertain 率", y["uncertain"]["rate"], v["uncertain"]["rate"]),
        row("コスト(円)", y["cost_jpy"], v["cost_jpy"]),
        row("レイテンシ中央値(s)", y["latency_sec"]["median"], v["latency_sec"]["median"]),
        row("レイテンシ p95(s)", y["latency_sec"]["p95"], v["latency_sec"]["p95"]),
        "",
        "### 主推論: 画像単位ペア差ブートストラップ(diff = VLM − YOLO, docs/DESIGN.md §4)",
        "",
        "| 指標 | 差(点推定) | CI95 |",
        "|---|---|---|",
        diff_row("macro-F1", "macro_f1"),
        diff_row("micro-F1", "micro_f1"),
        diff_row("item accuracy", "item_accuracy"),
        "",
        f"**McNemar(対応あり, strict, 補助的クロスチェック)**: n01(YOLO正/VLM誤)={mc['n01']}, "
        f"n10(YOLO誤/VLM正)={mc['n10']}, p={mc['p_value']:.4g} "
        "(item accuracy diff は代数的に (n10−n01)/N と一致=クラスタ対応版 McNemar 相当)",
        "",
        "### uncertain について(方針A)",
        f"VLM の uncertain 率は {v['uncertain']['rate']*100:.2f}%"
        f"({v['uncertain']['uncertain']}/{v['uncertain']['total_judgements']})。"
        "汎用 VLM は uncertain という出口を持つが校正された不確実性としてはほぼ使わない"
        "(過信)。YOLO は conf 閾値で coverage を連続的に振れるのと対照的。",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo", required=True)
    parser.add_argument("--vlm", required=True)
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--pricing", default="conf/pricing.yaml")
    parser.add_argument("--out", default="results/comparison.json")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = compare(args.yolo, args.vlm, args.eval_set, pricing_path=args.pricing,
                      n_boot=args.n_boot, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(to_markdown(summary))
    print(f"\n比較結果を保存: {args.out}")


if __name__ == "__main__":
    main()
