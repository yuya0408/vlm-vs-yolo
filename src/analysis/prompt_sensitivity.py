"""VLM 側のプロンプト感度検証(YOLO の閾値スイープに対する非対称な steelman)。

YOLO 側は conf(スカラー 1 次元)をグリッド全走査できるが、VLM 側の探索空間 = プロンプトには
次元も境界も無く、1 点あたり N 枚の再推論(課金)が発生するため全走査は原理的に不可能。
そこで「全走査」の代わりに **水準を数本振って結論が反転しないかだけを確認する** 最低限の検証を行う。

設計(YOLO 側と手続きを揃える):
- 選定はリークなし: `yolo_threshold.tune_test_split` と同一の分割(既定 test_frac=0.6 / seed=42)を
  使い、**tune で micro-F1 が最大のプロンプトを選び、test で報告する**。YOLO の閾値選定と同じ規則。
- 感度の主指標は「プロンプト間のばらつき(max−min)」。これが閾値を動かしたときの振れ幅(数 pt)と
  同程度なら、結論(精度は互角でコスト/レイテンシが支配)はプロンプト選択に依存しないと言える。
- YOLO の生検出(`yolo_raw_detections.json`)を渡すと、tune 選定プロンプトと baseline の両方について
  test 上で YOLO と対応あり比較を行い、**プロンプトを振ると有意判定が反転するか**を明示する。

実行例:
    # 1) プロンプト水準ごとにラン(prompt が run_id に入るので結果は上書きされない)
    for m in flash flash_deliberate flash_calibrated; do
      python -m src.runner --eval-set data/eval_set.json --model $m
    done
    # 2) 感度分析(再推論なし・無料)
    python -m src.analysis.prompt_sensitivity \
        --run results/gemini-gemini-3.5-flash-concise-289d6c0e.json \
        --run results/gemini-gemini-3.5-flash-deliberate-289d6c0e.json \
        --run results/gemini-gemini-3.5-flash-calibrated-289d6c0e.json \
        --eval-set data/eval_set.json --baseline concise \
        --yolo-raw results/yolo_raw_detections.json --yolo-conf 0.075
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import yaml

from ..gate import _result_cost_jpy
from .metrics import compute_metrics
from .stats import mcnemar_test, paired_bootstrap_diff
from .yolo_threshold import _records_at_threshold, tune_test_split


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_runs(specs: list[str]) -> dict[str, dict]:
    """`label=path` または `path` の並びを {プロンプト名: 結果dict} にする。

    label 省略時は結果ファイルの "prompt" フィールド(runner が記録)を使う。
    全ランが同一評価セット(eval_set_sha256 一致)であることを検証する。
    """
    runs: dict[str, dict] = {}
    eval_hashes = set()
    for spec in specs:
        label, _, path = spec.partition("=")
        if not path:
            label, path = "", label
        result = _load(path)
        label = label or result.get("prompt") or Path(path).stem
        if label in runs:
            raise ValueError(f"プロンプト名 {label!r} が重複している(label=path で明示指定できる)")
        runs[label] = result
        eval_hashes.add(result.get("eval_set_sha256"))
    if len(eval_hashes) > 1:
        raise ValueError(f"評価セットが不一致: {sorted(map(str, eval_hashes))}。同一 eval_set のランで比較すること。")
    if len(runs) < 2:
        raise ValueError("プロンプト感度の検証には 2 本以上のランが必要")
    return runs


def _subset(records: list[dict], ids: set[int]) -> list[dict]:
    return [r for r in records if r["image_id"] in ids]


def _uncertain_rate(records: list[dict]) -> float:
    total = sum(len(r["judgements"]) for r in records)
    unc = sum(1 for r in records for j in r["judgements"] if j["judgement"] == "uncertain")
    return round(unc / total, 4) if total else 0.0


def evaluate(records: list[dict], eval_set: list[dict]) -> dict:
    """1 ラン × 1 データ集合の指標(strict 主・excl の coverage を併記)。"""
    strict = compute_metrics(records, eval_set, mode="strict")
    excl = compute_metrics(records, eval_set, mode="excl")
    return {
        "n_images": len(eval_set),
        "macro_f1": round(strict.macro_f1, 4),
        "micro_f1": round(strict.micro_f1, 4),
        "item_accuracy": round(strict.item_accuracy, 4),
        "coverage_excl": round(excl.coverage, 4),
        "uncertain_rate": _uncertain_rate(records),
    }


def _spread(rows: dict[str, dict], key: str) -> dict:
    """プロンプト間のばらつき(max−min)。感度の主指標。"""
    vals = {name: row[key] for name, row in rows.items()}
    lo_name = min(vals, key=lambda k: vals[k])
    hi_name = max(vals, key=lambda k: vals[k])
    return {"min": vals[lo_name], "min_prompt": lo_name,
            "max": vals[hi_name], "max_prompt": hi_name,
            "range_pt": round((vals[hi_name] - vals[lo_name]) * 100, 2)}


def _argmax_prompt(rows: dict[str, dict], key: str) -> str:
    """指標最大のプロンプト。タイは名前順で決定的に解く。"""
    return min(sorted(rows), key=lambda name: (-rows[name][key], name))


def _pair_stats(records_a: list[dict], records_b: list[dict], eval_set: list[dict],
                n_boot: int, seed: int) -> dict:
    """diff = B − A の対応あり比較(画像単位ペア差ブートストラップ + McNemar)。"""
    diff = paired_bootstrap_diff(records_a, records_b, eval_set,
                                 metrics=("macro_f1", "micro_f1", "item_accuracy"),
                                 n_boot=n_boot, seed=seed)
    mc = mcnemar_test(records_a, records_b, eval_set)
    out = {"mcnemar": {"n01": mc["n01"], "n10": mc["n10"],
                       "p_value": round(mc["p_value"], 4)},
           "significant_at_0.05": bool(mc["p_value"] < 0.05)}
    for m, d in diff.items():
        out[m] = {"point_diff_pt": round(d["point_diff"] * 100, 2),
                  "ci95_pt": [round(d["ci95"][0] * 100, 2), round(d["ci95"][1] * 100, 2)],
                  "p_value_boot": round(d["p_value_boot"], 4)}
    return out


def analyze(runs: dict[str, dict], eval_set: list[dict], baseline: str,
            pricing: dict | None = None, test_frac: float = 0.6, seed: int = 42,
            n_boot: int = 10_000, yolo_records: list[dict] | None = None,
            yolo_label: str = "yolo") -> dict:
    """プロンプト水準ごとの指標・ばらつき・リークなし選定・YOLO との有意判定の反転有無を返す。"""
    if baseline not in runs:
        raise ValueError(f"baseline {baseline!r} が --run に無い: {sorted(runs)}")
    pricing = pricing or {}

    tune, test = tune_test_split(eval_set, test_frac, seed)
    tune_ids = {e["image_id"] for e in tune}
    test_ids = {e["image_id"] for e in test}

    prompts: dict[str, dict] = {}
    full_rows: dict[str, dict] = {}
    tune_rows: dict[str, dict] = {}
    test_rows: dict[str, dict] = {}
    for name, result in runs.items():
        records = result["records"]
        full_rows[name] = evaluate(records, eval_set)
        tune_rows[name] = evaluate(_subset(records, tune_ids), tune)
        test_rows[name] = evaluate(_subset(records, test_ids), test)
        lat = [r["latency_sec"] for r in records]
        prompts[name] = {
            "run_id": result.get("run_id"),
            "model_version": result.get("model_version"),
            "full": full_rows[name],
            "tune": tune_rows[name],
            "test": test_rows[name],
            "cost_jpy": round(_result_cost_jpy(result, pricing), 2),
            "latency_median_sec": round(statistics.median(lat), 4) if lat else 0.0,
        }

    # リークなし選定: tune で micro-F1 最大のプロンプトを選び、test で報告(YOLO 閾値と同じ規則)
    selected = _argmax_prompt(tune_rows, "micro_f1")
    selection = {
        "rule": "tune 上の micro-F1 最大(YOLO の閾値選定と同一規則)",
        "test_frac": test_frac, "seed": seed,
        "n_tune": len(tune), "n_test": len(test),
        "selected_prompt": selected,
        "selected_is_baseline": selected == baseline,
        "baseline_prompt": baseline,
        "test_at_selected": test_rows[selected],
        "test_at_baseline": test_rows[baseline],
        # 参考: test 上で直接最良のプロンプト(到達可能上限。選定には使わない)
        "test_optimal_prompt": _argmax_prompt(test_rows, "micro_f1"),
    }

    spread = {
        scope: {m: _spread(rows, m) for m in ("macro_f1", "micro_f1", "item_accuracy")}
        for scope, rows in (("full", full_rows), ("tune", tune_rows), ("test", test_rows))
    }
    spread["uncertain_rate_full"] = _spread(full_rows, "uncertain_rate")

    # プロンプト間の対応あり比較(全データ, diff = 各水準 − baseline)
    base_records = runs[baseline]["records"]
    vs_baseline = {
        name: _pair_stats(base_records, runs[name]["records"], eval_set, n_boot, seed)
        for name in runs if name != baseline
    }

    out = {
        "eval_set_sha256": runs[baseline].get("eval_set_sha256"),
        "n_images": len(eval_set),
        "n_prompts": len(runs),
        "prompts": prompts,
        "selection": selection,
        "spread": spread,
        "vs_baseline": vs_baseline,
        "total_cost_jpy": round(sum(p["cost_jpy"] for p in prompts.values()), 2),
        "params": {"n_boot": n_boot, "seed": seed, "test_frac": test_frac},
    }

    # YOLO との対応あり比較を test 上で: プロンプトを振ると有意判定が反転するか
    if yolo_records is not None:
        y_test = _subset(yolo_records, test_ids)
        vs_yolo = {}
        for name in {baseline, selected}:
            vs_yolo[name] = _pair_stats(y_test, _subset(runs[name]["records"], test_ids),
                                        test, n_boot, seed)  # diff = VLM − YOLO
        out["vs_yolo_on_test"] = {
            "yolo": yolo_label,
            "baseline": vs_yolo[baseline],
            "selected": vs_yolo[selected],
            # 結論(= 有意差の有無)がプロンプト選択で変わるか。これが False なら
            # 「プロンプトを振っても結論は反転しない」と言える。
            "verdict_changed": (vs_yolo[baseline]["significant_at_0.05"]
                                != vs_yolo[selected]["significant_at_0.05"]),
        }
    return out


def to_markdown(res: dict) -> str:
    sel = res["selection"]
    lines = [
        f"## VLM プロンプト感度({res['n_prompts']} 水準, N={res['n_images']}, "
        f"eval {str(res['eval_set_sha256'])[:8]})",
        "",
        "| プロンプト | macro-F1 (全) | micro-F1 (全) | item acc (全) | uncertain率 | "
        "micro-F1 (tune) | micro-F1 (test) | 円 | 中央値s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, p in res["prompts"].items():
        mark = "**" if name == sel["selected_prompt"] else ""
        lines.append(
            f"| {mark}{name}{mark} | {p['full']['macro_f1']} | {p['full']['micro_f1']} | "
            f"{p['full']['item_accuracy']} | {p['full']['uncertain_rate']} | "
            f"{p['tune']['micro_f1']} | {p['test']['micro_f1']} | {p['cost_jpy']} | "
            f"{p['latency_median_sec']} |")

    sp = res["spread"]["full"]
    lines += [
        "",
        f"**プロンプト間のばらつき(全{res['n_images']})**: "
        f"macro-F1 {sp['macro_f1']['range_pt']}pt "
        f"({sp['macro_f1']['min_prompt']} {sp['macro_f1']['min']} → "
        f"{sp['macro_f1']['max_prompt']} {sp['macro_f1']['max']}) / "
        f"micro-F1 {sp['micro_f1']['range_pt']}pt / "
        f"item accuracy {sp['item_accuracy']['range_pt']}pt",
        "",
        f"**リークなし選定**: tune({sel['n_tune']})の micro-F1 最大 = `{sel['selected_prompt']}` "
        f"→ test({sel['n_test']}) で macro-F1={sel['test_at_selected']['macro_f1']} "
        f"micro-F1={sel['test_at_selected']['micro_f1']} "
        f"(baseline `{sel['baseline_prompt']}` の test: "
        f"macro-F1={sel['test_at_baseline']['macro_f1']} "
        f"micro-F1={sel['test_at_baseline']['micro_f1']})",
        "",
        f"### baseline `{sel['baseline_prompt']}` との対応あり差(全データ, diff = 各水準 − baseline)",
        "",
        "| プロンプト | macro-F1 差 | micro-F1 差 | item acc 差 | McNemar p |",
        "|---|---|---|---|---|",
    ]
    for name, d in res["vs_baseline"].items():
        lines.append(
            f"| {name} | {d['macro_f1']['point_diff_pt']:+.2f}pt "
            f"[{d['macro_f1']['ci95_pt'][0]:+.2f}, {d['macro_f1']['ci95_pt'][1]:+.2f}] | "
            f"{d['micro_f1']['point_diff_pt']:+.2f}pt | "
            f"{d['item_accuracy']['point_diff_pt']:+.2f}pt | "
            f"{d['mcnemar']['p_value']} |")

    if "vs_yolo_on_test" in res:
        vy = res["vs_yolo_on_test"]
        lines += [
            "",
            f"### YOLO(`{vy['yolo']}`)との対応あり比較(test のみ, diff = VLM − YOLO)",
            "",
            "| VLM プロンプト | macro-F1 差 | item acc 差 | McNemar p | 有意(5%) |",
            "|---|---|---|---|---|",
        ]
        for key, label in (("baseline", sel["baseline_prompt"]),
                           ("selected", sel["selected_prompt"])):
            d = vy[key]
            lines.append(
                f"| {label}({key}) | {d['macro_f1']['point_diff_pt']:+.2f}pt "
                f"[{d['macro_f1']['ci95_pt'][0]:+.2f}, {d['macro_f1']['ci95_pt'][1]:+.2f}] | "
                f"{d['item_accuracy']['point_diff_pt']:+.2f}pt | {d['mcnemar']['p_value']} | "
                f"{'YES' if d['significant_at_0.05'] else 'NO'} |")
        lines += [
            "",
            f"**有意判定の反転: {'あり(結論はプロンプト選択に依存する)' if vy['verdict_changed'] else 'なし(結論はプロンプト選択に依存しない)'}**",
        ]

    lines += [
        "",
        f"> 追加コスト合計: {res['total_cost_jpy']} 円({res['n_prompts']} 水準 × N={res['n_images']})。",
        "> プロンプト空間は次元も境界も定義できないため、これは全走査ではなく水準サンプリングである"
        "(YOLO の閾値グリッド全走査とは非対称)。",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="[LABEL=]PATH",
                        help="プロンプト水準ごとの結果 JSON。LABEL 省略時は結果の prompt を使う")
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--baseline", default="concise")
    parser.add_argument("--pricing", default="conf/pricing.yaml")
    parser.add_argument("--yolo-raw", default=None,
                        help="YOLO の生検出 JSON(渡すと test 上で YOLO と比較する)")
    parser.add_argument("--yolo-conf", type=float, default=0.075,
                        help="YOLO の運用しきい値(REPORT §2 の選定値)")
    parser.add_argument("--out", default="results/prompt_sensitivity.json")
    parser.add_argument("--test-frac", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=10_000)
    args = parser.parse_args()

    runs = load_runs(args.run)
    eval_set = _load(args.eval_set)
    pricing = yaml.safe_load(Path(args.pricing).read_text(encoding="utf-8")) \
        if Path(args.pricing).exists() else {}

    yolo_records = None
    yolo_label = "yolo"
    if args.yolo_raw:
        raw = _load(args.yolo_raw)
        yolo_records = _records_at_threshold(raw, eval_set, args.yolo_conf)
        yolo_label = f"{raw.get('weights', 'yolo')}@conf={args.yolo_conf}"

    res = analyze(runs, eval_set, args.baseline, pricing=pricing, test_frac=args.test_frac,
                  seed=args.seed, n_boot=args.n_boot, yolo_records=yolo_records,
                  yolo_label=yolo_label)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(to_markdown(res))
    print(f"\n保存: {args.out}")


if __name__ == "__main__":
    main()
