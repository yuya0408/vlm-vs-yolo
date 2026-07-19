"""回帰ゲート + コストガード。CI(GitHub Actions)から呼ばれる。

実行例:
    python -m src.gate regression --current results/abc.json --baseline results/base.json
    python -m src.gate cost --month 2026-06

終了コード: 0 = pass, 1 = fail(CI がそのまま成否に使う)

結果ファイル(results/{run_id}.json)は dict 形式(src/runner.py が出力):
    {"run_id", "provider", "model_version", "eval_set_path", "eval_set_sha256",
     "created_at", "records": [{image_id, judgements, usage, latency_sec, cached}, ...]}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .analysis.metrics import compute_metrics
from .analysis.stats import mcnemar_test


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_eval_set(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def regression_gate(current_results: str, baseline_results: str, config: dict) -> bool:
    """ベースライン比で有意な劣化がないか判定する。

    macro_f1(strict) の劣化幅が max_f1_drop を超えたら fail。
    McNemar p 値は参考情報として併記する(smoke n=50 では検定力が低く、
    p 値だけだと劣化を見逃すため、fail 判定には幅を使う)。
    """
    cur = _load_json(current_results)
    base = _load_json(baseline_results)

    eval_path = cur.get("eval_set_path")
    if eval_path != base.get("eval_set_path"):
        print(f"⚠ current と baseline の eval_set_path が異なる "
              f"(current={eval_path}, baseline={base.get('eval_set_path')})。current 側で評価する。")
    eval_set = _load_eval_set(eval_path)

    cur_rep = compute_metrics(cur["records"], eval_set, mode="strict")
    base_rep = compute_metrics(base["records"], eval_set, mode="strict")
    drop = base_rep.macro_f1 - cur_rep.macro_f1
    max_drop = float(config["regression"]["max_f1_drop"])
    passed = drop <= max_drop

    print("=== regression gate (macro_f1 strict) ===")
    print(f"baseline macro_f1 = {base_rep.macro_f1:.4f}")
    print(f"current  macro_f1 = {cur_rep.macro_f1:.4f}")
    print(f"drop = {drop:+.4f}  (許容 {max_drop:.4f})")
    if config["regression"].get("report_mcnemar", False):
        mc = mcnemar_test(cur["records"], base["records"], eval_set)
        print(f"McNemar: n01(cur正/base誤)={mc['n01']} n10(cur誤/base正)={mc['n10']} "
              f"p={mc['p_value']:.4f}  ※参考")
    print("結果:", "PASS" if passed else "FAIL")
    return passed


def _result_cost_jpy(result: dict, pricing: dict) -> float:
    """1 結果ファイルの合計コスト(円)。pricing に無いモデル(YOLO/mock)は 0。"""
    model = result.get("model_version", "")
    models = pricing.get("models", {})
    rates = models.get(model)
    if not rates:
        # model_version に版サフィックス(例 gemini-3.5-flash-001)が付く場合、
        # 単価表キーの前方一致で最長マッチを採用する。
        candidates = [k for k in models if model.startswith(k)]
        if candidates:
            rates = models[max(candidates, key=len)]
    if not rates:
        return 0.0  # 単価不明(YOLO/mock 等)= コスト 0 扱い
    usd_jpy = float(pricing.get("usd_jpy", 0))
    in_rate = float(rates.get("input", 0.0))
    out_rate = float(rates.get("output", 0.0))
    total_usd = 0.0
    for rec in result.get("records", []):
        u = rec.get("usage") or {}
        total_usd += (u.get("input_tokens", 0) / 1e6) * in_rate
        total_usd += (u.get("output_tokens", 0) / 1e6) * out_rate
    return total_usd * usd_jpy


def cost_guard(month: str, config: dict, results_dir: str = "results",
               pricing_path: str = "conf/pricing.yaml") -> bool:
    """月間累積コストが上限(monthly_cost_limit_jpy)未満か判定する。

    results_dir 内の各結果ファイルの created_at が `month`(YYYY-MM)で始まるものを集計。
    超過時は False(定期評価をスキップさせる)。
    """
    with open(pricing_path, encoding="utf-8") as f:
        pricing = yaml.safe_load(f)

    total_jpy = 0.0
    n_runs = 0
    rdir = Path(results_dir)
    for path in sorted(rdir.glob("*.json")) if rdir.is_dir() else []:
        result = _load_json(str(path))
        if not str(result.get("created_at", "")).startswith(month):
            continue
        n_runs += 1
        total_jpy += _result_cost_jpy(result, pricing)

    limit = float(config["cost"]["monthly_cost_limit_jpy"])
    passed = total_jpy < limit

    print(f"=== cost guard ({month}) ===")
    print(f"対象ラン数 = {n_runs}")
    print(f"累積コスト = {total_jpy:.2f} 円  (上限 {limit:.0f} 円)")
    print("結果:", "PASS" if passed else "FAIL(以降の定期評価をスキップ)")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/gate.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    p_reg = sub.add_parser("regression")
    p_reg.add_argument("--current", required=True)
    p_reg.add_argument("--baseline", required=True)
    p_cost = sub.add_parser("cost")
    p_cost.add_argument("--month", required=True)
    p_cost.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.command == "regression":
        ok = regression_gate(args.current, args.baseline, config)
    else:
        ok = cost_guard(args.month, config, results_dir=args.results_dir)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
