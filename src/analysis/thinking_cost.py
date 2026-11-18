"""思考(thinking)トークンの費用を、300 枚の本番ランに足す推定。

2026-06 の本番ラン(300 枚)は思考のトークンを記録していない。Gemini は思考を出力と同じ単価で
課金するので、REPORT の 251 円は過小になる。300 枚を回し直す代わりに、同じ評価セットから 30 枚を
抜き出して(seed 固定)思考を記録したランを取り、次のように推定する。

    思考 ÷ 出力の比 R = 30 枚の思考トークンの合計 ÷ 30 枚の出力トークンの合計
    思考の費用(300 枚) = R × 本番ランの出力トークンの合計 × 出力の単価

入力・出力は本番ランの記録をそのまま使い、標本からは比 R だけを借りる。区間は 30 枚を画像単位で
引き直すブートストラップ。標本ランの応答の版は本番ランと違い得る(別名 gemini-3.5-flash が
2026-11 には gemini-3.6-flash を返した)ので、判定が本番ランとどれだけ一致したかも併せて出す。

実行例:
    python -m src.analysis.thinking_cost
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

_N_BOOT = 10_000


def _usage_sum(records: list[dict], key: str) -> int:
    return sum(int(r["usage"].get(key, 0)) for r in records)


def thinking_ratio(records: list[dict]) -> float:
    """思考トークン ÷ 出力トークン(合計どうしの比)。"""
    out = _usage_sum(records, "output_tokens")
    if out <= 0:
        raise ValueError("出力トークンが 0")
    return _usage_sum(records, "thoughts_tokens") / out


def bootstrap_ratio(records: list[dict], n_boot: int = _N_BOOT, seed: int = 0) -> tuple[float, float]:
    """比 R の 95% 区間(画像単位のパーセンタイル・ブートストラップ)。"""
    rng = random.Random(seed)
    n = len(records)
    ratios = sorted(thinking_ratio([records[rng.randrange(n)] for _ in range(n)])
                    for _ in range(n_boot))
    return ratios[int(0.025 * n_boot)], ratios[int(0.975 * n_boot) - 1]


def judgement_agreement(sample: list[dict], base: list[dict]) -> tuple[int, int]:
    """標本ランの判定のうち、本番ランの同じ画像・同じ項目と一致した数と総数。"""
    by_id = {r["image_id"]: {j["category"]: j["judgement"] for j in r["judgements"]} for r in base}
    agree = total = 0
    for r in sample:
        ref = by_id[r["image_id"]]
        for j in r["judgements"]:
            total += 1
            agree += ref.get(j["category"]) == j["judgement"]
    return agree, total


def estimate(base: dict, sample: dict, price: dict, usd_jpy: float) -> dict:
    base_recs, sample_recs = base["records"], sample["records"]
    missing = {r["image_id"] for r in sample_recs} - {r["image_id"] for r in base_recs}
    if missing:
        raise ValueError(f"本番ランに無い画像が標本にある: {sorted(missing)}")

    def jpy(tokens: float, rate: float) -> float:
        return tokens / 1e6 * rate * usd_jpy

    in_tok, out_tok = _usage_sum(base_recs, "input_tokens"), _usage_sum(base_recs, "output_tokens")
    recorded = jpy(in_tok, price["input"]) + jpy(out_tok, price["output"])
    r = thinking_ratio(sample_recs)
    lo, hi = bootstrap_ratio(sample_recs)
    thinking = jpy(r * out_tok, price["output"])
    n = len(base_recs)
    agree, total = judgement_agreement(sample_recs, base_recs)
    return {
        "base_run": base["run_id"],
        "sample_run": sample["run_id"],
        "sample_model_version": sample.get("model_version"),
        "n_base": n,
        "n_sample": len(sample_recs),
        "thinking_to_output_ratio": round(r, 3),
        "ratio_ci95": [round(lo, 3), round(hi, 3)],
        "recorded_cost_jpy": round(recorded, 1),
        "thinking_cost_jpy": round(thinking, 1),
        "thinking_cost_ci95_jpy": [round(jpy(lo * out_tok, price["output"]), 1),
                                   round(jpy(hi * out_tok, price["output"]), 1)],
        "total_cost_jpy": round(recorded + thinking, 1),
        "unit_jpy_per_image": round((recorded + thinking) / n, 3),
        "unit_ci95_jpy_per_image": [round((recorded + jpy(lo * out_tok, price["output"])) / n, 3),
                                    round((recorded + jpy(hi * out_tok, price["output"])) / n, 3)],
        "judgement_agreement": [agree, total],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="results/gemini-gemini-3.5-flash-289d6c0e.json")
    parser.add_argument("--sample", default="results/gemini-gemini-3.5-flash-concise-eb0e9a0b.json")
    parser.add_argument("--pricing", default="conf/pricing.yaml")
    parser.add_argument("--model", default="gemini-3.5-flash", help="pricing.yaml の単価のキー")
    parser.add_argument("--out", default="results/thinking_cost.json")
    args = parser.parse_args()

    pricing = yaml.safe_load(Path(args.pricing).read_text(encoding="utf-8"))
    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    sample = json.loads(Path(args.sample).read_text(encoding="utf-8"))
    res = estimate(base, sample, pricing["models"][args.model], float(pricing["usd_jpy"]))
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"保存: {args.out}")


if __name__ == "__main__":
    main()
