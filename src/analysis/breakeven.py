"""VLM(従量課金)と YOLO(固定費先行)の損益分岐分析。

REPORT のコスト比較は「300 枚あたり 251 円 vs 0 円」で、YOLO 側が無料に見える。しかし YOLO の
0 円は API 課金が 0 という意味でしかなく、実際には**アノテーション・学習工数・推論基盤の運用**が
固定費として先に立つ。実務の技術選定で効くのはこの固定費と従量課金の交点である。

    YOLO 累計(T ヶ月) = C_fix + ops_monthly × T
    VLM  累計(T ヶ月) = unit_jpy × 月間枚数 M × T
    C_fix = ラベル枚数 × 1枚あたり矩形数 × 矩形単価 × 手戻り係数 + 学習工数 × 時間単価

固定費は環境依存で測れないため、点推定は出さない。`conf/costs.yaml` に低位/中位/高位の
3 シナリオを出典付きで置き、**結論が反転する境界**(何枚/月で逆転するか、回収に何ヶ月かかるか)を
出す。しきい値を点で決めず band で語った REPORT §2 と同じ扱い。

前提: 精度が互角であること(REPORT §1)。語彙外・ラベル無しの第二象限では YOLO は
いくら払っても要件を満たさないため、そもそも交点が存在しない(§7)。本分析は第一象限限定の道具。

実行例:
    python -m src.analysis.breakeven --costs conf/costs.yaml \
        --out results/breakeven.json --figure report/figures/breakeven.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

# 回収期間の上限(月)。これを超える / 交点が無い場合は「実質回収不能」として扱う
_HORIZON_CAP_MONTHS = 120


def vlm_unit_cost_jpy(costs: dict) -> float:
    """VLM の 1 枚あたり単価(円)。実測値から割り戻す。"""
    v = costs["vlm"]
    n = int(v["measured_n_images"])
    if n <= 0:
        raise ValueError("measured_n_images は正の整数")
    return float(v["measured_cost_jpy"]) / n


def fixed_cost(scenario: dict) -> dict:
    """1 シナリオの初期固定費(円)。ラベル費と学習工数に分けて返す。"""
    label = (float(scenario["n_label_images"]) * float(scenario["boxes_per_image"])
             * float(scenario["cost_per_box_jpy"]) * float(scenario.get("rework_factor", 1.0)))
    engineering = float(scenario.get("engineering_hours", 0)) * \
        float(scenario.get("engineering_rate_jpy", 0))
    return {"label_jpy": round(label, 1),
            "engineering_jpy": round(engineering, 1),
            "total_jpy": round(label + engineering, 1),
            "n_boxes": int(scenario["n_label_images"] * scenario["boxes_per_image"])}


def break_even_images(fixed_total: float, unit_jpy: float) -> float:
    """運用費を無視したときに固定費と釣り合う累計推論枚数。YOLO に最も有利な下限値。"""
    return fixed_total / unit_jpy


def break_even_monthly_volume(fixed_total: float, ops_monthly: float,
                              unit_jpy: float, months: int) -> float:
    """months ヶ月で元を取るのに必要な月間推論枚数。"""
    if months <= 0:
        raise ValueError("months は正の整数")
    return (fixed_total + ops_monthly * months) / (unit_jpy * months)


def payback_months(fixed_total: float, ops_monthly: float, unit_jpy: float,
                   monthly_volume: float) -> float | None:
    """その月間枚数で固定費を回収するのに要する月数。回収不能なら None。

    月あたりの節約額 = VLM の月額 − YOLO の運用月額。これが 0 以下なら永久に逆転しない
    (YOLO を持つこと自体の月額が、VLM を使い続ける月額を上回っている状態)。
    """
    monthly_saving = unit_jpy * monthly_volume - ops_monthly
    if monthly_saving <= 0:
        return None
    months = fixed_total / monthly_saving
    return None if months > _HORIZON_CAP_MONTHS else round(months, 1)


def analyze(costs: dict) -> dict:
    unit = vlm_unit_cost_jpy(costs)
    horizons = [int(h) for h in costs.get("horizons_months", [12])]
    volumes = [float(v) for v in costs.get("monthly_volumes", [1000])]

    scenarios = {}
    for name, sc in costs["scenarios"].items():
        fc = fixed_cost(sc)
        ops = float(sc.get("ops_monthly_jpy", 0))
        scenarios[name] = {
            "label": sc.get("label", name),
            "fixed_cost": fc,
            "ops_monthly_jpy": ops,
            "break_even_images": round(break_even_images(fc["total_jpy"], unit)),
            "break_even_monthly_volume": {
                str(h): round(break_even_monthly_volume(fc["total_jpy"], ops, unit, h))
                for h in horizons
            },
            "payback_months": {
                str(int(v)): payback_months(fc["total_jpy"], ops, unit, v) for v in volumes
            },
        }

    return {
        "vlm_unit_jpy_per_image": round(unit, 4),
        "vlm_measured": costs["vlm"],
        "horizons_months": horizons,
        "monthly_volumes": volumes,
        "horizon_cap_months": _HORIZON_CAP_MONTHS,
        "scenarios": scenarios,
        "note": ("固定費は公開値から置いた仮定であり実測ではない。点推定ではなく"
                 "反転境界として読むこと。精度互角(第一象限)が前提。"),
    }


def _fmt(n: float | None) -> str:
    return "回収不能" if n is None else f"{n:,.0f}"


def to_markdown(res: dict) -> str:
    unit = res["vlm_unit_jpy_per_image"]
    lines = [
        f"## 損益分岐(VLM 従量課金 {unit:.2f} 円/枚 vs YOLO 固定費先行)",
        "",
        "| シナリオ | ラベル費 | 学習工数 | 固定費 計 | 運用/月 | 損益分岐の累計枚数 |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in res["scenarios"].items():
        fc = s["fixed_cost"]
        lines.append(
            f"| `{name}` | {fc['label_jpy']:,.0f} 円 | {fc['engineering_jpy']:,.0f} 円 | "
            f"**{fc['total_jpy']:,.0f} 円** | {s['ops_monthly_jpy']:,.0f} 円 | "
            f"**{s['break_even_images']:,} 枚** |")

    lines += ["", "### 回収に必要な月間推論枚数(運用費込み)", "",
              "| シナリオ | " + " | ".join(f"{h} ヶ月で回収" for h in res["horizons_months"]) + " |",
              "|---" * (len(res["horizons_months"]) + 1) + "|"]
    for name, s in res["scenarios"].items():
        cells = " | ".join(f"{s['break_even_monthly_volume'][str(h)]:,} 枚/月"
                           for h in res["horizons_months"])
        lines.append(f"| `{name}` | {cells} |")

    lines += ["", "### 月間枚数ごとの回収期間", "",
              "| シナリオ | " + " | ".join(f"{int(v):,} 枚/月" for v in res["monthly_volumes"]) + " |",
              "|---" * (len(res["monthly_volumes"]) + 1) + "|"]
    for name, s in res["scenarios"].items():
        cells = " | ".join(
            ("回収不能" if s["payback_months"][str(int(v))] is None
             else f"{s['payback_months'][str(int(v))]:.1f} ヶ月")
            for v in res["monthly_volumes"])
        lines.append(f"| `{name}` | {cells} |")

    lines += [
        "",
        f"> 「回収不能」= 交点が存在しない(VLM の月額 ≤ YOLO の運用月額)、または回収に "
        f"{res['horizon_cap_months']} ヶ月超。",
        f"> {res['note']}",
    ]
    return "\n".join(lines)


def breakeven_figure(res: dict, out_path: str) -> str:
    """月間推論枚数(対数)× 回収期間(月)。回収不能域は上端に貼り付けて示す。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    unit = res["vlm_unit_jpy_per_image"]
    cap = res["horizon_cap_months"]
    volumes = np.logspace(2, 6, 200)  # 100 〜 1,000,000 枚/月

    fig, ax = plt.subplots(figsize=(8, 5))
    # ラベルは英語に統一(matplotlib 既定フォントは CJK 非対応で文字化けするため)
    for name, s in res["scenarios"].items():
        fixed = s["fixed_cost"]["total_jpy"]
        ops = s["ops_monthly_jpy"]
        saving = unit * volumes - ops
        months = np.where(saving > 0, fixed / np.where(saving > 0, saving, 1), np.inf)
        ax.plot(volumes, np.clip(months, None, cap * 1.4), lw=2,
                label=f"{name}: fixed {fixed/1e6:.2f}M JPY, ops {ops/1e3:.0f}k JPY/mo")

    ax.axhline(cap, color="gray", ls="--", lw=1)
    ax.text(1.2e2, cap * 1.05, f"never pays back (> {cap} months)", fontsize=9, color="gray")
    ax.axvspan(100, 1000, color="#d62728", alpha=0.08)
    ax.text(1.1e2, 1.6, "PoC scale", fontsize=9, color="#d62728")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1, cap * 1.6)
    ax.set_xlabel("Monthly inference volume (images / month)")
    ax.set_ylabel("Payback period (months)")
    ax.set_title(f"Break-even: VLM pay-per-use ({unit:.2f} JPY/image) vs YOLO fixed cost")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--costs", default="conf/costs.yaml")
    parser.add_argument("--out", default="results/breakeven.json")
    parser.add_argument("--figure", default=None, help="指定すると PNG も書き出す")
    args = parser.parse_args()

    costs = yaml.safe_load(Path(args.costs).read_text(encoding="utf-8"))
    res = analyze(costs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(to_markdown(res))
    if args.figure:
        print(f"\n図: {breakeven_figure(res, args.figure)}")
    print(f"保存: {args.out}")


if __name__ == "__main__":
    main()
