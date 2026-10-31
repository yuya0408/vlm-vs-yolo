"""VLM(従量課金)と YOLO(固定費先行)の損益分岐分析。

REPORT のコスト比較は「300 枚あたり 251 円 vs 0 円」で、YOLO 側が無料に見える。しかし YOLO の
0 円は API 課金が 0 という意味でしかなく、実際には**アノテーション・学習工数・推論基盤の運用**が
固定費として先に立つ。技術選定で効くのはこの固定費と従量課金の交点である。

    YOLO 累計(T ヶ月) = C_fix + ops_monthly × T
    VLM  累計(T ヶ月) = unit_jpy × 月間枚数 M × T
    C_fix = ラベル枚数 × 1枚あたり矩形数 × 矩形単価 × 手戻り係数 + 学習工数 × 時間単価

固定費は環境依存で測れないため、点推定としては読まない。`conf/costs.yaml` には**公開値で最も
安い単価と少ない枚数(= YOLO に有利な仮定)**を置き、固定費を上下に振った感度を併せて出す。
学習工数などは出典のない仮定で、真の下限ではない。自分の値に差し替えて読むこと。

前提: 精度の差を検出できなかったこと(REPORT §1)。語彙外・ラベル無しの第二象限では YOLO は
いくら払っても要件を満たさないため、そもそも交点が存在しない(§7)。本分析は第一象限限定の道具。

実行例:
    python -m src.analysis.breakeven --costs conf/costs.yaml \
        --out results/breakeven.json --figure report/figures/breakeven_ja.png
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

    multipliers = [float(m) for m in costs.get("fixed_cost_multipliers", [1])]

    scenarios = {}
    for name, sc in costs["scenarios"].items():
        fc = fixed_cost(sc)
        ops = float(sc.get("ops_monthly_jpy", 0))
        # 上振れ側の感度: 固定費が n 倍になっても結論の向きが変わらないかを見る
        sensitivity = {
            f"x{m:g}": {
                "fixed_total_jpy": round(fc["total_jpy"] * m),
                "break_even_images": round(break_even_images(fc["total_jpy"] * m, unit)),
                "payback_months": {
                    str(int(v)): payback_months(fc["total_jpy"] * m, ops, unit, v)
                    for v in volumes
                },
            }
            for m in multipliers
        }
        scenarios[name] = {
            "sensitivity_by_fixed_cost": sensitivity,
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
        "fixed_cost_multipliers": multipliers,
        "horizon_cap_months": _HORIZON_CAP_MONTHS,
        "scenarios": scenarios,
        "note": ("固定費は公開値から置いた仮定であり実測ではない。基本シナリオは"
                 "「YOLO に最も有利な最安条件」で、実際はこれより高くなる方向にしか動かない"
                 "(上振れは感度表で確認)。精度の差を検出できない第一象限が前提。"),
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

    lines += ["", "### 感度: 固定費が n 倍に膨らんだ場合", "",
              "| シナリオ | 固定費 | 損益分岐の累計枚数 | "
              + " | ".join(f"{int(v):,} 枚/月" for v in res["monthly_volumes"]) + " |",
              "|---" * (len(res["monthly_volumes"]) + 3) + "|"]
    for name, s_ in res["scenarios"].items():
        for key, sv in s_["sensitivity_by_fixed_cost"].items():
            cells = " | ".join(
                ("回収不能" if sv["payback_months"][str(int(v))] is None
                 else f"{sv['payback_months'][str(int(v))]:.1f} ヶ月")
                for v in res["monthly_volumes"])
            lines.append(f"| `{name}` {key} | {sv['fixed_total_jpy']:,} 円 | "
                         f"{sv['break_even_images']:,} 枚 | {cells} |")

    lines += [
        "",
        f"> 「回収不能」= 交点が存在しない(VLM の月額が YOLO の運用月額以下)、または回収に "
        f"{res['horizon_cap_months']} ヶ月超。",
        f"> {res['note']}",
    ]
    return "\n".join(lines)


def breakeven_figure(res: dict, out_path: str) -> str:
    """月間推論枚数 × 回収期間(両対数)。120 ヶ月超は線を消し、網掛けで回収不能を示す。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mt
    import numpy as np

    from .plots import INK, INK_2, MUTED, SHADE, set_figure_title, use_figure_style
    use_figure_style(plt)

    scenarios = res["scenarios"]
    s = scenarios.get("lean") or next(iter(scenarios.values()))
    unit, cap = res["vlm_unit_jpy_per_image"], res["horizon_cap_months"]
    vols = np.logspace(3, 6, 200)
    fig, ax = plt.subplots(figsize=(8.6, 4.9))

    def plot_line(fixed, color, lw, ls):
        months = fixed / (unit * vols)
        ax.plot(vols, np.where(months > cap, np.nan, months), color=color, lw=lw, ls=ls, zorder=2)

    if "zero_annotation" in scenarios:
        plot_line(scenarios["zero_annotation"]["fixed_cost"]["total_jpy"], MUTED, 1.7, "-.")
        ax.text(1.45e5, 1.3, "アノテーション0円 24万円", fontsize=9, color=INK_2, ha="right")
    ax.axhspan(cap, 220, color=SHADE, lw=0, zorder=0)
    ax.axhline(cap, color=INK_2, lw=1, ls="--", zorder=1)
    ax.text(320, cap * 1.12, "回収不能(120ヶ月超)", fontsize=9, color=INK_2, va="bottom")
    look = {"x1": (INK, 2.4, "-", 3), "x3": (MUTED, 1.7, "--", 8), "x10": (MUTED, 1.7, ":", 20)}
    for key, sv in s["sensitivity_by_fixed_cost"].items():
        if key not in look:
            continue
        fixed = sv["fixed_total_jpy"]
        color, lw, ls, label_y = look[key]
        plot_line(fixed, color, lw, ls)
        name = f"{fixed / 1e4:.0f}万円" if key == "x1" else f"×{key[1:]} {fixed / 1e4:.0f}万円"
        ax.text(fixed / (unit * label_y) * 1.1, label_y, name, fontsize=9.5, va="bottom",
                color=INK if key == "x1" else INK_2, fontweight="bold" if key == "x1" else "normal")
    ax.axvline(500, color=INK_2, lw=0.9, ls=":", zorder=1)
    ax.text(530, 60, "月500枚:4本とも回収不能", fontsize=9, color=INK_2)
    v36 = s["break_even_monthly_volume"].get("36")
    if v36:
        ax.plot([v36], [36], marker="o", ms=8, color=INK, zorder=4)
        ax.annotate(f"36ヶ月で回収:月{v36:,}枚", (v36, 36), textcoords="offset points",
                    xytext=(-16, -32), fontsize=9.5, color=INK, ha="right", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8, shrinkB=5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(300, 1e6)
    ax.set_ylim(1, 220)
    ax.set_yticks([1, 3, 12, 36, 120], ["1", "3", "12", "36", "120"])
    ax.set_xticks([500, 1e3, 1e4, 1e5, 1e6], ["500", "1,000", "10,000", "100,000", "1,000,000"])
    ax.xaxis.set_minor_locator(mt.NullLocator())
    ax.yaxis.set_minor_locator(mt.NullLocator())
    ax.set_xlabel("月間推論枚数(枚/月、対数)")
    ax.set_ylabel("回収期間(ヶ月、対数)")
    ax.set_title("下ほど早く回収できる", loc="left", fontsize=10.5, color=INK)
    set_figure_title(fig, "損益分岐(参考。判定には使わない):YOLOに有利な仮定でも、回収には月1万枚のオーダーが要る",
                     ["回収期間=固定費÷(VLMの単価×月間枚数)。単価0.84円/枚は実測、固定費は仮定(YOLOに有利に置いた。うち24万円は出典のない学習工数)。",
                      "運用月額0円の場合。4本は別の手法ではなく、同じ計算で固定費だけを振った感度。"])
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.78 / fig.get_figheight()))
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
