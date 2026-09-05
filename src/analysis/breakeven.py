"""VLM(従量課金)と YOLO(固定費先行)の損益分岐分析。

REPORT のコスト比較は「300 枚あたり 251 円 vs 0 円」で、YOLO 側が無料に見える。しかし YOLO の
0 円は API 課金が 0 という意味でしかなく、実際には**アノテーション・学習工数・推論基盤の運用**が
固定費として先に立つ。実務の技術選定で効くのはこの固定費と従量課金の交点である。

    YOLO 累計(T ヶ月) = C_fix + ops_monthly × T
    VLM  累計(T ヶ月) = unit_jpy × 月間枚数 M × T
    C_fix = ラベル枚数 × 1枚あたり矩形数 × 矩形単価 × 手戻り係数 + 学習工数 × 時間単価

固定費は環境依存で測れないため、点推定としては読まない。`conf/costs.yaml` には**最も安い条件
(= YOLO に最も有利な下限)**を出典付きで置き、そこから固定費を n 倍に振った感度を併せて出す。
下限で結論が出れば、実際の条件はそれより高くなる方向にしか動かないので結論は補強される
(a fortiori)。しきい値を点で決めず band で語った REPORT §2 と同じ扱い。

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
                 "(上振れは感度表で確認)。精度互角(第一象限)が前提。"),
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
        f"> 「回収不能」= 交点が存在しない(VLM の月額 ≤ YOLO の運用月額)、または回収に "
        f"{res['horizon_cap_months']} ヶ月超。",
        f"> {res['note']}",
    ]
    return "\n".join(lines)


_MULT_LABEL_JA = {"x1": "下限(×1)", "x3": "×3", "x10": "×10"}


def breakeven_figure(res: dict, out_path: str, lang: str = "en") -> str:
    """月間推論枚数(対数)× 回収期間(月)。回収不能域は上端に貼り付けて示す。

    lang="en"(既定): matplotlib 既定フォントは CJK 非対応で文字化けするため英語ラベル。
    lang="ja": Zenn 記事など日本語読者向けの埋め込み用。CJK 対応フォントに切り替える。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if lang == "ja":
        plt.rcParams["font.family"] = "Meiryo"
        plt.rcParams["axes.unicode_minus"] = False

    unit = res["vlm_unit_jpy_per_image"]
    cap = res["horizon_cap_months"]
    volumes = np.logspace(2, 6, 200)  # 100 〜 1,000,000 枚/月

    fig, ax = plt.subplots(figsize=(8, 5))
    styles = ["-", "--", ":", "-."]
    for name, s in res["scenarios"].items():
        ops = s["ops_monthly_jpy"]
        for i, (key, sv) in enumerate(s["sensitivity_by_fixed_cost"].items()):
            fixed = sv["fixed_total_jpy"]
            saving = unit * volumes - ops
            months = np.where(saving > 0, fixed / np.where(saving > 0, saving, 1), np.inf)
            if lang == "ja":
                label = (f"{_MULT_LABEL_JA.get(key, key)}: "
                         f"固定費 {fixed/1e4:.0f}万円, 運用 {ops/1e4:.0f}万円/月")
            else:
                label = f"{name} {key}: fixed {fixed/1e6:.2f}M JPY, ops {ops/1e3:.0f}k JPY/mo"
            ax.plot(volumes, np.clip(months, None, cap * 1.4), lw=2,
                    ls=styles[i % len(styles)], label=label)

    ax.axhline(cap, color="gray", ls="--", lw=1)
    cap_text = f"回収不能({cap} ヶ月超)" if lang == "ja" else f"never pays back (> {cap} months)"
    ax.text(1.2e2, cap * 1.05, cap_text, fontsize=9, color="gray")
    ax.axvspan(100, 1000, color="#d62728", alpha=0.08)
    poc_text = "PoC 規模" if lang == "ja" else "PoC scale"
    ax.text(1.1e2, 1.6, poc_text, fontsize=9, color="#d62728")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1, cap * 1.6)
    if lang == "ja":
        ax.set_xlabel("月間推論枚数(枚/月)")
        ax.set_ylabel("回収期間(ヶ月)")
        ax.set_title(f"損益分岐: VLM 従量課金({unit:.2f} 円/枚) vs YOLO 固定費\n"
                     "(下限 = 最安のアノテーション仮定; ×3/×10 = 固定費の上振れ感度)", fontsize=11)
    else:
        ax.set_xlabel("Monthly inference volume (images / month)")
        ax.set_ylabel("Payback period (months)")
        ax.set_title(f"Break-even: VLM pay-per-use ({unit:.2f} JPY/image) vs YOLO fixed cost\n(lean = cheapest annotation assumption; x3 / x10 = cost overrun)", fontsize=11)
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
    parser.add_argument("--lang", default="en", choices=["en", "ja"],
                         help="図のラベル言語(既定 en。Zenn 記事埋め込み用途では ja)")
    args = parser.parse_args()

    costs = yaml.safe_load(Path(args.costs).read_text(encoding="utf-8"))
    res = analyze(costs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(to_markdown(res))
    if args.figure:
        print(f"\n図: {breakeven_figure(res, args.figure, lang=args.lang)}")
    print(f"保存: {args.out}")


if __name__ == "__main__":
    main()
