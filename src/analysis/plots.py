"""M4 仕上げ: 図の生成。非インタラクティブ環境で動くよう Agg バックエンドを使う。

1. パレート図(pareto_figure): results/comparison.json から 2 パネル
   (横: コスト / レイテンシ中央値、縦: macro-F1 strict)で YOLO と VLM を散布。
2. しきい値スイープ図(threshold_sweep_figure): results/yolo_threshold.json から
   conf に対する YOLO の指標カーブを引き、VLM を水平線として重ねる。
   **本作の主張「精度差はバンド全域で数 pt」を一枚で示す図**(REPORT §1/§2)。
   点推定の勝敗ではなくバンドの重なりを見せるのが目的なので、バンド区間を網掛けする。

実行例:
    python -m src.analysis.plots --comparison results/comparison.json \
        --out report/figures/pareto.png
    # スイープ図も一緒に(VLM の水平線は comparison から自動で読む)
    python -m src.analysis.plots --comparison results/comparison.json \
        --threshold-result results/yolo_threshold.json \
        --sweep-out report/figures/threshold_sweep.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pareto_figure(comparison: dict, out_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = comparison["pareto"]
    labels = ["YOLO", "VLM"]
    f1 = [p["yolo"]["macro_f1"], p["vlm"]["macro_f1"]]
    cost = [p["yolo"]["cost_jpy"], p["vlm"]["cost_jpy"]]
    lat = [p["yolo"]["latency_median"], p["vlm"]["latency_median"]]
    colors = ["#1f77b4", "#d62728"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    n = comparison.get("n_images", "?")

    # ラベルは英語に統一(matplotlib 既定フォントは CJK 非対応で文字化けするため)
    for ax, xvals, xlabel in ((axes[0], cost, f"Cost (JPY / {n} images)"),
                              (axes[1], lat, "Latency median (sec)")):
        for i, lab in enumerate(labels):
            ax.scatter(xvals[i], f1[i], s=160, color=colors[i], zorder=3, label=lab)
            ax.annotate(lab, (xvals[i], f1[i]), textcoords="offset points",
                        xytext=(8, 6), fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("macro-F1 (strict)")
        ax.grid(True, alpha=0.3)
        ax.margins(0.25)

    fig.suptitle("YOLO vs VLM Pareto (top-left = high accuracy, low cost/latency)")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def threshold_sweep_figure(threshold_result: dict, out_path: str,
                           vlm: dict | None = None) -> str:
    """conf スイープ曲線 + VLM 水平線 + バンド網掛け + 操作点。

    threshold_result: yolo_threshold.analyze() の出力(results/yolo_threshold.json)。
    vlm: {"macro_f1": float, "micro_f1": float} 相当。省略すると水平線を描かない。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = threshold_result["full_curve"]
    xs = [c["threshold"] for c in curve]
    band = threshold_result.get("band") or {}
    op = (threshold_result.get("operating_point") or {}).get("threshold")

    # ラベルは英語に統一(matplotlib 既定フォントは CJK 非対応で文字化けするため)
    series = [
        ("macro_f1", "YOLO macro-F1", "#1f77b4", "o"),
        ("micro_f1", "YOLO micro-F1", "#2ca02c", "s"),
        ("item_accuracy", "YOLO item accuracy", "#9467bd", "^"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    if band.get("low") is not None:
        ax.axvspan(band["low"], band["high"], color="#cccccc", alpha=0.35, zorder=0,
                   label=f"plausible band ({band['low']}–{band['high']})")

    for key, label, color, marker in series:
        ys = [c.get(key) for c in curve]
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=marker,
                    color=color, label=label, zorder=3, markersize=5)

    if vlm:
        for key, color, style in (("macro_f1", "#d62728", "--"),
                                  ("micro_f1", "#ff7f0e", ":")):
            if vlm.get(key) is not None:
                ax.axhline(vlm[key], color=color, linestyle=style, zorder=2,
                           label=f"VLM {key.replace('_', '-')} = {vlm[key]}")

    if op is not None:
        ax.axvline(op, color="#333333", linestyle="-.", linewidth=1.2, zorder=2,
                   label=f"operating point (conf={op})")
    ax.axvline(0.25, color="#888888", linestyle="-", linewidth=1.0, alpha=0.7, zorder=1,
               label="library default (conf=0.25)")

    ax.set_xscale("log")
    ax.set_xlabel("YOLO confidence threshold (log scale)")
    ax.set_ylabel("score (strict)")
    ax.set_title("Tuning the baseline closes the gap:\n"
                 "YOLO metrics vs confidence threshold, with VLM as horizontal reference")
    ax.grid(True, alpha=0.3, which="both")
    # 実データでは全曲線が 0.9 付近に密集するため、凡例は軸の外(下)に逃がす。
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
              frameon=False)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", default="results/comparison.json")
    parser.add_argument("--out", default="report/figures/pareto.png")
    parser.add_argument("--threshold-result",
                        help="results/yolo_threshold.json。指定するとスイープ図も出す")
    parser.add_argument("--sweep-out", default="report/figures/threshold_sweep.png")
    parser.add_argument("--no-pareto", action="store_true",
                        help="パレート図を出さない(スイープ図だけ欲しいとき)")
    args = parser.parse_args()

    # comparison は パレート図の入力であると同時に、スイープ図に重ねる VLM 水平線の出所でもある。
    # 無ければスイープ図は水平線なしで描く(VLM ラン前の M2 段階でも図が出せる)。
    comparison = None
    if Path(args.comparison).exists():
        comparison = json.loads(Path(args.comparison).read_text(encoding="utf-8"))

    if not args.no_pareto:
        if comparison is None:
            raise SystemExit(f"{args.comparison} が無い(--no-pareto でスキップできる)")
        print(f"パレート図を保存: {pareto_figure(comparison, args.out)}")

    if args.threshold_result:
        tr = json.loads(Path(args.threshold_result).read_text(encoding="utf-8"))
        # VLM の水平線は comparison の strict 値をそのまま使う(別途入力しない = 転記ミス防止)
        vlm = comparison["models"]["vlm"]["strict"] if comparison else None
        print(f"スイープ図を保存: {threshold_sweep_figure(tr, args.sweep_out, vlm)}")


if __name__ == "__main__":
    main()
