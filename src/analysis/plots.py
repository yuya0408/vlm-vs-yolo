"""M4 仕上げ: 精度 × コスト / 精度 × レイテンシ のパレート図。

results/comparison.json を読み、2 パネル(横: コスト / レイテンシ中央値、縦: macro-F1 strict)で
YOLO と VLM を散布。非インタラクティブ環境で動くよう Agg バックエンドを使う。

実行例:
    python -m src.analysis.plots --comparison results/comparison.json \
        --out report/figures/pareto.png
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", default="results/comparison.json")
    parser.add_argument("--out", default="report/figures/pareto.png")
    args = parser.parse_args()

    comparison = json.loads(Path(args.comparison).read_text(encoding="utf-8"))
    path = pareto_figure(comparison, args.out)
    print(f"パレート図を保存: {path}")


if __name__ == "__main__":
    main()
