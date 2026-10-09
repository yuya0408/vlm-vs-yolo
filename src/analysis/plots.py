"""M4 仕上げ: 精度 × コスト / 精度 × レイテンシ のパレート図。

results/comparison.json を読み、2 パネル(横: コスト / レイテンシ中央値、縦: macro-F1 strict)で
YOLO と VLM を散布。非インタラクティブ環境で動くよう Agg バックエンドを使う。

実行例:
    python -m src.analysis.plots --comparison results/comparison.json \
        --out report/figures/pareto.png
    # README 用の日本語図(パレート: 既定→調整の移動つき / 閾値スイープ: macro-F1 と McNemar p)
    python -m src.analysis.plots --figure pareto_ja
    python -m src.analysis.plots --figure threshold_ja
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


YOLO_COLOR = "#1f77b4"
VLM_COLOR = "#d62728"


def _use_ja_font(plt) -> None:
    # breakeven.py の lang="ja" と同じ扱い(README / Zenn の日本語読者向け)
    plt.rcParams["font.family"] = "Meiryo"
    plt.rcParams["axes.unicode_minus"] = False


def pareto_figure_ja(default_cmp: dict, tuned_cmp: dict, out_path: str) -> str:
    """日本語版パレート図。既定 conf=0.25 の YOLO から tuned YOLO への移動も描く。

    default_cmp = results/comparison.json(既定 0.25)、tuned_cmp = results/comparison_tuned.json。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _use_ja_font(plt)

    d, t = default_cmp["pareto"], tuned_cmp["pareto"]
    p_tuned = tuned_cmp["mcnemar"]["p_value"]
    n = tuned_cmp.get("n_images", "?")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, key, xlabel in ((axes[0], "cost_jpy", f"コスト(円 / {n} 枚)"),
                            (axes[1], "latency_median", "レイテンシ中央値(秒)")):
        x_def, x_tun, x_vlm = d["yolo"][key], t["yolo"][key], t["vlm"][key]
        ax.scatter(x_def, d["yolo"]["macro_f1"], s=110, facecolors="white",
                   edgecolors=YOLO_COLOR, linewidths=2, zorder=3)
        ax.annotate("", xy=(x_tun, t["yolo"]["macro_f1"]), xytext=(x_def, d["yolo"]["macro_f1"]),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5,
                                    shrinkA=7, shrinkB=8), zorder=2)
        ax.scatter(x_tun, t["yolo"]["macro_f1"], s=130, color=YOLO_COLOR, zorder=3)
        ax.scatter(x_vlm, t["vlm"]["macro_f1"], s=130, color=VLM_COLOR, zorder=3)
        ax.annotate(f"YOLO 既定 conf=0.25\n({d['yolo']['macro_f1']:.3f})",
                    (x_def, d["yolo"]["macro_f1"]), textcoords="offset points",
                    xytext=(12, -4), fontsize=9.5, color="#333333", va="top")
        ax.annotate(f"YOLO 調整後 conf=0.075\n({t['yolo']['macro_f1']:.3f})",
                    (x_tun, t["yolo"]["macro_f1"]), textcoords="offset points",
                    xytext=(12, 2), fontsize=9.5, color="#333333", va="bottom")
        ax.annotate(f"VLM\n({t['vlm']['macro_f1']:.3f})", (x_vlm, t["vlm"]["macro_f1"]),
                    textcoords="offset points", xytext=(-12, -4), fontsize=9.5,
                    color="#333333", ha="right", va="top")
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.22, y=0.25)
    axes[0].set_ylabel("macro-F1(strict)")
    axes[0].set_xlim(-30, max(t["vlm"]["cost_jpy"], 1) * 1.25)
    axes[1].text(0.03, 0.95, f"調整後 YOLO vs VLM: McNemar p={p_tuned:.2f}(有意差なし)",
                 transform=axes[1].transAxes, ha="left", va="top", fontsize=9.5, color="#333333")
    fig.suptitle("精度 × コスト / レイテンシ(左上ほど良い。白丸 → 青丸 = YOLO の閾値調整)")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def mcnemar_by_threshold(raw: dict, eval_set: list[dict], vlm_records: list[dict],
                         thresholds: list[float]) -> list[dict]:
    """各しきい値の YOLO と VLM で McNemar(対応あり, 項目単位)。再推論なし・無料。"""
    from .stats import mcnemar_test
    from .yolo_threshold import _records_at_threshold

    out = []
    for t in thresholds:
        mc = mcnemar_test(_records_at_threshold(raw, eval_set, t), vlm_records, eval_set)
        out.append({"threshold": t, **mc})
    return out


def threshold_figure_ja(threshold_res: dict, vlm_summary: dict, mcnemar_curve: list[dict],
                        out_path: str, p_floor: float = 1e-6) -> str:
    """閾値スイープ図(日本語)。上段 macro-F1、下段 McNemar p(対数)の 2 段、x 軸共有。

    threshold_res = results/yolo_threshold.json、vlm_summary = comparison_tuned.json の models.vlm。
    p_floor 未満の p は下端に寄せて描く(0.3 以上の conf では p が極端に小さくなるため)。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _use_ja_font(plt)

    curve = threshold_res["full_curve"]
    ts = [c["threshold"] for c in curve]
    f1 = [c["macro_f1"] for c in curve]
    vlm_f1 = vlm_summary["strict"]["macro_f1"]
    vlm_lo, vlm_hi = vlm_summary["macro_f1_ci95"]
    p_by_t = {m["threshold"]: m["p_value"] for m in mcnemar_curve}
    ps = [max(p_by_t[t], p_floor) for t in ts]
    f1_by_t = dict(zip(ts, f1))
    marks = {0.075: "調整後 0.075", 0.25: "既定 0.25"}
    # ラベル位置(offset points)。曲線と重ならないよう、右下がりの曲線に対し上側/下側に逃がす
    f1_offsets = {0.075: (6, -40), 0.25: (10, 10)}
    p_offsets = {0.075: (8, 4), 0.25: (10, 8)}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})

    for ax in (ax1, ax2):
        ax.axvspan(0.01, 0.10, color="#999999", alpha=0.12, lw=0)
        for t in marks:
            ax.axvline(t, color="#777777", lw=1, ls=":", zorder=1)
        ax.grid(True, alpha=0.3)

    ax1.fill_between([0, 0.8], vlm_lo, vlm_hi, color=VLM_COLOR, alpha=0.10, lw=0)
    ax1.axhline(vlm_f1, color=VLM_COLOR, lw=2)
    ax1.plot(ts, f1, color=YOLO_COLOR, lw=2, marker="o", ms=5)
    ax1.text(0.745, vlm_f1 + 0.004, f"VLM {vlm_f1:.3f}(帯 = 95%CI)", ha="right", va="bottom",
             fontsize=9.5, color="#333333")
    ax1.text(0.745, f1[-1] + 0.012, "YOLO", ha="right", fontsize=9.5, color="#333333")
    box = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9)
    for t, lab in marks.items():
        ax1.annotate(f"{lab}: {f1_by_t[t]:.3f}", (t, f1_by_t[t]), textcoords="offset points",
                     xytext=f1_offsets[t], fontsize=9, color="#333333", bbox=box)
    ax1.text(0.055, 0.712, "妥当域\n0.01〜0.10", ha="center", fontsize=8.5, color="#555555")
    ax1.set_ylabel("macro-F1(strict)")
    ax1.set_ylim(0.70, 0.985)

    ax2.axhline(0.05, color="#333333", lw=1, ls="--")
    ax2.text(0.745, 0.05 * 1.3, "p = 0.05", ha="right", va="bottom", fontsize=9, color="#333333")
    ax2.plot(ts, ps, color="#444444", lw=1.5, marker="o", ms=5)
    for t in marks:
        ax2.annotate(f"p={p_by_t[t]:.3f}" if p_by_t[t] < 0.01 else f"p={p_by_t[t]:.2f}",
                     (t, max(p_by_t[t], p_floor)), textcoords="offset points",
                     xytext=p_offsets[t], fontsize=9, color="#333333", bbox=box)
    t0 = ts[0]
    ax2.annotate(f"conf={t0}: p={p_by_t[t0]:.3f}\n(偽陽性が増え VLM が有意)", (t0, ps[0]),
                 textcoords="offset points", xytext=(10, -30), fontsize=8.5, color="#333333",
                 bbox=box)
    ax2.set_yscale("log")
    ax2.set_ylim(p_floor / 2, 8)
    ax2.set_ylabel(f"McNemar p(項目単位・対数)\n{p_floor:.0e} 未満は下端に表示")
    ax2.set_xlabel("YOLO の信頼度しきい値 conf")
    ax2.set_xlim(0, 0.76)

    fig.suptitle("YOLO の閾値で「有意差」は出たり消えたりする\n"
                 "(既定 0.25 では VLM が有意に勝つが、VLM を見ずに選んだ運用点 0.075 では差がない)",
                 fontsize=11.5)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", choices=["pareto", "pareto_ja", "threshold_ja"],
                        default="pareto")
    parser.add_argument("--comparison", default="results/comparison.json")
    parser.add_argument("--comparison-tuned", default="results/comparison_tuned.json")
    parser.add_argument("--threshold", default="results/yolo_threshold.json")
    parser.add_argument("--yolo-raw", default="results/yolo_raw_detections.json")
    parser.add_argument("--vlm-run", default="results/gemini-gemini-3.5-flash-concise-289d6c0e.json")
    parser.add_argument("--eval-set", default="data/eval_set.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    def load(p: str):
        return json.loads(Path(p).read_text(encoding="utf-8"))

    if args.figure == "pareto":
        path = pareto_figure(load(args.comparison), args.out or "report/figures/pareto.png")
    elif args.figure == "pareto_ja":
        path = pareto_figure_ja(load(args.comparison), load(args.comparison_tuned),
                                args.out or "report/figures/pareto_ja.png")
    else:
        th = load(args.threshold)
        ts = [c["threshold"] for c in th["full_curve"]]
        mc = mcnemar_by_threshold(load(args.yolo_raw), load(args.eval_set),
                                  load(args.vlm_run)["records"], ts)
        path = threshold_figure_ja(th, load(args.comparison_tuned)["models"]["vlm"], mc,
                                   args.out or "report/figures/threshold_sweep_ja.png")
    print(f"図を保存: {path}")


if __name__ == "__main__":
    main()
