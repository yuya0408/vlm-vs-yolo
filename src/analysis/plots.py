"""README の閾値スイープ図と、図の共通スタイル。

上段は McNemar と同じ量(項目単位の正解率)、下段は McNemar の p。点ごとの判定を印の塗りで示す。
非インタラクティブ環境で動くよう Agg バックエンドを使う。

実行例:
    python -m src.analysis.plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

YOLO_COLOR = "#1f77b4"
VLM_COLOR = "#d62728"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
MUTED = "#b9b8b2"
SHADE = "#ecebe7"
ALPHA = 0.05
P_FLOOR = 1e-4  # これ未満は▼で下端に置く


def use_figure_style(plt) -> None:
    plt.rcParams.update({
        "font.family": ["Meiryo", "DejaVu Sans"], "font.size": 10, "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
        "savefig.facecolor": SURFACE,
    })


def set_figure_title(fig, text: str, sub: list[str]) -> None:
    h = fig.get_figheight()
    fig.suptitle(text, x=0.02, y=1 - 0.12 / h, ha="left", va="top", fontsize=12.5,
                 color=INK, fontweight="bold")
    for i, line in enumerate(sub):
        fig.text(0.02, 1 - (0.46 + 0.2 * i) / h, line, ha="left", va="top", fontsize=9, color=INK_2)


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


def _verdict_points(ax, xs, ys, significant, color) -> None:
    """塗り=VLM が有意に上、白抜き=差を検出できなかった。"""
    for x, y, s in zip(xs, ys, significant):
        ax.plot([x], [y], marker="o", ms=6.5, mfc=color if s else SURFACE, mec=color, mew=1.5,
                ls="none", zorder=4)


def threshold_figure_ja(threshold_res: dict, vlm_strict: dict, mcnemar_curve: list[dict],
                        out_path: str) -> str:
    """閾値スイープ図。threshold_res = results/yolo_threshold.json、
    vlm_strict = comparison_tuned.json の models.vlm.strict。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mt
    use_figure_style(plt)

    curve = threshold_res["full_curve"]
    ts = [c["threshold"] for c in curve]
    acc = [c["item_accuracy"] for c in curve]
    p = {m["threshold"]: m["p_value"] for m in mcnemar_curve}
    sig = [p[t] < ALPHA for t in ts]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.2, 6.9), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2.6]})
    for ax in (a1, a2):
        ax.set_xscale("log")
        for t in (0.075, 0.25):
            ax.axvline(t, color=INK_2, lw=0.9, ls=":", zorder=1)

    vlm_acc = vlm_strict["item_accuracy"]
    a1.axhline(vlm_acc, color=VLM_COLOR, lw=2, zorder=2)
    a1.plot(ts, acc, color=YOLO_COLOR, lw=2, zorder=3)
    _verdict_points(a1, ts, acc, sig, YOLO_COLOR)
    a1.text(0.75, vlm_acc + 0.0015, f"VLM {vlm_acc:.3f}", ha="right", va="bottom", fontsize=9.5,
            color=VLM_COLOR)
    a1.text(0.60, 0.9035, "YOLO", ha="right", fontsize=9.5, color=YOLO_COLOR)
    acc_by = dict(zip(ts, acc))
    for t, lab in ((0.075, "conf=0.075(選んだ値)"), (0.25, "conf=0.25(既定)")):
        if t in acc_by:
            a1.annotate(f"{lab} {acc_by[t]:.3f}", (t, acc_by[t]), textcoords="offset points",
                        xytext=(-5, -20), fontsize=9, color=INK, ha="right")
    a1.set_ylim(0.88, 0.985)
    a1.set_title("項目単位の正解率(McNemarと同じ量) 上ほど良い", loc="left", fontsize=10.5, color=INK)
    handles = [plt.Line2D([], [], marker="o", ms=6.5, color=INK_2, mfc=INK_2, ls="none"),
               plt.Line2D([], [], marker="o", ms=6.5, color=INK_2, mfc=SURFACE, mew=1.5, ls="none"),
               plt.Line2D([], [], marker="v", ms=6.5, color=INK_2, ls="none")]
    a1.legend(handles, ["VLMが有意に上(p<0.05)", "差を検出できなかった", "p<0.0001(下端に置いた)"],
              loc="lower left", fontsize=9, handletextpad=0.3, borderaxespad=0.2)

    a2.axhspan(P_FLOOR / 2, ALPHA, color=SHADE, zorder=0, lw=0)
    a2.axhline(ALPHA, color=INK, lw=1.1, ls="--", zorder=2)
    a2.text(0.75, ALPHA * 1.25, "p = 0.05", ha="right", va="bottom", fontsize=9, color=INK)
    a2.text(0.75, ALPHA * 0.72, "網掛け:VLMが有意に上", ha="right", va="top", fontsize=9, color=INK_2)
    ps = [max(p[t], P_FLOOR) for t in ts]
    a2.plot(ts, ps, color=INK_2, lw=1.5, zorder=3)
    for t, y in zip(ts, ps):
        if p[t] < P_FLOOR:
            a2.plot([t], [y], marker="v", ms=6.5, color=INK_2, ls="none", zorder=4)
    keep = [(t, y, s) for t, y, s in zip(ts, ps, sig) if p[t] >= P_FLOOR]
    _verdict_points(a2, [k[0] for k in keep], [k[1] for k in keep], [k[2] for k in keep], INK_2)
    # ラベル位置(offset points)。線と重ならないよう点ごとに逃がす
    labels = {0.01: ((9, -5), "left"), 0.02: ((6, -13), "left"), 0.075: ((6, 5), "left"),
              0.25: ((8, 3), "left"), 0.15: ((6, 5), "left"), 0.2: ((-6, -14), "right")}
    for t, (off, ha) in labels.items():
        if t in p:
            text = f"p={p[t]:.3f}" if p[t] < 0.1 else f"p={p[t]:.2f}"
            a2.annotate(text, (t, max(p[t], P_FLOOR)), textcoords="offset points", xytext=off,
                        ha=ha, fontsize=9, color=INK)
    a2.text(0.75, P_FLOOR * 2.0, "conf≥0.3 はすべて p<0.0001", ha="right", fontsize=9, color=INK_2)
    a2.set_yscale("log")
    a2.set_ylim(P_FLOOR / 2, 3)
    a2.set_yticks([1e-4, 1e-3, 1e-2, 0.1, 1], ["0.0001", "0.001", "0.01", "0.1", "1"])
    a2.yaxis.set_minor_locator(mt.NullLocator())
    a2.set_title("McNemar の p(対数) 下ほど差がはっきりしている", loc="left", fontsize=10.5, color=INK)

    a2.set_xlim(0.009, 0.82)
    ticks = [0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 0.75]
    a2.set_xticks(ticks, [f"{t:g}" for t in ticks])
    a2.xaxis.set_minor_locator(mt.NullLocator())
    a2.set_xlabel("YOLO の信頼度しきい値 conf(対数)")

    set_figure_title(fig, "閾値の公平化:既定0.25ではVLMが有意に上、グリッドの0.025〜0.15では差を検出できなかった",
                     ["0.01・0.02では再び有意。0.075はVLMのスコアを使わない式で選んだが、見直したきっかけは既定値での結果。",
                      "判定は点ごとのMcNemar(補正なし)。同等を示したのではない(許す差を決めていない)。"])
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.74 / fig.get_figheight()), h_pad=1.2)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-tuned", default="results/comparison_tuned.json")
    parser.add_argument("--threshold", default="results/yolo_threshold.json")
    parser.add_argument("--yolo-raw", default="results/yolo_raw_detections.json")
    parser.add_argument("--vlm-run", default="results/gemini-gemini-3.5-flash-concise-289d6c0e.json")
    parser.add_argument("--eval-set", default="data/eval_set.json")
    parser.add_argument("--out", default="report/figures/threshold_sweep_ja.png")
    args = parser.parse_args()

    def load(p: str):
        return json.loads(Path(p).read_text(encoding="utf-8"))

    th = load(args.threshold)
    ts = [c["threshold"] for c in th["full_curve"]]
    mc = mcnemar_by_threshold(load(args.yolo_raw), load(args.eval_set),
                              load(args.vlm_run)["records"], ts)
    path = threshold_figure_ja(th, load(args.comparison_tuned)["models"]["vlm"]["strict"], mc, args.out)
    print(f"図を保存: {path}")


if __name__ == "__main__":
    main()
