"""README の閾値スイープ図と、図の共通スタイル。

上段は YOLO と VLM の項目単位の正解率、下段は VLM−YOLO の画像単位ペア差の 95%CI(主な推論)。
点ごとの判定(3 指標のどれかの CI が 0 を含まなければ「VLM が有意に上」)を上段の印の塗りで示す。
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
DIFF_METRICS = (("macro_f1", "macro-F1", "#7a5195"), ("micro_f1", "micro-F1", "#ef5675"),
                ("item_accuracy", "item accuracy", "#ffa600"))
DIFF_YLIM = (-2.0, 9.0)  # pt。上端を超える CI は▲で上端に置く


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
    """各しきい値の YOLO と VLM で McNemar(対応あり, 項目単位)。再推論なし・無料。補助。"""
    from .stats import mcnemar_test
    from .yolo_threshold import _records_at_threshold

    out = []
    for t in thresholds:
        mc = mcnemar_test(_records_at_threshold(raw, eval_set, t), vlm_records, eval_set)
        out.append({"threshold": t, **mc})
    return out


def paired_diff_by_threshold(raw: dict, eval_set: list[dict], vlm_records: list[dict],
                             thresholds: list[float], n_boot: int = 10_000,
                             seed: int = 42) -> list[dict]:
    """各しきい値で VLM−YOLO の画像単位ペア差ブートストラップ CI(主な推論)。

    significant は 3 指標のどれかの 95%CI が 0 を含まないこと(点ごと、多重比較の補正なし)。
    """
    from .stats import paired_bootstrap_diff
    from .yolo_threshold import _records_at_threshold

    out = []
    for t in thresholds:
        d = paired_bootstrap_diff(_records_at_threshold(raw, eval_set, t), vlm_records, eval_set,
                                  n_boot=n_boot, seed=seed)
        row = {"threshold": t}
        for m, _, _ in DIFF_METRICS:
            row[m] = {"point_diff": d[m]["point_diff"], "ci95": list(d[m]["ci95"])}
        row["significant"] = any(lo > 0 or hi < 0
                                 for lo, hi in (row[m]["ci95"] for m, _, _ in DIFF_METRICS))
        out.append(row)
    return out


def _verdict_points(ax, xs, ys, significant, color) -> None:
    """塗り=VLM が有意に上、白抜き=差を検出できなかった。"""
    for x, y, s in zip(xs, ys, significant):
        ax.plot([x], [y], marker="o", ms=6.5, mfc=color if s else SURFACE, mec=color, mew=1.5,
                ls="none", zorder=4)


def threshold_figure_ja(threshold_res: dict, vlm_strict: dict, diff_curve: list[dict],
                        out_path: str) -> str:
    """閾値スイープ図。threshold_res = results/yolo_threshold.json、
    vlm_strict = comparison_tuned.json の models.vlm.strict、diff_curve = paired_diff_by_threshold。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mt
    use_figure_style(plt)

    curve = threshold_res["full_curve"]
    ts = [c["threshold"] for c in curve]
    acc = [c["item_accuracy"] for c in curve]
    diff = {d["threshold"]: d for d in diff_curve}
    sig = [diff[t]["significant"] for t in ts]
    ns = [t for t, s_ in zip(ts, sig) if not s_]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.2, 6.9), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2.6]})
    for ax in (a1, a2):
        ax.set_xscale("log")
        for t in (0.075, 0.25):
            ax.axvline(t, color=INK_2, lw=0.9, ls=":", zorder=1)
        if ns:
            ax.axvspan(min(ns) * 0.93, max(ns) * 1.07, color=SHADE, zorder=0, lw=0)

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
    a1.set_title("項目単位の正解率 上ほど良い", loc="left", fontsize=10.5, color=INK)
    handles = [plt.Line2D([], [], marker="o", ms=6.5, color=INK_2, mfc=INK_2, ls="none"),
               plt.Line2D([], [], marker="o", ms=6.5, color=INK_2, mfc=SURFACE, mew=1.5, ls="none")]
    a1.legend(handles, ["VLMが有意に上(3指標のどれかのCIが0を含まない)", "差を検出できなかった(網掛け)"],
              loc="lower left", fontsize=9, handletextpad=0.3, borderaxespad=0.2)

    a2.axhline(0, color=INK, lw=1.1, ls="--", zorder=2)
    top = DIFF_YLIM[1]
    for k, (m, label, color) in enumerate(DIFF_METRICS):
        for t in ts:
            x = t * (1 + 0.06 * (k - 1))  # 指標ごとに少しずらす
            lo, hi = (v * 100 for v in diff[t][m]["ci95"])
            pt = diff[t][m]["point_diff"] * 100
            a2.plot([x, x], [lo, min(hi, top)], color=color, lw=1.6, zorder=3)
            if pt <= top:
                a2.plot([x], [pt], marker="o", ms=3.5, color=color, zorder=4)
            if hi > top:
                a2.plot([x], [top], marker="^", ms=5, color=color, zorder=4)
        a2.plot([], [], color=color, lw=1.6, marker="o", ms=3.5, label=label)
    a2.legend(loc="upper left", fontsize=9, ncol=3, handletextpad=0.3, borderaxespad=0.2)
    a2.set_ylim(*DIFF_YLIM)
    a2.set_ylabel("VLM−YOLO(pt)")
    a2.set_title("VLM−YOLOの差と画像単位ペア差の95%CI(主な推論) 0の線より上はVLMが上",
                 loc="left", fontsize=10.5, color=INK)

    a2.set_xlim(0.009, 0.82)
    ticks = [0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 0.75]
    a2.set_xticks(ticks, [f"{t:g}" for t in ticks])
    a2.xaxis.set_minor_locator(mt.NullLocator())
    a2.set_xlabel("YOLO の信頼度しきい値 conf(対数)")

    rng = f"{min(ns):g}〜{max(ns):g}" if ns else "なし"
    set_figure_title(fig, f"閾値の公平化:既定0.25ではVLMが有意に上、グリッドの{rng}では差を検出できなかった",
                     ["0.075はVLMのスコアを使わない式で選んだが、見直したきっかけは既定値での結果。▲はCIの上端が図の外。",
                      "判定は点ごとのペア差CI(補正なし)。同等を示したのではない(許す差を決めていない)。"])
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
    parser.add_argument("--diff-out", default="results/threshold_paired_diff.json")
    args = parser.parse_args()

    def load(p: str):
        return json.loads(Path(p).read_text(encoding="utf-8"))

    th = load(args.threshold)
    ts = [c["threshold"] for c in th["full_curve"]]
    diff = paired_diff_by_threshold(load(args.yolo_raw), load(args.eval_set),
                                    load(args.vlm_run)["records"], ts)
    Path(args.diff_out).write_text(json.dumps(diff, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
    path = threshold_figure_ja(th, load(args.comparison_tuned)["models"]["vlm"]["strict"], diff,
                               args.out)
    print(f"図を保存: {path}")


if __name__ == "__main__":
    main()
