"""plots(閾値スイープ図)の単体テスト。ファイルが生成されることだけ確認する。"""

from src.analysis.plots import threshold_figure_ja


def test_threshold_figure_writes_png(tmp_path):
    curve = [{"threshold": t, "item_accuracy": a}
             for t, a in ((0.01, 0.96), (0.075, 0.97), (0.25, 0.96), (0.5, 0.93))]
    def row(t, lo, sig):
        ci = {"point_diff": lo + 0.01, "ci95": [lo, lo + 0.02]}
        return {"threshold": t, "macro_f1": ci, "micro_f1": ci, "item_accuracy": ci,
                "significant": sig}

    diff = [row(0.01, 0.005, True), row(0.075, -0.005, False), row(0.25, 0.01, True),
            row(0.5, 0.2, True)]
    out = tmp_path / "threshold.png"
    path = threshold_figure_ja({"full_curve": curve}, {"item_accuracy": 0.974}, diff, str(out))
    assert out.exists() and out.stat().st_size > 0
    assert path == str(out)
