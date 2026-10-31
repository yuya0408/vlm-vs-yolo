"""plots(閾値スイープ図)の単体テスト。ファイルが生成されることだけ確認する。"""

from src.analysis.plots import threshold_figure_ja


def test_threshold_figure_writes_png(tmp_path):
    curve = [{"threshold": t, "item_accuracy": a}
             for t, a in ((0.01, 0.96), (0.075, 0.97), (0.25, 0.96), (0.5, 0.93))]
    mc = [{"threshold": t, "p_value": p}
          for t, p in ((0.01, 0.003), (0.075, 0.71), (0.25, 0.004), (0.5, 1e-9))]
    out = tmp_path / "threshold.png"
    path = threshold_figure_ja({"full_curve": curve}, {"item_accuracy": 0.974}, mc, str(out))
    assert out.exists() and out.stat().st_size > 0
    assert path == str(out)
