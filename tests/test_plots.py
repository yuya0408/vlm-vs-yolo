"""plots(パレート図)の単体テスト。ファイルが生成されることだけ確認する。"""

from src.analysis.plots import pareto_figure


def test_pareto_figure_writes_png(tmp_path):
    comparison = {
        "n_images": 300,
        "pareto": {
            "yolo": {"macro_f1": 0.906, "cost_jpy": 0.0, "latency_median": 0.33},
            "vlm": {"macro_f1": 0.961, "cost_jpy": 251.0, "latency_median": 5.42},
        },
    }
    out = tmp_path / "pareto.png"
    path = pareto_figure(comparison, str(out))
    assert out.exists() and out.stat().st_size > 0
    assert path == str(out)
