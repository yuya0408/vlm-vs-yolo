"""plots(図の生成)の単体テスト。ファイルが生成されることだけ確認する。"""

from src.analysis.plots import pareto_figure, threshold_sweep_figure


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


def _threshold_result():
    return {
        "full_curve": [
            {"threshold": 0.01, "macro_f1": 0.945, "micro_f1": 0.941,
             "item_accuracy": 0.962, "count_recall": 0.985},
            {"threshold": 0.075, "macro_f1": 0.929, "micro_f1": 0.954,
             "item_accuracy": 0.972, "count_recall": 0.927},
            {"threshold": 0.25, "macro_f1": 0.906, "micro_f1": 0.934,
             "item_accuracy": 0.961, "count_recall": 0.767},
        ],
        "band": {"low": 0.01, "high": 0.10, "n_points": 2,
                 "macro_f1": [0.929, 0.945], "micro_f1": [0.941, 0.954],
                 "item_accuracy": [0.962, 0.972], "count_recall": [0.927, 0.985]},
        "operating_point": {"threshold": 0.075, "converged": True},
    }


def test_threshold_sweep_figure_writes_png(tmp_path):
    out = tmp_path / "sweep.png"
    path = threshold_sweep_figure(_threshold_result(), str(out),
                                  vlm={"macro_f1": 0.961, "micro_f1": 0.959})
    assert out.exists() and out.stat().st_size > 0
    assert path == str(out)


def test_threshold_sweep_figure_without_vlm_reference(tmp_path):
    """VLM ランがまだ無い段階(M2)でも図が引けること。"""
    out = tmp_path / "sweep_no_vlm.png"
    threshold_sweep_figure(_threshold_result(), str(out), vlm=None)
    assert out.exists() and out.stat().st_size > 0


def test_threshold_sweep_figure_tolerates_missing_count_recall(tmp_path):
    """gt_counts の無い評価セットでは count_recall が None。図はそれでも引ける。"""
    tr = _threshold_result()
    for c in tr["full_curve"]:
        c["count_recall"] = None
    tr["band"]["count_recall"] = None
    out = tmp_path / "sweep_no_cr.png"
    threshold_sweep_figure(tr, str(out), vlm={"macro_f1": 0.961, "micro_f1": 0.959})
    assert out.exists() and out.stat().st_size > 0
