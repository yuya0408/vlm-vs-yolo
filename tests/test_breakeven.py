"""breakeven(損益分岐の後処理)の単体テスト。外部依存なし。"""

import pytest

from src.analysis.breakeven import (
    analyze, break_even_images, break_even_monthly_volume, breakeven_figure,
    fixed_cost, payback_months, to_markdown, vlm_unit_cost_jpy,
)


def _costs() -> dict:
    return {
        "vlm": {"measured_cost_jpy": 251, "measured_n_images": 300},
        "scenarios": {
            # 固定費 = 100*2*10*1.5 + 10*1000 = 3000 + 10000 = 13000 円
            "cheap": {"label": "テスト", "n_label_images": 100, "boxes_per_image": 2,
                      "cost_per_box_jpy": 10, "rework_factor": 1.5,
                      "engineering_hours": 10, "engineering_rate_jpy": 1000,
                      "ops_monthly_jpy": 0},
            "with_ops": {"label": "運用費あり", "n_label_images": 100, "boxes_per_image": 2,
                         "cost_per_box_jpy": 10, "rework_factor": 1.5,
                         "engineering_hours": 10, "engineering_rate_jpy": 1000,
                         "ops_monthly_jpy": 100000},
        },
        "horizons_months": [6, 12],
        "monthly_volumes": [500, 500000],
    }


def test_vlm_unit_cost_is_measured_cost_divided_by_images():
    assert vlm_unit_cost_jpy(_costs()) == pytest.approx(251 / 300)
    with pytest.raises(ValueError):
        vlm_unit_cost_jpy({"vlm": {"measured_cost_jpy": 251, "measured_n_images": 0}})


def test_fixed_cost_splits_label_and_engineering():
    fc = fixed_cost(_costs()["scenarios"]["cheap"])
    assert fc["label_jpy"] == 3000.0
    assert fc["engineering_jpy"] == 10000.0
    assert fc["total_jpy"] == 13000.0
    assert fc["n_boxes"] == 200


def test_break_even_images_and_monthly_volume():
    unit = 0.5
    assert break_even_images(1000, unit) == 2000
    # 運用費 0 なら「必要な月間枚数」は期間に反比例する
    assert break_even_monthly_volume(1000, 0, unit, 4) == 500
    assert break_even_monthly_volume(1000, 0, unit, 8) == 250
    # 運用費があるぶんだけ必要枚数は増える
    assert break_even_monthly_volume(1000, 100, unit, 4) > 500
    with pytest.raises(ValueError):
        break_even_monthly_volume(1000, 0, unit, 0)


def test_payback_months_returns_none_when_never_breaks_even():
    unit = 1.0
    # 月の節約 = 1.0*1000 - 100 = 900 → 10000/900 ≒ 11.1 ヶ月
    assert payback_months(10000, 100, unit, 1000) == pytest.approx(11.1, abs=0.05)
    # 運用費が VLM の月額以上 → 交点が存在しない
    assert payback_months(10000, 2000, unit, 1000) is None
    assert payback_months(10000, 1000, unit, 1000) is None   # ちょうど等しい場合も回収不能
    # 交点はあるが上限(120 ヶ月)を超える
    assert payback_months(10_000_000, 0, unit, 1000) is None


def test_analyze_structure_and_never_payback_at_poc_scale():
    res = analyze(_costs())
    assert res["vlm_unit_jpy_per_image"] == pytest.approx(0.8367, abs=1e-4)
    cheap = res["scenarios"]["cheap"]
    assert cheap["fixed_cost"]["total_jpy"] == 13000.0
    assert cheap["break_even_images"] == round(13000 / (251 / 300))
    # 期間が延びるほど必要な月間枚数は減る
    assert cheap["break_even_monthly_volume"]["12"] < cheap["break_even_monthly_volume"]["6"]
    # 運用費 10万円/月 は PoC 規模(500枚/月 ≒ 418円)では永久に回収できない
    assert res["scenarios"]["with_ops"]["payback_months"]["500"] is None
    assert res["scenarios"]["with_ops"]["payback_months"]["500000"] is not None


def test_markdown_reports_unrecoverable_cases():
    md = to_markdown(analyze(_costs()))
    assert "損益分岐" in md and "回収不能" in md
    assert "cheap" in md and "with_ops" in md


def test_figure_writes_png(tmp_path):
    out = tmp_path / "breakeven.png"
    assert breakeven_figure(analyze(_costs()), str(out)) == str(out)
    assert out.exists() and out.stat().st_size > 0
