"""thinking_cost(思考トークンの費用の推定)の単体テスト。外部依存なし。"""

import pytest

from src.analysis.thinking_cost import estimate, judgement_agreement, thinking_ratio


def _rec(image_id, out, thoughts=None, judgement="present"):
    usage = {"input_tokens": 1000, "output_tokens": out}
    if thoughts is not None:
        usage["thoughts_tokens"] = thoughts
    return {"image_id": image_id, "usage": usage,
            "judgements": [{"category": "dog", "judgement": judgement}]}


def test_ratio_is_sum_over_sum():
    assert thinking_ratio([_rec(1, 100, 300), _rec(2, 300, 100)]) == pytest.approx(1.0)


def test_estimate_borrows_only_the_ratio_from_the_sample():
    base = {"run_id": "base", "records": [_rec(i, 1_000_000) for i in range(4)]}
    sample = {"run_id": "s", "model_version": "v2",
              "records": [_rec(0, 500, 1000), _rec(1, 500, 1000, judgement="absent")]}
    res = estimate(base, sample, {"input": 0.0, "output": 1.0}, usd_jpy=100)
    # 本番ランの出力 4M トークン = 400 円、思考は比 2.0 で 800 円
    assert res["recorded_cost_jpy"] == 400.0
    assert res["thinking_cost_jpy"] == 800.0
    assert res["unit_jpy_per_image"] == 300.0
    assert res["judgement_agreement"] == [1, 2]


def test_sample_image_must_exist_in_base():
    base = {"run_id": "base", "records": [_rec(1, 10)]}
    sample = {"run_id": "s", "records": [_rec(2, 10, 10)]}
    with pytest.raises(ValueError):
        estimate(base, sample, {"input": 1.0, "output": 1.0}, usd_jpy=1)


def test_agreement_counts_items():
    assert judgement_agreement([_rec(1, 1, 1)], [_rec(1, 1)]) == (1, 1)
