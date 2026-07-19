"""metrics の単体テスト。

方針: 手計算で検証できる極小ケース(画像2枚 × 3項目)を固定で書く。
mock プロバイダではなく、結果 dict を直接組み立ててテストする(依存を切る)。
"""

import pytest

from src.analysis.metrics import compute_metrics, confusion_counts


# 画像2枚 × checklist[dog, cat, car]
#   img1 gt: dog=present, cat=absent,  car=present
#   img2 gt: dog=present, cat=present, car=absent
# 予測:
#   img1: dog=present,  cat=present,    car=uncertain
#   img2: dog=present,  cat=uncertain,  car=absent
EVAL_SET = [
    {"image_id": 1, "checklist": ["dog", "cat", "car"],
     "ground_truth": {"dog": "present", "cat": "absent", "car": "present"}},
    {"image_id": 2, "checklist": ["dog", "cat", "car"],
     "ground_truth": {"dog": "present", "cat": "present", "car": "absent"}},
]

RESULTS = [
    {"image_id": 1, "judgements": [
        {"category": "dog", "judgement": "present"},
        {"category": "cat", "judgement": "present"},
        {"category": "car", "judgement": "uncertain"}]},
    {"image_id": 2, "judgements": [
        {"category": "dog", "judgement": "present"},
        {"category": "cat", "judgement": "uncertain"},
        {"category": "car", "judgement": "absent"}]},
]


def test_confusion_counts():
    c = confusion_counts(RESULTS, EVAL_SET)
    assert c["dog"]["pp"] == 2
    assert c["cat"]["ap"] == 1 and c["cat"]["pu"] == 1
    assert c["car"]["pu"] == 1 and c["car"]["aa"] == 1


def test_strict_mode_counts_uncertain_as_error():
    """strict: uncertain は present の取りこぼし(FN)に算入される。"""
    rep = compute_metrics(RESULTS, EVAL_SET, mode="strict")
    by_cat = {m.category: m for m in rep.per_category}
    # dog: tp=2, fp=0, fn=0 → 完璧
    assert by_cat["dog"].precision == 1.0
    assert by_cat["dog"].recall == 1.0
    assert by_cat["dog"].f1 == 1.0
    # car: gt=present を uncertain にした分が FN → recall 0
    assert by_cat["car"].recall == 0.0
    assert rep.macro_f1 == pytest.approx(1 / 3)
    assert rep.micro_f1 == pytest.approx(0.5714285, abs=1e-6)
    assert rep.coverage == 1.0


def test_excl_mode_coverage():
    """excl: uncertain を除外。coverage = 確定判定数 / 全項目数 = 4/6。"""
    rep = compute_metrics(RESULTS, EVAL_SET, mode="excl")
    assert rep.coverage == pytest.approx(4 / 6)
    # uncertain を除けば見逃しが消え、micro recall=1.0 → micro_f1=0.8
    assert rep.micro_f1 == pytest.approx(0.8)


def test_macro_vs_micro():
    """クラス不均衡で macro と micro が乖離する。"""
    rep = compute_metrics(RESULTS, EVAL_SET, mode="strict")
    assert rep.macro_f1 != pytest.approx(rep.micro_f1)


def test_missing_judgement_raises():
    bad = [
        {"image_id": 1, "judgements": [
            {"category": "dog", "judgement": "present"},
            {"category": "cat", "judgement": "present"}]},  # car 欠落
        RESULTS[1],
    ]
    with pytest.raises(ValueError, match="欠落"):
        compute_metrics(bad, EVAL_SET, mode="strict")


def test_image_id_mismatch_raises():
    bad = [RESULTS[0], {"image_id": 999, "judgements": RESULTS[1]["judgements"]}]
    with pytest.raises(ValueError, match="一致しない"):
        compute_metrics(bad, EVAL_SET, mode="strict")


def test_invalid_judgement_raises():
    bad = [
        {"image_id": 1, "judgements": [
            {"category": "dog", "judgement": "maybe"},
            {"category": "cat", "judgement": "present"},
            {"category": "car", "judgement": "absent"}]},
        RESULTS[1],
    ]
    with pytest.raises(ValueError, match="不正な judgement"):
        compute_metrics(bad, EVAL_SET, mode="strict")
