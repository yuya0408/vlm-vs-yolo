"""count_recall(副軸=個数 recall)の単体テスト。"""

import pytest

from src.analysis.metrics import count_recall


# gt_counts: 画像1 dog=3, 画像2 dog=2/cat=1
EVAL_SET = [
    {"image_id": 1, "checklist": ["dog"], "ground_truth": {"dog": "present"},
     "meta": {"gt_counts": {"dog": 3}}},
    {"image_id": 2, "checklist": ["dog", "cat", "car"],
     "ground_truth": {"dog": "present", "cat": "present", "car": "absent"},
     "meta": {"gt_counts": {"dog": 2, "cat": 1}}},
]


def _res(image_id, counts):
    return {"image_id": image_id,
            "judgements": [{"category": c, "judgement": "present" if n else "absent",
                            "count": n} for c, n in counts.items()]}


def test_count_recall_basic():
    # 予測: 画像1 dog=2(3個中2個)、画像2 dog=2(全部)/cat=0(取りこぼし)/car=0
    results = [_res(1, {"dog": 2}),
               _res(2, {"dog": 2, "cat": 0, "car": 0})]
    out = count_recall(results, EVAL_SET)
    # 見つけた = min(2,3)+min(2,2)+min(0,1) = 2+2+0 = 4、GT総数 = 3+2+1 = 6
    assert out["n_gt_instances"] == 6
    assert out["recall"] == pytest.approx(4 / 6)
    assert out["per_category"]["dog"] == pytest.approx((2 + 2) / (3 + 2))
    assert out["per_category"]["cat"] == pytest.approx(0.0)


def test_overcount_is_capped():
    # 予測が GT より多くても min で頭打ち(過剰検出で recall が 1 を超えない)
    results = [_res(1, {"dog": 9}), _res(2, {"dog": 9, "cat": 9, "car": 0})]
    out = count_recall(results, EVAL_SET)
    assert out["recall"] == pytest.approx(1.0)


def test_all_counts_missing_raises():
    """全予測に count が無い(個数を出さないモデル)→ 副軸は計算不能。"""
    results = [
        {"image_id": 1, "judgements": [{"category": "dog", "judgement": "present"}]},
        {"image_id": 2, "judgements": [
            {"category": "dog", "judgement": "present"},
            {"category": "cat", "judgement": "absent"},
            {"category": "car", "judgement": "absent"}]},
    ]
    with pytest.raises(ValueError, match="count が無い"):
        count_recall(results, EVAL_SET)


def test_null_count_treated_as_zero():
    """VLM が present 個体を取りこぼし absent(count=null)とした分は 0 個検出として算入。"""
    # 画像1 dog=3 を absent(null)で見逃し、画像2 dog=2 は当てて count=2
    results = [
        {"image_id": 1, "judgements": [{"category": "dog", "judgement": "absent", "count": None}]},
        _res(2, {"dog": 2, "cat": 1, "car": 0}),
    ]
    out = count_recall(results, EVAL_SET)
    # 見つけた = 0(画像1 dog) + 2(画像2 dog) + 1(画像2 cat) = 3、GT = 3+2+1 = 6
    assert out["recall"] == pytest.approx(3 / 6)
