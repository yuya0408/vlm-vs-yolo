"""MockProvider の単体テスト。"""

from pathlib import Path

from src.providers.mock import MockProvider

CHECKLIST = ["dog", "cat", "car", "truck", "person"]


def test_same_input_returns_same_result():
    """同じ入力で2回呼ぶと完全に同じ結果が返る(決定論性)。"""
    provider = MockProvider()
    r1 = provider.judge(Path("a.jpg"), CHECKLIST)
    r2 = provider.judge(Path("a.jpg"), CHECKLIST)
    assert r1 == r2


def test_all_items_judged():
    """checklist の全項目に判定が1つずつ返る。"""
    provider = MockProvider()
    r1 = provider.judge(Path("a.jpg"), CHECKLIST)
    assert len(r1.judgements) == len(CHECKLIST)
    assert set(j.category for j in r1.judgements) == set(CHECKLIST)


def test_judgement_values_are_valid():
    """判定値は present / absent / uncertain のいずれか。"""
    provider = MockProvider()
    r1 = provider.judge(Path("a.jpg"), CHECKLIST)
    for j in r1.judgements:
        assert j.judgement in {"present", "absent", "uncertain"}