"""GeminiProvider のパース/検証テスト。generate_fn を差し込み、API を呼ばずに検証。"""

import json
from pathlib import Path

import pytest

from src.providers.gemini import GeminiProvider

TEMPLATE = "judge these. Checklist: {checklist}"


def _provider(text, usage=None):
    """常に固定 text を返す generate_fn を仕込んだ provider。"""
    usage = usage or {"input_tokens": 100, "output_tokens": 20}

    def fake(model, image_path, prompt):
        # プロンプトに checklist が差し込まれていることも確認
        assert "Checklist:" in prompt
        return text, usage, model + "-001"

    return GeminiProvider(model="gemini-x", prompt_template=TEMPLATE, generate_fn=fake)


def test_valid_response_parsed_and_ordered():
    text = json.dumps([
        {"category": "cat", "judgement": "absent", "count": None, "rationale": "no cat"},
        {"category": "dog", "judgement": "present", "count": 2, "rationale": "two dogs"},
        {"category": "tv", "judgement": "uncertain", "count": None, "rationale": "occluded"},
    ])
    resp = _provider(text).judge(Path("a.jpg"), ["dog", "cat", "tv"])
    # checklist 順に並ぶ
    assert [j.category for j in resp.judgements] == ["dog", "cat", "tv"]
    by = {j.category: j for j in resp.judgements}
    assert by["dog"].judgement == "present" and by["dog"].count == 2
    assert by["cat"].judgement == "absent" and by["cat"].count is None
    assert by["tv"].judgement == "uncertain" and by["tv"].count is None
    assert resp.usage.input_tokens == 100 and resp.usage.output_tokens == 20
    assert resp.model_version == "gemini-x-001"


def test_present_with_null_count_defaults_to_one():
    text = json.dumps([{"category": "dog", "judgement": "present", "count": None}])
    resp = _provider(text).judge(Path("a.jpg"), ["dog"])
    assert resp.judgements[0].count == 1  # present の含意から最低 1


def test_absent_count_forced_to_none_even_if_model_returns_number():
    text = json.dumps([{"category": "dog", "judgement": "absent", "count": 5}])
    resp = _provider(text).judge(Path("a.jpg"), ["dog"])
    assert resp.judgements[0].count is None


def test_markdown_fence_is_stripped():
    text = "```json\n" + json.dumps([{"category": "dog", "judgement": "present", "count": 1}]) + "\n```"
    resp = _provider(text).judge(Path("a.jpg"), ["dog"])
    assert resp.judgements[0].judgement == "present"


def test_missing_item_raises():
    text = json.dumps([{"category": "dog", "judgement": "present", "count": 1}])
    with pytest.raises(ValueError, match="欠落"):
        _provider(text).judge(Path("a.jpg"), ["dog", "cat"])


def test_extra_item_raises():
    text = json.dumps([
        {"category": "dog", "judgement": "present", "count": 1},
        {"category": "zebra", "judgement": "present", "count": 1},
    ])
    with pytest.raises(ValueError, match="checklist 外"):
        _provider(text).judge(Path("a.jpg"), ["dog"])


def test_invalid_judgement_raises():
    text = json.dumps([{"category": "dog", "judgement": "maybe", "count": 1}])
    with pytest.raises(ValueError, match="不正な judgement"):
        _provider(text).judge(Path("a.jpg"), ["dog"])


def test_non_json_raises():
    with pytest.raises(ValueError, match="JSON パース失敗"):
        _provider("not json at all").judge(Path("a.jpg"), ["dog"])
