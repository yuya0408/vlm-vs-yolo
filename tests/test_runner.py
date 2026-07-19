"""runner の mock E2E テスト(評価セット → 結果 → 指標まで疎通、API 費ゼロ)。"""

import json

from src.runner import cache_key, run_eval
from src.analysis.metrics import compute_metrics


EVAL_SET = [
    {"image_id": 1, "file_name": "a.jpg", "checklist": ["dog", "cat"],
     "ground_truth": {"dog": "present", "cat": "absent"}},
    {"image_id": 2, "file_name": "b.jpg", "checklist": ["car", "person"],
     "ground_truth": {"car": "present", "person": "present"}},
    {"image_id": 3, "file_name": "c.jpg", "checklist": ["dog", "cat", "car"],
     "ground_truth": {"dog": "absent", "cat": "present", "car": "absent"}},
]


def _write_eval(tmp_path):
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(EVAL_SET), encoding="utf-8")
    return str(p)


def test_cache_key_is_order_independent():
    k1 = cache_key("mock", "m", "p", 1, ["dog", "cat"])
    k2 = cache_key("mock", "m", "p", 1, ["cat", "dog"])
    assert k1 == k2
    assert k1 != cache_key("mock", "m", "p", 2, ["dog", "cat"])


def test_mock_e2e(tmp_path):
    eval_path = _write_eval(tmp_path)
    out = run_eval(eval_path, {"provider": "mock"},
                   cache_dir=str(tmp_path / "cache"),
                   results_dir=str(tmp_path / "results"),
                   images_dir=str(tmp_path / "imgs"))
    obj = json.loads(open(out, encoding="utf-8").read())
    assert obj["provider"] == "mock"
    assert obj["model_version"] == "mock-model-v1"
    assert len(obj["records"]) == 3
    # 各レコードが checklist 全項目をカバー
    by_id = {e["image_id"]: e for e in EVAL_SET}
    for rec in obj["records"]:
        cats = {j["category"] for j in rec["judgements"]}
        assert cats == set(by_id[rec["image_id"]]["checklist"])
    # 指標計算まで通る
    rep = compute_metrics(obj["records"], EVAL_SET, mode="strict")
    assert 0.0 <= rep.macro_f1 <= 1.0


def test_cache_hit_on_rerun(tmp_path):
    eval_path = _write_eval(tmp_path)
    cache = str(tmp_path / "cache")
    results = str(tmp_path / "results")
    out = run_eval(eval_path, {"provider": "mock"}, cache_dir=cache,
                   results_dir=results, images_dir=str(tmp_path / "imgs"))
    first = json.loads(open(out, encoding="utf-8").read())
    assert all(r["cached"] is False for r in first["records"])

    # 結果ファイルだけ消してキャッシュは残す → 再実行で全件キャッシュヒット
    import os
    os.remove(out)
    out2 = run_eval(eval_path, {"provider": "mock"}, cache_dir=cache,
                    results_dir=results, images_dir=str(tmp_path / "imgs"))
    second = json.loads(open(out2, encoding="utf-8").read())
    assert all(r["cached"] is True for r in second["records"])
