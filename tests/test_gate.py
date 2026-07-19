"""gate の単体テスト。CI の成否を決めるロジックなので最優先でテストする。"""

import json

import yaml

from src.gate import cost_guard, regression_gate


EVAL_SET = [
    {"image_id": 1, "checklist": ["dog", "cat", "car"],
     "ground_truth": {"dog": "present", "cat": "absent", "car": "present"}},
    {"image_id": 2, "checklist": ["dog", "cat", "car"],
     "ground_truth": {"dog": "present", "cat": "present", "car": "absent"}},
]


def _judge(d):
    return [{"category": k, "judgement": v} for k, v in d.items()]


def _write_result(path, eval_path, records, model="mock-model-v1", created="2026-06-10T00:00:00"):
    obj = {"run_id": path.stem, "provider": "mock", "model_version": model,
           "eval_set_path": str(eval_path), "eval_set_sha256": "x",
           "created_at": created, "records": records}
    path.write_text(json.dumps(obj), encoding="utf-8")
    return str(path)


def _setup(tmp_path):
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps(EVAL_SET), encoding="utf-8")
    perfect = [
        {"image_id": 1, "judgements": _judge({"dog": "present", "cat": "absent", "car": "present"})},
        {"image_id": 2, "judgements": _judge({"dog": "present", "cat": "present", "car": "absent"})},
    ]
    return eval_path, perfect


CONFIG = {"regression": {"max_f1_drop": 0.02, "report_mcnemar": True},
          "cost": {"monthly_cost_limit_jpy": 500}}


def test_gate_passes_when_no_drop(tmp_path):
    eval_path, perfect = _setup(tmp_path)
    base = _write_result(tmp_path / "base.json", eval_path, perfect)
    cur = _write_result(tmp_path / "cur.json", eval_path, perfect)
    assert regression_gate(cur, base, CONFIG) is True


def test_gate_fails_on_large_drop(tmp_path):
    eval_path, perfect = _setup(tmp_path)
    base = _write_result(tmp_path / "base.json", eval_path, perfect)
    bad_records = [
        {"image_id": 1, "judgements": _judge({"dog": "absent", "cat": "absent", "car": "absent"})},
        {"image_id": 2, "judgements": _judge({"dog": "absent", "cat": "absent", "car": "absent"})},
    ]
    cur = _write_result(tmp_path / "cur.json", eval_path, bad_records)
    assert regression_gate(cur, base, CONFIG) is False


def _pricing(tmp_path):
    p = tmp_path / "pricing.yaml"
    p.write_text(yaml.safe_dump(
        {"usd_jpy": 150, "models": {"gemini-x": {"input": 10.0, "output": 30.0}}}),
        encoding="utf-8")
    return str(p)


def test_cost_guard_blocks_over_limit(tmp_path):
    rdir = tmp_path / "results"
    rdir.mkdir()
    # 1 レコードで (10+30)*150 = 6000 円 > 500
    _write_result(rdir / "r1.json", tmp_path / "eval.json",
                  [{"image_id": 1, "judgements": [],
                    "usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000}}],
                  model="gemini-x", created="2026-06-01T00:00:00")
    pricing = _pricing(tmp_path)
    assert cost_guard("2026-06", CONFIG, results_dir=str(rdir), pricing_path=pricing) is False


def test_cost_guard_passes_under_limit_and_month_filter(tmp_path):
    rdir = tmp_path / "results"
    rdir.mkdir()
    # 今月は小さなコスト
    _write_result(rdir / "r1.json", tmp_path / "eval.json",
                  [{"image_id": 1, "judgements": [],
                    "usage": {"input_tokens": 1000, "output_tokens": 1000}}],
                  model="gemini-x", created="2026-06-01T00:00:00")
    # 別月の高コストは集計対象外
    _write_result(rdir / "r2.json", tmp_path / "eval.json",
                  [{"image_id": 1, "judgements": [],
                    "usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000}}],
                  model="gemini-x", created="2026-05-01T00:00:00")
    pricing = _pricing(tmp_path)
    assert cost_guard("2026-06", CONFIG, results_dir=str(rdir), pricing_path=pricing) is True
