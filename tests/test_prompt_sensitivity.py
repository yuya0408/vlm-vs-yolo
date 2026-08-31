"""prompt_sensitivity(VLM プロンプト水準の後処理)の単体テスト。API 呼び出しは不要。"""

import json

import pytest

from src.analysis.prompt_sensitivity import (
    _argmax_prompt, _spread, analyze, evaluate, load_runs, to_markdown,
)


def _eval_set(n: int = 10) -> list[dict]:
    """dog は前半 present / cat は交互 present の決定的な評価セット。"""
    out = []
    for i in range(n):
        out.append({
            "image_id": i,
            "file_name": f"{i}.jpg",
            "checklist": ["dog", "cat"],
            "ground_truth": {"dog": "present" if i < n // 2 else "absent",
                             "cat": "present" if i % 2 == 0 else "absent"},
            "meta": {"n_items": 2, "min_bbox_area_ratio": 0.1, "has_crowd": False},
        })
    return out


def _run(eval_set: list[dict], flip_ids: set[int] = frozenset(),
         uncertain_ids: set[int] = frozenset(), prompt: str = "concise") -> dict:
    """GT どおりに答えるランを作り、flip_ids の画像だけ dog を誤答させる。"""
    records = []
    for e in eval_set:
        judgements = []
        for cat in e["checklist"]:
            gt = e["ground_truth"][cat]
            j = gt
            if cat == "dog" and e["image_id"] in flip_ids:
                j = "absent" if gt == "present" else "present"
            if cat == "dog" and e["image_id"] in uncertain_ids:
                j = "uncertain"
            judgements.append({"category": cat, "judgement": j,
                               "count": 1 if j == "present" else None, "rationale": ""})
        records.append({"image_id": e["image_id"], "judgements": judgements,
                        "usage": {"input_tokens": 100, "output_tokens": 10},
                        "latency_sec": 5.0, "cached": True})
    return {"run_id": f"gemini-flash-{prompt}-deadbeef", "provider": "gemini",
            "model_version": "gemini-3.5-flash", "prompt": prompt,
            "eval_set_sha256": "abc123", "records": records}


def test_load_runs_labels_from_spec_and_from_prompt_field(tmp_path):
    eval_set = _eval_set()
    p1 = tmp_path / "a.json"
    p1.write_text(json.dumps(_run(eval_set, prompt="concise")), encoding="utf-8")
    p2 = tmp_path / "b.json"
    p2.write_text(json.dumps(_run(eval_set, prompt="deliberate")), encoding="utf-8")

    runs = load_runs([str(p1), f"custom={p2}"])
    assert set(runs) == {"concise", "custom"}  # ラベル省略時は結果の prompt を採用


def test_load_runs_rejects_mismatched_eval_set(tmp_path):
    eval_set = _eval_set()
    p1 = tmp_path / "a.json"
    p1.write_text(json.dumps(_run(eval_set, prompt="concise")), encoding="utf-8")
    other = _run(eval_set, prompt="deliberate")
    other["eval_set_sha256"] = "different"
    p2 = tmp_path / "b.json"
    p2.write_text(json.dumps(other), encoding="utf-8")

    with pytest.raises(ValueError, match="評価セットが不一致"):
        load_runs([str(p1), str(p2)])


def test_load_runs_requires_two_levels(tmp_path):
    p1 = tmp_path / "a.json"
    p1.write_text(json.dumps(_run(_eval_set())), encoding="utf-8")
    with pytest.raises(ValueError, match="2 本以上"):
        load_runs([str(p1)])


def test_evaluate_perfect_run_and_uncertain_rate():
    eval_set = _eval_set()
    perfect = evaluate(_run(eval_set)["records"], eval_set)
    assert perfect["macro_f1"] == 1.0
    assert perfect["uncertain_rate"] == 0.0

    # dog を 2 枚 uncertain にすると strict では誤り扱い、uncertain 率は 2/20
    withunc = evaluate(_run(eval_set, uncertain_ids={0, 1})["records"], eval_set)
    assert withunc["uncertain_rate"] == 0.1
    assert withunc["macro_f1"] < 1.0
    assert withunc["coverage_excl"] == 0.9


def test_spread_and_argmax_are_deterministic():
    rows = {"a": {"micro_f1": 0.90}, "b": {"micro_f1": 0.95}, "c": {"micro_f1": 0.95}}
    sp = _spread(rows, "micro_f1")
    assert sp["min_prompt"] == "a" and sp["range_pt"] == 5.0
    # タイは名前順で決定的に解く
    assert _argmax_prompt(rows, "micro_f1") == "b"


def test_analyze_selects_on_tune_and_reports_on_test():
    eval_set = _eval_set(20)
    tune_ids = {e["image_id"] for e in eval_set} - set(range(12))  # 参考: 分割は下で確認
    runs = {
        "concise": _run(eval_set, flip_ids={0, 1, 2, 3}),
        "deliberate": _run(eval_set),                 # 完璧 → tune でも test でも最良
        "calibrated": _run(eval_set, uncertain_ids={4, 5}),
    }
    res = analyze(runs, eval_set, baseline="concise", n_boot=200, test_frac=0.6, seed=42)

    assert res["n_prompts"] == 3
    sel = res["selection"]
    assert sel["selected_prompt"] == "deliberate"
    assert sel["selected_is_baseline"] is False
    assert sel["n_tune"] + sel["n_test"] == len(eval_set)
    assert sel["test_at_selected"]["macro_f1"] >= sel["test_at_baseline"]["macro_f1"]
    assert tune_ids  # 分割そのものは yolo_threshold 側のテストで担保

    # ばらつきは全/tune/test の 3 スコープで出す
    assert set(res["spread"]) >= {"full", "tune", "test", "uncertain_rate_full"}
    assert res["spread"]["full"]["macro_f1"]["max_prompt"] == "deliberate"

    # baseline との対応あり差は baseline 以外の水準ぶん
    assert set(res["vs_baseline"]) == {"deliberate", "calibrated"}
    d = res["vs_baseline"]["deliberate"]
    assert d["macro_f1"]["point_diff_pt"] > 0        # diff = 水準 − baseline
    assert d["mcnemar"]["n10"] > 0                   # baseline 誤 / 水準 正
    assert "vs_yolo_on_test" not in res


def test_analyze_flags_verdict_change_against_yolo():
    eval_set = _eval_set(20)
    # YOLO は dog を 6 枚落とす(見逃し)。baseline VLM も同程度に誤り、選定水準は完璧。
    yolo = _run(eval_set, flip_ids=set(range(6)))["records"]
    runs = {
        "concise": _run(eval_set, flip_ids=set(range(6))),   # YOLO と同一 → 差なし
        "deliberate": _run(eval_set),                        # 完璧 → 差が付く
    }
    res = analyze(runs, eval_set, baseline="concise", n_boot=200, yolo_records=yolo)

    vy = res["vs_yolo_on_test"]
    assert vy["baseline"]["macro_f1"]["point_diff_pt"] == 0.0
    assert vy["baseline"]["significant_at_0.05"] is False
    assert vy["selected"]["macro_f1"]["point_diff_pt"] > 0
    assert vy["verdict_changed"] == vy["selected"]["significant_at_0.05"]
    assert "反転" in to_markdown(res)


def test_analyze_rejects_unknown_baseline():
    eval_set = _eval_set()
    runs = {"concise": _run(eval_set), "deliberate": _run(eval_set)}
    with pytest.raises(ValueError, match="baseline"):
        analyze(runs, eval_set, baseline="missing", n_boot=50)


def test_markdown_contains_every_prompt_row():
    eval_set = _eval_set(20)
    runs = {"concise": _run(eval_set, flip_ids={1}), "deliberate": _run(eval_set)}
    md = to_markdown(analyze(runs, eval_set, baseline="concise", n_boot=200))
    assert "concise" in md and "**deliberate**" in md
    assert "プロンプト間のばらつき" in md
