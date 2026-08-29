"""compare(M4 比較ドライバ)の単体テスト。"""

import json

import pytest

from src.analysis.compare import compare, to_markdown


def _write_eval_set(tmp_path):
    eval_set = []
    for i in range(6):
        eval_set.append({
            "image_id": i,
            "file_name": f"{i}.jpg",
            "checklist": ["dog", "car"],
            "ground_truth": {"dog": "present" if i % 2 == 0 else "absent",
                             "car": "present"},
            "meta": {"n_objects": 3, "min_bbox_area_ratio": 0.05 + 0.01 * i,
                     "has_crowd": bool(i % 2),
                     "gt_counts": {"dog": 1 if i % 2 == 0 else 0, "car": 2}},
        })
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(eval_set), encoding="utf-8")
    return str(p), eval_set


def _result(tmp_path, name, eval_set, provider, model_version, *, with_uncertain, with_count):
    records = []
    for e in eval_set:
        js = []
        for cat in e["checklist"]:
            gt = e["ground_truth"][cat]  # 完全正解ベースで組み立てる
            j = {"category": cat, "judgement": gt, "rationale": "x"}
            if with_count:
                j["count"] = e["meta"]["gt_counts"][cat] if gt == "present" else 0
            else:
                j["count"] = None
            js.append(j)
        if with_uncertain and e["image_id"] == 1:
            js[0]["judgement"] = "uncertain"  # 1 件だけ uncertain を混ぜる
            js[0]["count"] = None
        records.append({"image_id": e["image_id"], "judgements": js,
                        "usage": {"input_tokens": 100, "output_tokens": 20},
                        "latency_sec": 0.5 + 0.1 * e["image_id"], "cached": False})
    obj = {"run_id": name, "provider": provider, "model_version": model_version,
           "eval_set_sha256": "deadbeef", "records": records}
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return str(p)


def _pricing(tmp_path):
    p = tmp_path / "pricing.yaml"
    p.write_text("usd_jpy: 160\nmodels:\n  vlm-x:\n    input: 1.0\n    output: 2.0\n",
                 encoding="utf-8")
    return str(p)


def test_compare_structure_and_values(tmp_path):
    eval_path, eval_set = _write_eval_set(tmp_path)
    yolo = _result(tmp_path, "yolo", eval_set, "yolo", "yolo-x",
                   with_uncertain=False, with_count=True)
    vlm = _result(tmp_path, "vlm", eval_set, "gemini", "vlm-x",
                  with_uncertain=True, with_count=True)
    pricing = _pricing(tmp_path)

    out = compare(yolo, vlm, eval_path, pricing_path=pricing, n_boot=200, seed=1)

    assert out["n_images"] == 6
    # YOLO は uncertain を出さない / VLM は 1 件出す
    assert out["models"]["yolo"]["uncertain"]["uncertain"] == 0
    assert out["models"]["vlm"]["uncertain"]["uncertain"] == 1
    # YOLO はローカル推論でコスト 0、VLM は pricing から課金される
    assert out["models"]["yolo"]["cost_jpy"] == 0.0
    assert out["models"]["vlm"]["cost_jpy"] > 0.0
    # 個数 recall(副軸)が両モデルで算出される
    assert out["models"]["yolo"]["count_recall"] is not None
    assert out["models"]["vlm"]["count_recall"] is not None
    # CI は [lo, hi] で 0..1
    lo, hi = out["models"]["yolo"]["macro_f1_ci95"]
    assert 0.0 <= lo <= hi <= 1.0
    # McNemar の鍵が揃う
    assert {"n01", "n10", "p_value"} <= set(out["mcnemar"])
    # 主推論: 画像単位ペア差ブートストラップ(diff = VLM - YOLO)
    for metric in ("macro_f1", "micro_f1", "item_accuracy"):
        d = out["paired_diff"][metric]
        assert {"point_diff", "ci95", "ci90", "p_value_boot"} <= set(d)
        lo, hi = d["ci95"]
        assert lo <= hi
    # item accuracy が strict/excl サマリに追加されている
    assert 0.0 <= out["models"]["yolo"]["strict"]["item_accuracy"] <= 1.0
    # Markdown が生成できる
    md = to_markdown(out)
    assert "YOLO vs VLM" in md and "McNemar" in md and "ペア差ブートストラップ" in md


def test_compare_rejects_mismatched_eval_set(tmp_path):
    eval_path, eval_set = _write_eval_set(tmp_path)
    yolo = _result(tmp_path, "yolo", eval_set, "yolo", "yolo-x",
                   with_uncertain=False, with_count=True)
    vlm = _result(tmp_path, "vlm", eval_set, "gemini", "vlm-x",
                  with_uncertain=False, with_count=True)
    # vlm 側の eval ハッシュを書き換えて不一致にする
    obj = json.loads(open(vlm).read())
    obj["eval_set_sha256"] = "0000"
    open(vlm, "w").write(json.dumps(obj))
    pricing = _pricing(tmp_path)
    with pytest.raises(ValueError, match="評価セットが不一致"):
        compare(yolo, vlm, eval_path, pricing_path=pricing, n_boot=50)
