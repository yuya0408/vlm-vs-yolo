"""yolo_threshold(しきい値スイープの後処理)の単体テスト。推論はモックで代替。"""

import pytest

from src.analysis.yolo_threshold import (
    DEFAULT_BAND, DEFAULT_THRESHOLDS, _argmax, _records_at_threshold, analyze,
    band_summary, export_records, sweep, tune_test_split,
)


def _eval_set():
    return [
        {"image_id": 0, "checklist": ["dog"], "ground_truth": {"dog": "present"},
         "meta": {"gt_counts": {"dog": 1}}},
        {"image_id": 1, "checklist": ["dog"], "ground_truth": {"dog": "absent"},
         "meta": {"gt_counts": {}}},
    ]


def _raw():
    # img0: dog を conf0.1 で検出(低い)。img1: dog を conf0.6 で誤検出。
    return {"weights": "yolo26x.pt", "base_conf": 0.001,
            "detections": {"0": {"dog": [0.1]}, "1": {"dog": [0.6]}},
            "latencies": {"0": 0.31, "1": 0.35}}


def _json_keys(obj, acc=None):
    acc = set() if acc is None else acc
    if isinstance(obj, dict):
        for k, v in obj.items():
            acc.add(k)
            _json_keys(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _json_keys(v, acc)
    return acc


def test_records_at_threshold_present_absent_and_count():
    eval_set, raw = _eval_set(), _raw()
    # t=0.05: 両方 present(img0 正・img1 誤)
    recs = _records_at_threshold(raw, eval_set, 0.05)
    by = {r["image_id"]: r["judgements"][0] for r in recs}
    assert by[0]["judgement"] == "present" and by[0]["count"] == 1
    assert by[1]["judgement"] == "present"
    # t=0.3: img0 の検出(0.1)は閾値割れ→absent(見逃し)、img1(0.6)は present のまま
    recs = _records_at_threshold(raw, eval_set, 0.3)
    by = {r["image_id"]: r["judgements"][0] for r in recs}
    assert by[0]["judgement"] == "absent" and by[0]["count"] == 0
    assert by[1]["judgement"] == "present"


def test_sweep_returns_all_report_metrics_per_threshold():
    eval_set, raw = _eval_set(), _raw()
    curve = sweep(raw, eval_set, [0.05, 0.3, 0.7])
    assert [c["threshold"] for c in curve] == [0.05, 0.3, 0.7]
    # REPORT §2 のスイープ表と同じ 4 指標が揃っていること
    for c in curve:
        assert set(c) == {"threshold", "macro_f1", "micro_f1", "item_accuracy", "count_recall"}
        assert 0.0 <= c["macro_f1"] <= 1.0
        assert 0.0 <= c["item_accuracy"] <= 1.0
    # t=0.7 では両方 absent(img0 見逃し / img1 正)→ present の F1 は 0、item acc は 1/2
    assert curve[-1]["macro_f1"] == 0.0
    assert curve[-1]["item_accuracy"] == 0.5
    # t=0.05 では img0 の dog 1 個を拾えている → 個数 recall = 1.0
    assert curve[0]["count_recall"] == 1.0


def test_default_grid_covers_low_confidence_region():
    """有無還元では低 conf が最適になりうる。0.05 刻みだと最適点を跨ぐ(REPORT §2)。"""
    assert 0.075 in DEFAULT_THRESHOLDS
    assert 0.01 in DEFAULT_THRESHOLDS
    assert 0.25 in DEFAULT_THRESHOLDS  # 既定値は対照として必ず含める
    assert len([t for t in DEFAULT_THRESHOLDS if t < 0.05]) >= 2
    assert DEFAULT_THRESHOLDS == sorted(DEFAULT_THRESHOLDS)


def test_argmax_selects_by_requested_metric():
    curve = [
        {"threshold": 0.01, "macro_f1": 0.9, "micro_f1": 0.5, "item_accuracy": 0.5,
         "count_recall": 1.0},
        {"threshold": 0.2, "macro_f1": 0.8, "micro_f1": 0.9, "item_accuracy": 0.9,
         "count_recall": 0.7},
    ]
    # macro-F1 の argmax は低 conf の退化点に張り付く。既定の選定指標にしない理由。
    assert _argmax(curve, "macro_f1")["threshold"] == 0.01
    assert _argmax(curve, "micro_f1")["threshold"] == 0.2
    assert _argmax(curve, "item_accuracy")["threshold"] == 0.2
    with pytest.raises(ValueError):
        _argmax(curve, "not_a_metric")


def test_argmax_tie_breaks_to_lower_threshold():
    curve = [{"threshold": 0.05, "micro_f1": 0.9}, {"threshold": 0.5, "micro_f1": 0.9}]
    assert _argmax(curve, "micro_f1")["threshold"] == 0.05


def test_band_summary_reports_min_max_inside_band():
    curve = [
        {"threshold": 0.005, "macro_f1": 0.10, "micro_f1": 0.1, "item_accuracy": 0.1,
         "count_recall": 0.1},
        {"threshold": 0.01, "macro_f1": 0.945, "micro_f1": 0.941, "item_accuracy": 0.962,
         "count_recall": 0.985},
        {"threshold": 0.075, "macro_f1": 0.929, "micro_f1": 0.954, "item_accuracy": 0.972,
         "count_recall": 0.927},
        {"threshold": 0.9, "macro_f1": 0.0, "micro_f1": 0.0, "item_accuracy": 0.0,
         "count_recall": 0.0},
    ]
    b = band_summary(curve, (0.01, 0.10))
    assert b["n_points"] == 2                      # バンド外(0.005 と 0.9)は入らない
    assert b["macro_f1"] == [0.929, 0.945]
    assert b["micro_f1"] == [0.941, 0.954]
    with pytest.raises(ValueError):
        band_summary(curve, (0.3, 0.4))            # グリッドに点が無いバンド


def test_tune_test_split_is_deterministic_and_disjoint():
    eval_set = [{"image_id": i, "checklist": ["x"], "ground_truth": {"x": "absent"},
                 "meta": {}} for i in range(10)]
    tune, test = tune_test_split(eval_set, test_frac=0.6, seed=42)
    assert len(test) == 6 and len(tune) == 4
    ids_t = {e["image_id"] for e in tune}
    ids_s = {e["image_id"] for e in test}
    assert ids_t.isdisjoint(ids_s) and (ids_t | ids_s) == set(range(10))
    # 同じシードなら同じ分割
    tune2, _ = tune_test_split(eval_set, test_frac=0.6, seed=42)
    assert {e["image_id"] for e in tune2} == ids_t


def test_analyze_reports_operating_point_and_band():
    eval_set, raw = _eval_set(), _raw()
    res = analyze(raw, eval_set, [0.01, 0.05, 0.075, 0.1, 0.25], test_frac=0.5, seed=42)

    assert res["selection_metric"] == "micro_f1"
    op = res["operating_point"]
    assert op["threshold"] in res["thresholds"]
    # 3 本の線が明示的に出ていること(「テストで選んでテストで報告」への反証材料)
    assert set(op["lines"]) == {"standard_practice_argmax", "leak_free_tune_split",
                                "agreement_item_accuracy_argmax"}
    assert isinstance(op["converged"], bool)
    assert res["band"]["low"] == DEFAULT_BAND[0]
    assert "default_0.25_full" in res
    # 選定は VLM を一切参照しない = 出力のどこにも VLM 由来のキーが無い
    assert "vlm" not in _json_keys(res)


def test_export_records_is_consumable_as_a_run_result():
    """export の出力が compare/error_taxonomy が食えるラン形式になっていること。"""
    eval_set, raw = _eval_set(), _raw()
    obj = export_records(raw, eval_set, 0.075, "data/eval_set.json", "a" * 64)

    # runner.py の結果スキーマと同じキー
    for key in ("run_id", "provider", "model_version", "eval_set_path",
                "eval_set_sha256", "created_at", "records"):
        assert key in obj
    assert obj["provider"] == "yolo"
    assert obj["eval_set_sha256"] == "a" * 64
    assert "conf=0.075" in obj["model_version"]

    rec = obj["records"][0]
    assert set(rec) == {"image_id", "judgements", "usage", "latency_sec", "cached"}
    # ローカル推論なのでトークン 0(コスト 0 の根拠)
    assert rec["usage"] == {"input_tokens": 0, "output_tokens": 0}
    # レイテンシは capture 時の実測値を引き継ぐ(捏造しない)
    lat = {r["image_id"]: r["latency_sec"] for r in obj["records"]}
    assert lat == {0: 0.31, 1: 0.35}

    # 実際に指標計算に通せること
    from src.analysis.metrics import compute_metrics
    rep = compute_metrics(obj["records"], eval_set, mode="strict")
    assert 0.0 <= rep.macro_f1 <= 1.0


def test_export_without_latencies_falls_back_to_zero():
    """古い capture(レイテンシ未記録)でも落ちない。値は 0 で、CLI が警告を出す。"""
    eval_set = _eval_set()
    raw = {"weights": "yolo26x.pt", "detections": {"0": {"dog": [0.9]}, "1": {}}}
    obj = export_records(raw, eval_set, 0.075, "data/eval_set.json", "b" * 64)
    assert all(r["latency_sec"] == 0.0 for r in obj["records"])
