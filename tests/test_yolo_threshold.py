"""yolo_threshold(しきい値スイープの後処理)の単体テスト。推論はモックで代替。"""

from src.analysis.yolo_threshold import (
    _records_at_threshold, sweep, tune_test_split, _argmax,
)


def _eval_set():
    return [
        {"image_id": 0, "checklist": ["dog"], "ground_truth": {"dog": "present"},
         "meta": {}},
        {"image_id": 1, "checklist": ["dog"], "ground_truth": {"dog": "absent"},
         "meta": {}},
    ]


def _raw():
    # img0: dog を conf0.1 で検出(低い)。img1: dog を conf0.6 で誤検出。
    return {"detections": {"0": {"dog": [0.1]}, "1": {"dog": [0.6]}}}


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


def test_sweep_returns_macro_f1_per_threshold():
    eval_set, raw = _eval_set(), _raw()
    curve = sweep(raw, eval_set, [0.05, 0.3, 0.7])
    assert [c["threshold"] for c in curve] == [0.05, 0.3, 0.7]
    for c in curve:
        assert 0.0 <= c["macro_f1"] <= 1.0
    # t=0.7 では両方 absent(img0 見逃し / img1 正)→ present の F1 は 0
    assert curve[-1]["macro_f1"] == 0.0


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


def test_argmax_picks_highest_macro_f1():
    curve = [{"threshold": 0.1, "macro_f1": 0.8}, {"threshold": 0.2, "macro_f1": 0.9}]
    assert _argmax(curve)["threshold"] == 0.2
