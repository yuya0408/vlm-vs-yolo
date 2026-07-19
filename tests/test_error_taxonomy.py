"""error_taxonomy(誤り分類・代表例抽出)の単体テスト。"""

from src.analysis.error_taxonomy import _confusable_map, _classify, taxonomy, to_markdown


def test_confusable_map_is_bidirectional():
    m = _confusable_map({"confusable_pairs": [["dog", "cat"], ["car", "truck"]]})
    assert m["dog"] == {"cat"} and m["cat"] == {"dog"}
    assert m["truck"] == {"car"}


def test_classify_buckets():
    conf = {"dog": {"cat"}, "cat": {"dog"}}
    present = {"dog", "person"}
    # 正解 → None
    assert _classify("present", "present", "dog", present, conf) is None
    # 見逃し
    assert _classify("present", "absent", "dog", present, conf) == "missed_present"
    # 混同 FP: cat は absent だが同画像に dog(present)がいる
    assert _classify("absent", "present", "cat", present, conf) == "fp_confusable"
    # 混同相方が居ない FP
    assert _classify("absent", "present", "tv", present, conf) == "fp_other"
    # uncertain
    assert _classify("present", "uncertain", "dog", present, conf) == "uncertain_on_present"
    assert _classify("absent", "uncertain", "tv", present, conf) == "uncertain_on_absent"


def _mk(image_id, checklist, gt, preds):
    return ({"image_id": image_id, "checklist": checklist, "ground_truth": gt,
             "meta": {"min_bbox_area_ratio": 0.01}},
            {"image_id": image_id,
             "judgements": [{"category": c, "judgement": preds[c], "rationale": f"r-{c}"}
                            for c in checklist]})


def test_taxonomy_counts_and_examples():
    confusable = {"dog": {"cat"}, "cat": {"dog"}}
    # img0: dog present だが absent と誤判定(見逃し)、cat は absent を present(混同FP)
    e0, r0 = _mk(0, ["dog", "cat"], {"dog": "present", "cat": "absent"},
                 {"dog": "absent", "cat": "present"})
    # img1: 全部正解
    e1, r1 = _mk(1, ["dog", "cat"], {"dog": "present", "cat": "absent"},
                 {"dog": "present", "cat": "absent"})
    result = {"run_id": "x", "model_version": "m", "records": [r0, r1]}
    tax = taxonomy(result, [e0, e1], confusable, max_examples=5)

    assert tax["total_errors"] == 2
    assert tax["counts"]["missed_present"] == 1
    assert tax["counts"]["fp_confusable"] == 1
    assert len(tax["examples"]["missed_present"]) == 1
    assert tax["examples"]["missed_present"][0]["category"] == "dog"
    # Markdown が生成できる
    md = to_markdown({"yolo": tax, "vlm": tax})
    assert "誤り taxonomy" in md and "missed_present" in md
