"""build_eval_set の単体テスト。実 COCO をDLせず、小さな合成 COCO JSON で検証する。"""

import json
import random

import pytest

from src.data.build_eval_set import (
    _build_confusable,
    build_checklist,
    build_records,
    load_coco_annotations,
    stratified_sample_images,
)

CATS = ["dog", "cat", "car", "truck", "person", "tv",
        "bus", "bicycle", "chair", "couch", "cup", "bottle"]
NAME2ID = {n: i + 1 for i, n in enumerate(CATS)}

# (file index, [(cat, count, side)]) side: 7=小, 15=中, 30=大 (画像 100x100)
IMG_SPECS = [
    [("dog", 1, 30)],
    [("cat", 2, 15)],
    [("car", 1, 7), ("truck", 1, 7)],
    [("person", 3, 15)],
    [("tv", 1, 30)],
    [("dog", 1, 7), ("cat", 1, 7)],
    [("car", 2, 15)],
    [("person", 1, 30)],
    [("truck", 1, 15), ("tv", 1, 15)],
    [("dog", 4, 7), ("person", 2, 7)],
]


def _make_coco(path):
    images, annotations = [], []
    ann_id = 1
    for i, spec in enumerate(IMG_SPECS, start=1):
        images.append({"id": i, "file_name": f"{i:012d}.jpg", "width": 100, "height": 100})
        for cat, count, side in spec:
            for _ in range(count):
                annotations.append({
                    "id": ann_id, "image_id": i, "category_id": NAME2ID[cat],
                    "bbox": [0, 0, side, side], "area": side * side, "iscrowd": 0})
                ann_id += 1
    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": NAME2ID[n], "name": n} for n in CATS]}
    path.write_text(json.dumps(coco), encoding="utf-8")
    return str(path)


@pytest.fixture
def coco_index(tmp_path):
    return load_coco_annotations(_make_coco(tmp_path / "inst.json"))


def test_load_index(coco_index):
    assert len(coco_index["images"]) == 10
    assert coco_index["all_cats"] == sorted(CATS)
    img10 = coco_index["images"][10]
    assert img10["n_objects"] == 6
    assert img10["cat_counts"] == {"dog": 4, "person": 2}
    assert img10["min_bbox_area_ratio"] == pytest.approx(49 / 10000)


def test_stratified_sample_deterministic(coco_index):
    a = stratified_sample_images(coco_index, 6, seed=42)
    b = stratified_sample_images(coco_index, 6, seed=42)
    assert a == b
    assert len(a) == 6
    assert set(a) <= set(coco_index["images"])


def test_stratified_sample_caps_at_available(coco_index):
    allids = stratified_sample_images(coco_index, 999, seed=1)
    assert len(allids) == 10  # 物体ありの画像は全 10 枚


def test_build_checklist_prioritizes_confusable():
    conf = _build_confusable([["dog", "cat"], ["car", "truck"]])
    rng = random.Random(0)
    checklist, gt = build_checklist({"dog"}, sorted(CATS), conf, rng,
                                    min_items=5, max_items=5)
    assert len(checklist) == 5
    assert gt["dog"] == "present"
    assert "cat" in checklist and gt["cat"] == "absent"  # 混同ペアが優先混入
    assert set(checklist) == set(gt)


def test_build_records_schema_and_determinism(coco_index):
    conf = _build_confusable([["dog", "cat"], ["car", "truck"], ["couch", "chair"]])
    ids = stratified_sample_images(coco_index, 8, seed=7)
    r1 = build_records(coco_index, ids, conf, seed=7, min_items=5, max_items=8)
    r2 = build_records(coco_index, ids, conf, seed=7, min_items=5, max_items=8)
    assert r1 == r2  # 決定論

    for rec in r1:
        cl, gt, meta = rec["checklist"], rec["ground_truth"], rec["meta"]
        assert 5 <= len(cl) <= 8
        assert set(cl) == set(gt)
        cats = coco_index["images"][rec["image_id"]]["cats"]
        for c in cl:
            assert gt[c] == ("present" if c in cats else "absent")
        # gt_counts は present 項目のみ、coco の個数と一致
        for c, n in meta["gt_counts"].items():
            assert gt[c] == "present"
            assert n == coco_index["images"][rec["image_id"]]["cat_counts"][c]
