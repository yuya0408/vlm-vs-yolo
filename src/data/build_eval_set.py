"""COCO val2017 から評価セット(eval_set.json)を構築する。

実行例:
    python -m src.data.build_eval_set --config conf/eval.yaml --out data/eval_set.json

評価セットの設計(docs/DESIGN.md §3):
- N 枚を層化サンプリング(物体数・最小 bbox 面積比で層化)
- 各画像に positive(写っている)+ negative(写っていない)カテゴリを 5〜10 項目
- negative には混同しやすいカテゴリ(conf/eval.yaml の confusable_pairs)を優先的に混入
- 難易度メタデータ(min_bbox_area_ratio, has_crowd, n_objects)+ gt_counts(副軸=個数 recall 用)を付与
- シード固定で再現可能。出力 JSON はリポジトリ管理(DVC は使わない)

eval_set.json のスキーマ(1 レコード = 1 画像):
{
  "image_id": 139,
  "file_name": "000000000139.jpg",
  "checklist": ["person", "cat", "tv"],
  "ground_truth": {"person": "present", "cat": "absent", "tv": "present"},
  "meta": {"n_objects": 4, "min_bbox_area_ratio": 0.01, "has_crowd": false,
           "gt_counts": {"person": 2, "tv": 1}}
}
"""

from __future__ import annotations

import argparse
import json
import random
from bisect import bisect_left
from collections import Counter, defaultdict

import yaml


def load_coco_annotations(ann_path: str) -> dict:
    """instances_val2017.json をロードし、image_id → カテゴリ集合等の索引を作る。"""
    from pycocotools.coco import COCO

    coco = COCO(ann_path)
    catid2name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    all_cats = sorted(catid2name.values())

    images: dict[int, dict] = {}
    for img_id in coco.getImgIds():
        info = coco.loadImgs([img_id])[0]
        img_area = float(info["width"]) * float(info["height"]) or 1.0
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        cat_counts: Counter = Counter()
        min_ratio = 1.0
        has_crowd = False
        for a in anns:
            cat_counts[catid2name[a["category_id"]]] += 1
            x, y, w, h = a["bbox"]
            ratio = (float(w) * float(h)) / img_area
            min_ratio = min(min_ratio, ratio)
            if a.get("iscrowd", 0):
                has_crowd = True
        images[img_id] = {
            "file_name": info["file_name"],
            "cats": set(cat_counts),
            "cat_counts": dict(cat_counts),
            "n_objects": len(anns),
            "min_bbox_area_ratio": round(min_ratio, 6) if anns else 0.0,
            "has_crowd": has_crowd,
        }
    return {"images": images, "all_cats": all_cats}


def _stratum(meta: dict, nobj_bounds: list[int], area_bounds: list[float]) -> tuple[int, int]:
    nobj_bin = bisect_left(nobj_bounds, meta["n_objects"])
    area_bin = bisect_left(area_bounds, meta["min_bbox_area_ratio"])
    return nobj_bin, area_bin


def stratified_sample_images(coco_index: dict, n: int, seed: int,
                             strata: dict | None = None) -> list[int]:
    """物体数・最小 bbox 面積比の層で層化サンプリングし image_id のリストを返す。

    各層から比例配分で抽出(端数は大きい層へ)。グローバル乱数は使わず random.Random(seed)。
    """
    strata = strata or {"n_objects": [1, 3, 8], "min_bbox_area_ratio": [0.005, 0.05]}
    nobj_bounds = strata["n_objects"]
    area_bounds = strata["min_bbox_area_ratio"]
    rng = random.Random(seed)

    # 物体ゼロの画像は positive を作れないので除外
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for img_id, meta in coco_index["images"].items():
        if meta["n_objects"] == 0 or not meta["cats"]:
            continue
        buckets[_stratum(meta, nobj_bounds, area_bounds)].append(img_id)

    total = sum(len(v) for v in buckets.values())
    n = min(n, total)
    if total == 0:
        return []

    # 比例配分(floor)→ 残りを層の大きい順に 1 枚ずつ配る
    alloc: dict[tuple, int] = {}
    for k, ids in buckets.items():
        alloc[k] = min(len(ids), int(n * len(ids) / total))
    remaining = n - sum(alloc.values())
    for k in sorted(buckets, key=lambda k: len(buckets[k]), reverse=True):
        if remaining <= 0:
            break
        if alloc[k] < len(buckets[k]):
            alloc[k] += 1
            remaining -= 1

    chosen: list[int] = []
    for k, ids in buckets.items():
        pick = rng.sample(ids, alloc[k])
        chosen.extend(pick)
    rng.shuffle(chosen)
    return chosen


def build_checklist(image_cats: set[str], all_cats: list[str],
                    confusable: dict[str, list[str]], rng: random.Random,
                    min_items: int = 5, max_items: int = 10
                    ) -> tuple[list[str], dict[str, str]]:
    """1 画像の checklist と ground_truth を作る。

    positive は画像内カテゴリから、negative は画像外カテゴリから。negative には
    positive と混同しやすいカテゴリ(confusable)を優先混入。合計 5〜10 項目、順序はシャッフル。
    """
    positives = sorted(image_cats)
    negatives_pool = [c for c in all_cats if c not in image_cats]

    target = rng.randint(min_items, max_items)
    n_pos = min(len(positives), max(1, target // 2))
    chosen_pos = rng.sample(positives, n_pos)
    n_neg = min(len(negatives_pool), target - n_pos)

    # 混同しやすい negative を優先
    prioritized: list[str] = []
    for p in chosen_pos:
        for partner in confusable.get(p, []):
            if partner in negatives_pool and partner not in prioritized:
                prioritized.append(partner)
    rng.shuffle(prioritized)

    chosen_neg = prioritized[:n_neg]
    if len(chosen_neg) < n_neg:
        rest = [c for c in negatives_pool if c not in chosen_neg]
        chosen_neg += rng.sample(rest, n_neg - len(chosen_neg))

    checklist = chosen_pos + chosen_neg
    rng.shuffle(checklist)
    ground_truth = {c: ("present" if c in image_cats else "absent") for c in checklist}
    return checklist, ground_truth


def _build_confusable(pairs: list[list[str]]) -> dict[str, list[str]]:
    conf: dict[str, list[str]] = defaultdict(list)
    for a, b in pairs:
        conf[a].append(b)
        conf[b].append(a)
    return dict(conf)


def build_records(coco_index: dict, image_ids: list[int], confusable: dict[str, list[str]],
                  seed: int, min_items: int, max_items: int) -> list[dict]:
    rng = random.Random(seed)
    all_cats = coco_index["all_cats"]
    records = []
    for img_id in image_ids:
        meta = coco_index["images"][img_id]
        checklist, gt = build_checklist(meta["cats"], all_cats, confusable, rng,
                                        min_items, max_items)
        gt_counts = {c: meta["cat_counts"][c] for c in checklist if gt[c] == "present"}
        records.append({
            "image_id": img_id,
            "file_name": meta["file_name"],
            "checklist": checklist,
            "ground_truth": gt,
            "meta": {
                "n_objects": meta["n_objects"],
                "min_bbox_area_ratio": meta["min_bbox_area_ratio"],
                "has_crowd": meta["has_crowd"],
                "gt_counts": gt_counts,
            },
        })
    return records


def _summary(records: list[dict]) -> str:
    n = len(records)
    pos = sum(sum(1 for v in r["ground_truth"].values() if v == "present") for r in records)
    items = sum(len(r["checklist"]) for r in records)
    crowd = sum(1 for r in records if r["meta"]["has_crowd"])
    return (f"画像 {n} 枚 / 項目 {items}(positive {pos} = {pos / items:.1%}) / "
            f"crowd 画像 {crowd}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/eval.yaml")
    parser.add_argument("--out", default="data/eval_set.json")
    parser.add_argument("--smoke-out", default="data/eval_set_smoke.json")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    coco_index = load_coco_annotations(cfg["paths"]["coco_annotations"])
    confusable = _build_confusable(cfg.get("confusable_pairs", []))

    image_ids = stratified_sample_images(coco_index, cfg["n_images"], cfg["seed"],
                                         cfg.get("strata"))
    records = build_records(coco_index, image_ids, confusable, cfg["seed"],
                            cfg["checklist"]["min_items"], cfg["checklist"]["max_items"])

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    smoke = records[: cfg["smoke_n"]]
    with open(args.smoke_out, "w", encoding="utf-8") as f:
        json.dump(smoke, f, ensure_ascii=False, indent=2)

    print(f"[full]  {args.out}: {_summary(records)}")
    print(f"[smoke] {args.smoke_out}: {_summary(smoke)}")


if __name__ == "__main__":
    main()
