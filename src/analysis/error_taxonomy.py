"""M4 仕上げ: 誤り taxonomy + 代表誤答例の抽出。

各誤答を以下のバケットに分類する(strict 基準):
- missed_present       : gt=present, pred=absent(見逃し / FN)
- fp_confusable        : gt=absent, pred=present で、その項目が同一画像内の present カテゴリと
                         混同ペア(conf/eval.yaml: confusable_pairs)→ 過剰主張(混同)
- fp_other             : gt=absent, pred=present(混同ペア以外の偽陽性)
- uncertain_on_present : pred=uncertain かつ gt=present(出すべきを出せた難ケース)
- uncertain_on_absent  : pred=uncertain かつ gt=absent

混同ペア FP は「チェックリスト事前情報による過剰主張」の代理指標。YOLO はチェックリストを
見ないため原理的にこのバイアスを持たず、VLM との対比が鮮明になる。

実行例:
    python -m src.analysis.error_taxonomy \
        --yolo results/yolo-yolo26x-289d6c0e.json \
        --vlm  results/gemini-gemini-3.5-flash-289d6c0e.json \
        --eval-set data/eval_set.json --eval-config conf/eval.yaml \
        --out results/error_taxonomy.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .metrics import _pred_lookup

BUCKETS = ("missed_present", "fp_confusable", "fp_other",
           "uncertain_on_present", "uncertain_on_absent")


def _confusable_map(eval_config: dict) -> dict[str, set[str]]:
    """confusable_pairs を双方向の {cat: {相方,...}} に展開する。"""
    out: dict[str, set[str]] = {}
    for a, b in eval_config.get("confusable_pairs", []):
        out.setdefault(a, set()).add(b)
        out.setdefault(b, set()).add(a)
    return out


def _classify(gt: str, pred: str, cat: str, present_cats: set[str],
              confusable: dict[str, set[str]]) -> str | None:
    """1 判定をバケットに分類。正解なら None。"""
    if pred == "uncertain":
        return "uncertain_on_present" if gt == "present" else "uncertain_on_absent"
    if pred == gt:
        return None
    if gt == "present" and pred == "absent":
        return "missed_present"
    if gt == "absent" and pred == "present":
        # 同一画像の present カテゴリに混同相方がいれば「混同による過剰主張」
        if confusable.get(cat, set()) & present_cats:
            return "fp_confusable"
        return "fp_other"
    return "fp_other"


def taxonomy(result: dict, eval_set: list[dict], confusable: dict[str, set[str]],
             max_examples: int = 5) -> dict:
    preds = _pred_lookup(result["records"], eval_set)
    eval_by_id = {e["image_id"]: e for e in eval_set}
    rationale_by = {(r["image_id"], j["category"]): j.get("rationale", "")
                    for r in result["records"] for j in r["judgements"]}

    counts = {b: 0 for b in BUCKETS}
    examples: dict[str, list[dict]] = {b: [] for b in BUCKETS}
    for image_id, pmap in preds.items():
        e = eval_by_id[image_id]
        gt = e["ground_truth"]
        present_cats = {c for c, g in gt.items() if g == "present"}
        meta = e.get("meta", {})
        for cat, pred in pmap.items():
            bucket = _classify(gt[cat], pred, cat, present_cats, confusable)
            if bucket is None:
                continue
            counts[bucket] += 1
            if len(examples[bucket]) < max_examples:
                examples[bucket].append({
                    "image_id": image_id,
                    "file_name": e.get("file_name"),
                    "category": cat,
                    "gt": gt[cat],
                    "pred": pred,
                    "min_bbox_area_ratio": meta.get("min_bbox_area_ratio"),
                    "rationale": rationale_by.get((image_id, cat), ""),
                })

    total_errors = sum(counts.values())
    return {"run_id": result.get("run_id"), "model_version": result.get("model_version"),
            "total_errors": total_errors, "counts": counts, "examples": examples}


def build(yolo_path: str, vlm_path: str, eval_set_path: str,
          eval_config_path: str = "conf/eval.yaml", max_examples: int = 5) -> dict:
    yolo = json.loads(Path(yolo_path).read_text(encoding="utf-8"))
    vlm = json.loads(Path(vlm_path).read_text(encoding="utf-8"))
    eval_set = json.loads(Path(eval_set_path).read_text(encoding="utf-8"))
    eval_config = yaml.safe_load(Path(eval_config_path).read_text(encoding="utf-8"))
    confusable = _confusable_map(eval_config)
    return {
        "yolo": taxonomy(yolo, eval_set, confusable, max_examples),
        "vlm": taxonomy(vlm, eval_set, confusable, max_examples),
    }


def to_markdown(tax: dict) -> str:
    y, v = tax["yolo"], tax["vlm"]
    lines = ["## 誤り taxonomy(strict)", "",
             "| バケット | YOLO | VLM |", "|---|---|---|"]
    for b in BUCKETS:
        lines.append(f"| {b} | {y['counts'][b]} | {v['counts'][b]} |")
    lines.append(f"| **合計誤答** | **{y['total_errors']}** | **{v['total_errors']}** |")
    lines += ["", "### 代表例", ""]
    for model_key, m in (("YOLO", y), ("VLM", v)):
        lines.append(f"**{model_key}**")
        for b in BUCKETS:
            for ex in m["examples"][b][:2]:
                area = ex.get("min_bbox_area_ratio")
                area_s = f", area={area:.4f}" if isinstance(area, (int, float)) else ""
                lines.append(f"- [{b}] img{ex['image_id']} `{ex['category']}` "
                             f"(gt={ex['gt']}→pred={ex['pred']}{area_s}) {ex['rationale'][:40]}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo", required=True)
    parser.add_argument("--vlm", required=True)
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--eval-config", default="conf/eval.yaml")
    parser.add_argument("--out", default="results/error_taxonomy.json")
    parser.add_argument("--max-examples", type=int, default=5)
    args = parser.parse_args()

    tax = build(args.yolo, args.vlm, args.eval_set, args.eval_config, args.max_examples)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(tax, ensure_ascii=False, indent=2), encoding="utf-8")
    print(to_markdown(tax))
    print(f"\n誤り taxonomy を保存: {args.out}")


if __name__ == "__main__":
    main()
