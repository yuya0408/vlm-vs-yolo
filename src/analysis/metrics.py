"""評価指標の計算。

3 値分類(present / absent / uncertain)を 2 通りで集計する(docs/DESIGN.md §4):
- strict: uncertain を誤りとして扱う(present の取りこぼし = FN に算入)
- excl:   uncertain の項目を除外して計算し、カバレッジ(= 判定確定率)を併記

positive クラス = present。per-category / macro / micro の P・R・F1 を出す。
YOLO は uncertain を返さない 2 値なので、strict と excl は一致し coverage = 1.0 になる。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


VALID_GT = {"present", "absent"}
VALID_PRED = {"present", "absent", "uncertain"}


@dataclass(frozen=True)
class CategoryMetrics:
    category: str
    precision: float
    recall: float
    f1: float
    support: int  # そのカテゴリの GT=present 件数(positive クラスの support)


@dataclass(frozen=True)
class MetricsReport:
    mode: str                      # "strict" | "excl"
    per_category: list[CategoryMetrics]
    macro_f1: float
    micro_f1: float
    coverage: float                # excl モード時の判定確定率(strict では 1.0)
    item_accuracy: float           # 項目単位の正解率((pred==gt)の割合)。strictはuncertainを誤り算入、exclは除外


def _pred_lookup(results: list[dict], eval_set: list[dict]) -> dict[int, dict[str, str]]:
    """image_id → {category: judgement} を作りつつ、結果の整合を検証する。"""
    eval_by_id = {e["image_id"]: e for e in eval_set}

    if len(eval_by_id) != len(eval_set):
        raise ValueError("eval_set の image_id に重複がある")
    if {r["image_id"] for r in results} != set(eval_by_id):
        raise ValueError("eval_set と results の image_id 集合が一致しない")
    if len({r["image_id"] for r in results}) != len(results):
        raise ValueError("results の image_id に重複がある")

    lookup: dict[int, dict[str, str]] = {}
    for r in results:
        image_id = r["image_id"]
        checklist = eval_by_id[image_id]["checklist"]
        preds: dict[str, str] = {}
        for j in r["judgements"]:
            cat, judgement = j["category"], j["judgement"]
            if judgement not in VALID_PRED:
                raise ValueError(f"image {image_id}: 不正な judgement {judgement!r}")
            if cat in preds:
                raise ValueError(f"image {image_id}: カテゴリ {cat!r} の判定が重複")
            preds[cat] = judgement
        missing = set(checklist) - set(preds)
        if missing:
            raise ValueError(f"image {image_id}: 判定が欠落 {sorted(missing)}")
        unknown = set(preds) - set(checklist)
        if unknown:
            raise ValueError(f"image {image_id}: checklist 外のカテゴリ {sorted(unknown)}")
        lookup[image_id] = preds
    return lookup


def confusion_counts(results: list[dict], eval_set: list[dict]) -> dict[str, dict[str, int]]:
    """image_id で突合し、カテゴリ別に (gt, pred) ペアの件数を集計する。

    返り値: {category: {"pp": n, "pa": n, "pu": n, "ap": n, "aa": n, "au": n}}
      キーの 1 文字目 = gt(p=present / a=absent)、2 文字目 = pred(p/a/u=uncertain)。
    """
    lookup = _pred_lookup(results, eval_set)
    eval_by_id = {e["image_id"]: e for e in eval_set}

    counts: dict[str, dict[str, int]] = {}
    code = {"present": "p", "absent": "a", "uncertain": "u"}
    for image_id, preds in lookup.items():
        gt = eval_by_id[image_id]["ground_truth"]
        for cat, pred in preds.items():
            if cat not in gt:
                raise ValueError(f"image {image_id}: ground_truth に {cat!r} が無い")
            g = gt[cat]
            if g not in VALID_GT:
                raise ValueError(f"image {image_id}: 不正な ground_truth {g!r}")
            key = code[g] + code[pred]
            cell = counts.setdefault(cat, {k: 0 for k in ("pp", "pa", "pu", "ap", "aa", "au")})
            cell[key] += 1
    return counts


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def compute_metrics(results: list[dict], eval_set: list[dict], mode: str) -> MetricsReport:
    """指標を計算する。mode は "strict" か "excl"。"""
    if mode not in ("strict", "excl"):
        raise ValueError(f"mode は 'strict' か 'excl': {mode!r}")

    counts = confusion_counts(results, eval_set)

    per_category: list[CategoryMetrics] = []
    micro_tp = micro_fp = micro_fn = 0
    total_items = 0
    uncertain_items = 0
    correct_items = 0

    for cat in sorted(counts):
        c = counts[cat]
        # tp/fp/fn は positive クラス(present)について定義する。
        tp = c["pp"]
        fp = c["ap"]
        if mode == "strict":
            # uncertain は誤り扱い: present の取りこぼし(pu)は FN に算入。
            fn = c["pa"] + c["pu"]
        else:  # excl: uncertain の項目は集計から除外
            fn = c["pa"]

        precision, recall, f1 = _prf(tp, fp, fn)
        support = c["pp"] + c["pa"] + c["pu"]  # GT=present の総数
        per_category.append(CategoryMetrics(cat, precision, recall, f1, support))

        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

        cat_total = sum(c.values())
        total_items += cat_total
        uncertain_items += c["pu"] + c["au"]
        correct_items += c["pp"] + c["aa"]

    # macro は support>0(GT に present が 1 度でも出る)カテゴリで平均する。
    scored = [m.f1 for m in per_category if m.support > 0]
    macro_f1 = sum(scored) / len(scored) if scored else 0.0
    _, _, micro_f1 = _prf(micro_tp, micro_fp, micro_fn)

    if mode == "strict":
        coverage = 1.0
        item_accuracy = correct_items / total_items if total_items else 0.0
    else:
        scored_items = total_items - uncertain_items
        coverage = scored_items / total_items if total_items else 1.0
        item_accuracy = correct_items / scored_items if scored_items else 0.0

    return MetricsReport(
        mode=mode,
        per_category=per_category,
        macro_f1=macro_f1,
        micro_f1=micro_f1,
        coverage=coverage,
        item_accuracy=item_accuracy,
    )


def count_recall(results: list[dict], eval_set: list[dict]) -> dict:
    """副軸: 個数 recall。存在するインスタンスのうち何個を検出できたか(docs/DESIGN.md §2/§4)。

    recall = Σ min(予測個数, GT個数) / Σ GT個数。フル IoU 照合はしない近似で、
    「密シーン・小物体での取りこぼし」を捉える。min で過剰検出を頭打ちにする。
    予測に count が無いモデル(present/absent のみ)では計算できず ValueError。
    """
    eval_by_id = {e["image_id"]: e for e in eval_set}

    pred_counts: dict[tuple[int, str], int | None] = {}
    for r in results:
        for j in r["judgements"]:
            pred_counts[(r["image_id"], j["category"])] = j.get("count")

    # 全予測が count 無し = そのモデルは個数を出さない(mock 等)→ 副軸は計算不能。
    # 一部だけ None(present のみ個数を返す VLM の absent/uncertain)は 0 個検出として扱う。
    if pred_counts and all(v is None for v in pred_counts.values()):
        raise ValueError("全予測に count が無い(このモデルは個数を出さないため副軸 recall は計算できない)")

    found_by_cat: dict[str, int] = defaultdict(int)
    gt_by_cat: dict[str, int] = defaultdict(int)
    total_found = total_gt = 0
    for r in results:
        gt_counts = eval_by_id[r["image_id"]].get("meta", {}).get("gt_counts", {})
        for cat, gt_n in gt_counts.items():
            pred_n = pred_counts.get((r["image_id"], cat))
            pred_n = 0 if pred_n is None else pred_n  # null(absent/uncertain)= 0 個検出
            found = min(pred_n, gt_n)
            found_by_cat[cat] += found
            gt_by_cat[cat] += gt_n
            total_found += found
            total_gt += gt_n

    per_category = {c: found_by_cat[c] / gt_by_cat[c] for c in gt_by_cat if gt_by_cat[c] > 0}
    overall = total_found / total_gt if total_gt else 0.0
    return {"recall": overall, "per_category": per_category, "n_gt_instances": total_gt}
