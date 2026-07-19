"""建設ドメイン定性デモ — 「第二象限(語彙外・ラベル無し)」の存在証明。

本デモの位置づけ(REPORT の意思決定ルール):
- COCO ベンチ(=YOLO の本拠地・ラベルあり)では、調整済み YOLO が VLM と精度互角・無料・高速。
- 対して専門ドメイン(建設)は **対象物が検出器の語彙に無く、学習/調整用ラベルも無い** 象限。
  ここでは専用 YOLO は土俵に上がれず、汎用 VLM がゼロショットで唯一動く現実解になる。
- 本スクリプトはその「象限の存在」を **公開 CC BY/PD 画像** で定性的に示す。

重要な但し書き(誇張しないための線引き):
- これは **精度比較ではなく「適用可能性」の実証**。正解ラベルが無いので F1・有意差は出さない。
- 想定する対象は配管・フランジ・チーズ等の配管施工部品。デモもそれに合わせる。
- 示すのは 2 点だけ:
  (1) COCO 学習済み YOLO は flange/tee fitting/valve 等の配管部品を **出力語彙に持たない**
      (構造的に判定不能)。配管継手は COCO に概念ごと無く、語彙ギャップが特に鮮明。
  (2) VLM はゼロショットで妥当な present/absent/uncertain + 根拠を返す。
- 「ラベルがあれば YOLO も学習できるのでは」への先回り: その通り。本象限の定義は
  まさに **そのラベルが無い/集めにくい** こと。ゆえにゼロショットの汎用が現実解になる。

実行例(ローカル。GEMINI_API_KEY が必要):
    python -m src.analysis.construction_demo \
        --images data/construction --out results/construction_demo.json \
        --report report/construction_demo_section.md --with-yolo

画像は data/construction/ に置き、data/construction/attributions.json に
{ファイル名: {"source","license","author","url"}} を用意する(CC BY は帰属表示が必須)。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..providers.factory import build_provider

# --- COCO 80 クラス(ultralytics の既定モデルが出力できる語彙)---
# YOLO(COCO学習済み)が「名指しできる」カテゴリの全集合。これに無いものは構造的に判定不能。
COCO_80 = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
}

# --- 配管施工ドメインの既定チェックリスト ---
# 想定する配管施工部品(配管・フランジ・チーズ等)に合わせる。
# COCO 内(YOLO が答えられる)と COCO 外(YOLO が構造的に答えられない)を意図的に混在させ、
# 語彙ギャップを 1 枚の表で見せる。配管継手は COCO に概念ごと存在しない=ギャップが特に鮮明。
DEFAULT_CHECKLIST = [
    # COCO 内(作業員 = 唯一の対照)
    "person",
    # COCO 外(配管施工部品 = YOLO は名指し不能)
    "pipe",                 # 配管
    "flange",               # フランジ
    "tee fitting",          # チーズ(T 字継手)
    "elbow fitting",        # エルボ(曲がり継手)
    "valve",                # バルブ
    "gasket",               # ガスケット(フランジ間パッキン)
    "bolted flange joint",  # ボルト締めフランジ接合部
    "pipe support",         # 配管サポート/ハンガー
    "pressure gauge",       # 圧力計
]

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def vocab_coverage(checklist: list[str], known_vocab: set[str] = COCO_80) -> dict:
    """チェックリストを「YOLO 語彙内 / 語彙外」に分類する(推論不要・純粋関数)。

    返り値:
        {"in_vocab": [...], "out_of_vocab": [...],
         "out_of_vocab_ratio": float, "n": int}
    語彙外の項目は、COCO 学習済み YOLO が conf によらず **絶対に present を出せない**
    = 構造的に判定不能であることを意味する(本デモの中核証拠)。
    """
    in_vocab = [c for c in checklist if c in known_vocab]
    out_vocab = [c for c in checklist if c not in known_vocab]
    n = len(checklist)
    return {
        "in_vocab": in_vocab,
        "out_of_vocab": out_vocab,
        "out_of_vocab_ratio": (len(out_vocab) / n) if n else 0.0,
        "n": n,
    }


def _load_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in _IMG_EXTS)


def _load_attributions(images_dir: Path) -> dict:
    f = images_dir / "attributions.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    return {}


def _load_flash_config(repo_root: Path) -> dict:
    """conf/models.yaml の flash エントリを読む(本ランと同じ VLM 設定を再利用)。"""
    import yaml
    models = yaml.safe_load((repo_root / "conf" / "models.yaml").read_text(encoding="utf-8"))
    return models["flash"]


def _judgements_to_dict(resp) -> list[dict]:
    return [
        {"category": j.category, "judgement": j.judgement,
         "count": j.count, "rationale": j.rationale}
        for j in resp.judgements
    ]


def run_demo(images_dir: Path, checklist: list[str], repo_root: Path,
             with_yolo: bool = False) -> dict:
    """各画像に対し VLM(と任意で YOLO)の判定を集め、語彙カバレッジと併せて返す。"""
    images = _load_images(images_dir)
    if not images:
        raise FileNotFoundError(f"画像が見つからない: {images_dir} (対応拡張子: {_IMG_EXTS})")
    attributions = _load_attributions(images_dir)
    coverage = vocab_coverage(checklist)

    vlm = build_provider(_load_flash_config(repo_root))
    yolo = None
    if with_yolo:
        import yaml
        models = yaml.safe_load((repo_root / "conf" / "models.yaml").read_text(encoding="utf-8"))
        yolo = build_provider(models["yolo"])

    per_image = []
    for img in images:
        rec = {
            "file_name": img.name,
            "attribution": attributions.get(img.name),
            "vlm": _judgements_to_dict(vlm.judge(img, checklist)),
        }
        if yolo is not None:
            # YOLO は語彙外カテゴリには構造的に present を出せない(常に absent)。
            rec["yolo"] = _judgements_to_dict(yolo.judge(img, checklist))
        per_image.append(rec)

    return {
        "checklist": checklist,
        "vocab_coverage": coverage,
        "vlm_model": vlm.name,
        "yolo_model": (yolo.name if yolo is not None else None),
        "n_images": len(images),
        "per_image": per_image,
        "disclaimer": (
            "定性デモ。正解ラベルが無いため精度・有意差は算出しない。"
            "示すのは(1)YOLO語彙ギャップ(2)VLMのゼロショット適用可能性のみ。"
        ),
    }


def render_markdown(demo: dict) -> str:
    """REPORT に貼る Markdown 節を生成する(結果から機械生成 = プレースホルダ無し)。"""
    cov = demo["vocab_coverage"]
    lines = []
    lines.append("## 付録: 建設ドメインでの適用可能性デモ(定性)\n")
    lines.append(
        "> 本ベンチ(COCO)は YOLO の本拠地で、調整済み YOLO は VLM と精度互角・無料・高速だった。"
        "では専門ドメイン(建設)は? 対象物が検出器の語彙に無く、学習/調整用ラベルも無い。"
        "ここでは専用 YOLO は土俵に上がれず、汎用 VLM がゼロショットで動く唯一の選択肢になる。"
        "**正解ラベルが無いため精度比較ではなく『適用可能性』の定性実証**(F1・有意差は出さない)。\n"
    )
    lines.append("### YOLO の語彙ギャップ(推論不要・クラス一覧から確定)\n")
    lines.append(
        f"建設チェックリスト {cov['n']} 項目のうち、COCO 80 クラスに**無い**項目は "
        f"**{len(cov['out_of_vocab'])} 件({cov['out_of_vocab_ratio']*100:.0f}%)**。"
        "これらは COCO 学習済み YOLO が conf 値によらず**絶対に present を出せない**"
        "= 構造的に判定不能。\n"
    )
    lines.append(f"- YOLO 語彙内: {', '.join(cov['in_vocab']) or '(なし)'}")
    lines.append(f"- YOLO 語彙外(判定不能): {', '.join(cov['out_of_vocab']) or '(なし)'}\n")

    has_yolo = demo.get("yolo_model") is not None
    lines.append("### 画像ごとの判定\n")
    for rec in demo["per_image"]:
        lines.append(f"#### {rec['file_name']}\n")
        attr = rec.get("attribution")
        if attr:
            lines.append(
                f"> 出典: {attr.get('source','?')} / {attr.get('author','?')} / "
                f"{attr.get('license','?')} — {attr.get('url','')}\n"
            )
        header = "| カテゴリ | 語彙 | VLM 判定 | 根拠 |"
        sep = "|---|---|---|---|"
        if has_yolo:
            header = "| カテゴリ | 語彙 | YOLO | VLM 判定 | 根拠 |"
            sep = "|---|---|---|---|---|"
        lines.append(header)
        lines.append(sep)
        vlm_map = {j["category"]: j for j in rec["vlm"]}
        yolo_map = {j["category"]: j for j in rec.get("yolo", [])}
        for cat in demo["checklist"]:
            in_v = "内" if cat in COCO_80 else "**外**"
            v = vlm_map.get(cat, {})
            vlm_cell = f"{v.get('judgement','?')}"
            if v.get("count"):
                vlm_cell += f"×{v['count']}"
            rationale = v.get("rationale", "")
            if has_yolo:
                y = yolo_map.get(cat, {})
                ycell = y.get("judgement", "—") if cat in COCO_80 else "判定不能"
                lines.append(f"| {cat} | {in_v} | {ycell} | {vlm_cell} | {rationale} |")
            else:
                lines.append(f"| {cat} | {in_v} | {vlm_cell} | {rationale} |")
        lines.append("")
    lines.append("### 読み\n")
    lines.append(
        "COCO 学習済み YOLO は語彙外カテゴリ(flange/tee fitting/valve 等の配管部品)に対し"
        "**原理的に present を返せない**(検出クラスに存在しない)。一方 VLM は同じ画像・"
        "同じチェックリストにゼロショットで判定と根拠を返す。COCO ベンチの結論"
        "「語彙内・ラベルありなら調整 YOLO で十分」と、本デモの「語彙外・ラベル無しなら VLM 一択」は"
        "**矛盾せず、一つの意思決定ルールの両輪**である。\n"
    )
    return "\n".join(lines)


def main() -> None:
    # .env から GEMINI_API_KEY を読み込む(VLM コールに必須。runner.main と同じ扱い)。
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="建設ドメイン定性デモ")
    parser.add_argument("--images", default="data/construction",
                        help="CC BY/PD の建設画像フォルダ(attributions.json 同梱)")
    parser.add_argument("--out", default="results/construction_demo.json")
    parser.add_argument("--report", default="report/construction_demo_section.md")
    parser.add_argument("--with-yolo", action="store_true",
                        help="YOLO も走らせ、語彙外で absent しか返せない様子を併記する")
    parser.add_argument("--checklist", nargs="*", default=None,
                        help="チェックリストを上書き(既定は建設の標準セット)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    images_dir = (repo_root / args.images) if not Path(args.images).is_absolute() else Path(args.images)
    checklist = args.checklist or DEFAULT_CHECKLIST

    demo = run_demo(images_dir, checklist, repo_root, with_yolo=args.with_yolo)

    out_path = (repo_root / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(demo, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = (repo_root / args.report) if not Path(args.report).is_absolute() else Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(demo), encoding="utf-8")

    cov = demo["vocab_coverage"]
    print(f"画像 {demo['n_images']} 枚 / VLM={demo['vlm_model']} "
          f"/ YOLO={demo['yolo_model']}")
    print(f"語彙外(YOLO判定不能): {len(cov['out_of_vocab'])}/{cov['n']} "
          f"= {cov['out_of_vocab_ratio']*100:.0f}%  -> {cov['out_of_vocab']}")
    print(f"結果: {out_path}")
    print(f"レポート節: {report_path}")


if __name__ == "__main__":
    main()
