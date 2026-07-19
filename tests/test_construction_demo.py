"""construction_demo の純粋関数(語彙カバレッジ・Markdown生成)のテスト。

VLM/YOLO の実推論は伴わない(API・重み不要)。語彙ギャップ判定とレポート生成のロジックを検証。
"""

from src.analysis.construction_demo import (
    COCO_80,
    DEFAULT_CHECKLIST,
    render_markdown,
    vocab_coverage,
)


def test_vocab_coverage_splits_in_and_out():
    checklist = ["person", "truck", "scaffolding", "rebar"]
    cov = vocab_coverage(checklist)
    assert cov["in_vocab"] == ["person", "truck"]
    assert cov["out_of_vocab"] == ["scaffolding", "rebar"]
    assert cov["n"] == 4
    assert abs(cov["out_of_vocab_ratio"] - 0.5) < 1e-9


def test_default_checklist_mixes_vocab():
    cov = vocab_coverage(DEFAULT_CHECKLIST)
    # COCO 内が必ず含まれる(YOLO が答えられる対照)
    assert "person" in cov["in_vocab"]
    # 配管施工部品が語彙外に分類される(YOLO 判定不能 = デモの核)
    for item in ["pipe", "flange", "tee fitting", "elbow fitting", "valve"]:
        assert item in cov["out_of_vocab"]
    # 配管デモは対照(person)以外ほぼ全て語彙外
    assert len(cov["out_of_vocab"]) >= len(cov["in_vocab"])


def test_coco_known_vocab_sanity():
    assert "person" in COCO_80
    assert "truck" in COCO_80
    # 配管施工部品は COCO に概念ごと無い
    assert "pipe" not in COCO_80
    assert "flange" not in COCO_80
    assert "tee fitting" not in COCO_80
    assert "valve" not in COCO_80


def test_empty_checklist():
    cov = vocab_coverage([])
    assert cov["n"] == 0
    assert cov["out_of_vocab_ratio"] == 0.0


def test_render_markdown_marks_out_of_vocab_and_attribution():
    demo = {
        "checklist": ["person", "scaffolding"],
        "vocab_coverage": vocab_coverage(["person", "scaffolding"]),
        "vlm_model": "gemini-x",
        "yolo_model": "yolo-x",
        "n_images": 1,
        "per_image": [{
            "file_name": "img01.jpg",
            "attribution": {"source": "Wikimedia", "author": "A",
                            "license": "CC BY 4.0", "url": "http://x"},
            "vlm": [
                {"category": "person", "judgement": "present", "count": 2, "rationale": "two workers"},
                {"category": "scaffolding", "judgement": "present", "count": None, "rationale": "metal frame"},
            ],
            "yolo": [
                {"category": "person", "judgement": "present", "count": 2, "rationale": "det=2"},
                {"category": "scaffolding", "judgement": "absent", "count": 0, "rationale": "det=0"},
            ],
        }],
    }
    md = render_markdown(demo)
    # 語彙外項目には YOLO 列に「判定不能」が出る(構造的に present 不可)
    assert "判定不能" in md
    # 帰属が載る(CC BY 必須)
    assert "Wikimedia" in md and "CC BY 4.0" in md
    # VLM の present×count 表示
    assert "present×2" in md
