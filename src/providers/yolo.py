"""YOLO プロバイダ(ultralytics、ローカル推論)。

検出器の出力を「カテゴリの有無」に還元して VLMProvider 抽象に乗せる(docs/DESIGN.md §2):
- 信頼度しきい値以上で 1 個でも検出 → present、無ければ absent。
- uncertain は返さない 2 値(「VLM だけが不確実性を表現できる」点を比較の観察対象にする)。
- 検出個数を ItemJudgement.count に入れる(副軸=個数 recall 用)。
- ローカル推論なので usage は 0、latency_sec は実測。

COCO 80 クラスの名前は ultralytics と pycocotools で一致するため、カテゴリ名の対応付けは不要。
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

from .base import ItemJudgement, ProviderResponse, Usage, VLMProvider


class YoloProvider(VLMProvider):
    def __init__(self, weights: str = "yolov8x.pt", conf_threshold: float = 0.25,
                 model=None) -> None:
        self._weights = weights
        self._conf = conf_threshold
        self._model = model  # 注入可(テスト用)。None なら初回 judge で遅延ロード。

    @property
    def name(self) -> str:
        return f"yolo-{Path(self._weights).stem}"

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self._weights)
        return self._model

    def _detect_counts(self, image_path: Path) -> Counter:
        """画像を推論し、しきい値以上の検出をカテゴリ名→個数で返す。"""
        model = self._ensure_model()
        result = model.predict(source=str(image_path), conf=self._conf, verbose=False)[0]
        names = result.names  # {class_id: name}
        boxes = getattr(result, "boxes", None)
        counts: Counter = Counter()
        if boxes is not None and boxes.cls is not None:
            for cls_id in boxes.cls.tolist():
                counts[names[int(cls_id)]] += 1
        return counts

    def detect_confidences(self, image_path: Path, base_conf: float = 0.001
                           ) -> dict[str, list[float]]:
        """閾値スイープ用: 非常に低い base_conf で推論し、カテゴリ名→検出信頼度リストを返す。

        これを 1 回保存しておけば、任意のしきい値 t について
        「conf>=t の検出が 1 個でもあれば present、個数 = conf>=t の件数」を
        再推論なしの後処理で再構成できる(docs/DESIGN.md §4 の閾値感度分析)。
        """
        model = self._ensure_model()
        result = model.predict(source=str(image_path), conf=base_conf, verbose=False)[0]
        names = result.names
        boxes = getattr(result, "boxes", None)
        out: dict[str, list[float]] = {}
        if boxes is not None and boxes.cls is not None:
            cls_list = boxes.cls.tolist()
            conf_list = boxes.conf.tolist()
            for cls_id, conf in zip(cls_list, conf_list):
                out.setdefault(names[int(cls_id)], []).append(float(conf))
        return out

    def judge(self, image_path: Path, checklist: list[str]) -> ProviderResponse:
        t0 = time.time()
        counts = self._detect_counts(image_path)
        judgements = []
        for cat in checklist:
            n = counts.get(cat, 0)
            judgements.append(ItemJudgement(
                category=cat,
                judgement="present" if n > 0 else "absent",  # uncertain は返さない
                rationale=f"detections={n}@conf>={self._conf}",
                count=n,
            ))
        return ProviderResponse(
            judgements=judgements,
            usage=Usage(input_tokens=0, output_tokens=0),  # ローカル=コスト 0
            latency_sec=round(time.time() - t0, 4),
            model_version=self.name,
            raw_text="",
        )
