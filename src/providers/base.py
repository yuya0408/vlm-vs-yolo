"""VLM プロバイダの抽象インターフェース。

全プロバイダ(gemini / yolo / mock)はこの抽象に従う。
runner / analysis はこのインターフェースのみに依存し、プロバイダ実装を知らない。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Judgement = Literal["present", "absent", "uncertain"]


@dataclass(frozen=True)
class ItemJudgement:
    """チェックリスト 1 項目に対する判定。"""

    category: str
    judgement: Judgement
    rationale: str  # 根拠の短文(誤り分析で使用)
    count: int | None = None  # 検出インスタンス数(副軸=個数 recall 用)。出せないモデルは None


@dataclass(frozen=True)
class Usage:
    """1 リクエストのトークン使用量。コスト集計・コストガードで使用。"""

    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ProviderResponse:
    """プロバイダの 1 画像分のレスポンス。"""

    judgements: list[ItemJudgement]
    usage: Usage
    latency_sec: float
    model_version: str  # 再現性のため実際に応答したモデル ID を記録
    raw_text: str = ""  # デバッグ用の生レスポンス


class VLMProvider(ABC):
    """VLM プロバイダの抽象基底クラス。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """プロバイダ識別子(例: 'gemini-2.0-flash', 'yolov8x', 'mock')。結果 JSON に記録する。"""

    @abstractmethod
    def judge(self, image_path: Path, checklist: list[str]) -> ProviderResponse:
        """画像とチェックリストを受け取り、各項目の有無を判定する。

        実装上の要件:
        - 温度 0(再現性)
        - checklist の全項目について必ず 1 判定を返す(欠落・重複は呼び出し側で検証)
        - パース失敗時は例外を投げる(runner 側でリトライ/記録)
        """
