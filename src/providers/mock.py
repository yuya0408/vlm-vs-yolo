"""mock プロバイダ: API キー・課金なしで E2E を疎通確認するためのダミー実装。

設計方針:
- 画像パスとカテゴリ名のハッシュから決定論的に判定を返す(同入力 → 同出力)
- 一定割合で uncertain を混ぜる(uncertain の扱いを検証できるように)
- usage / latency もダミー値を返し、コスト集計のコードパスを通す
"""
from __future__ import annotations

import hashlib

from pathlib import Path

from .base import ItemJudgement, ProviderResponse, Usage, VLMProvider

def _det_int(key: str, mod: int = 10) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()  # bytes 32個
    return int.from_bytes(digest[:4], "big") % mod      # 先頭4バイト→整数→剰余

class MockProvider(VLMProvider):
    @property
    def name(self) -> str:
        return "mock"

    def judge(self, image_path: Path, checklist: list[str]) -> ProviderResponse:
        judgements = []
        for cat in checklist:
            key = f"{image_path.name}:{cat}"
            n = _det_int(key)

            if n <= 5:
                judgement = "present"
            elif n <= 8:
                judgement = "absent"
            else:
                judgement = "uncertain"
            
            judgements.append(ItemJudgement(category=cat, judgement=judgement, rationale="mock"))
            
        return ProviderResponse(
            judgements=judgements,
            usage=Usage(input_tokens=1000, output_tokens=200), 
            latency_sec=0.1, 
            model_version="mock-model-v1",
            raw_text="mock raw response"
        )