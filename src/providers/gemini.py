"""Gemini プロバイダ (google-genai SDK、API キー認証)。

設計上の要点(docs/DESIGN.md §2/§4):
- 温度 0(再現性)。出力は JSON 強制(response_mime_type="application/json")。
- プロンプトへの checklist 差し込みは str.replace(JSON 例の波括弧が format を壊すため)。
- レスポンスは信頼せず厳格に検証: checklist 全項目が一度ずつ・judgement は 3 値・count は規約どおり。
  モデルの順序も信用せず checklist 順に並べ直す。
- count 規約: present は概数(>=1。null は present の含意から 1 に補正)、absent/uncertain は None。
- usage_metadata からトークンを記録(金額換算は gate.py + pricing.yaml)。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .base import ItemJudgement, ProviderResponse, Usage, VLMProvider

_VALID = {"present", "absent", "uncertain"}


class GeminiProvider(VLMProvider):
    """Gemini による判定。モデル名は conf/models.yaml から渡す。

    generate_fn を渡すと API を呼ばずにそれを使う(テスト用)。
    シグネチャ: generate_fn(model, image_path, prompt) -> (text, usage_dict, model_version)
    """

    def __init__(self, model: str, prompt_template: str, generate_fn=None,
                 timeout_sec: float = 90.0) -> None:
        self._model = model
        self._prompt_template = prompt_template
        self._generate_fn = generate_fn
        self._timeout_sec = timeout_sec  # 応答ハング対策。超過で例外→runner がリトライ
        self._client = None  # 遅延初期化

    @property
    def name(self) -> str:
        return self._model

    # --- API 呼び出し(差し込み可能) ---
    def _ensure_client(self):
        if self._client is None:
            from google import genai
            from google.genai import types
            # timeout はミリ秒。応答が返らない接続を無限待機しないための保険。
            self._client = genai.Client(
                api_key=os.environ["GEMINI_API_KEY"],
                http_options=types.HttpOptions(timeout=int(self._timeout_sec * 1000)),
            )
        return self._client

    def _raw_generate(self, model: str, image_path: Path, prompt: str):
        from google.genai import types

        client = self._ensure_client()
        data = Path(image_path).read_bytes()
        mime = "image/png" if str(image_path).lower().endswith(".png") else "image/jpeg"
        resp = client.models.generate_content(
            model=model,
            contents=[types.Part.from_bytes(data=data, mime_type=mime), prompt],
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        um = getattr(resp, "usage_metadata", None)
        usage = {
            "input_tokens": int(getattr(um, "prompt_token_count", 0) or 0),
            "output_tokens": int(getattr(um, "candidates_token_count", 0) or 0),
        }
        model_version = getattr(resp, "model_version", None) or model
        return resp.text, usage, model_version

    # --- パース・検証 ---
    @staticmethod
    def _strip_fences(text: str) -> str:
        t = text.strip()
        if t.startswith("```"):
            t = t.split("\n", 1)[1] if "\n" in t else t
            if t.rstrip().endswith("```"):
                t = t.rstrip()[:-3]
        return t.strip()

    @staticmethod
    def _norm_count(judgement: str, raw) -> int | None:
        if judgement != "present":
            return None  # absent/uncertain は必ず None
        if raw is None:
            return 1  # present の含意から最低 1
        try:
            n = int(round(float(raw)))
        except (TypeError, ValueError):
            return 1
        return max(1, n)

    def _parse_and_validate(self, text: str, checklist: list[str]) -> list[ItemJudgement]:
        try:
            data = json.loads(self._strip_fences(text))
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON パース失敗: {e}") from e
        if not isinstance(data, list):
            raise ValueError("レスポンスが JSON 配列でない")

        by_cat: dict[str, ItemJudgement] = {}
        for item in data:
            if not isinstance(item, dict):
                raise ValueError(f"配列要素が object でない: {item!r}")
            cat = item.get("category")
            judgement = item.get("judgement")
            if judgement not in _VALID:
                raise ValueError(f"{cat!r}: 不正な judgement {judgement!r}")
            if cat in by_cat:
                raise ValueError(f"カテゴリ {cat!r} が重複")
            by_cat[cat] = ItemJudgement(
                category=cat,
                judgement=judgement,
                rationale=str(item.get("rationale", "")),
                count=self._norm_count(judgement, item.get("count")),
            )

        missing = set(checklist) - set(by_cat)
        if missing:
            raise ValueError(f"判定が欠落: {sorted(missing)}")
        extra = set(by_cat) - set(checklist)
        if extra:
            raise ValueError(f"checklist 外のカテゴリ: {sorted(extra)}")
        # モデルの順序を信用せず checklist 順に並べ直す
        return [by_cat[c] for c in checklist]

    def judge(self, image_path: Path, checklist: list[str]) -> ProviderResponse:
        prompt = self._prompt_template.replace("{checklist}", ", ".join(checklist))
        gen = self._generate_fn or self._raw_generate
        t0 = time.time()
        text, usage, model_version = gen(self._model, image_path, prompt)
        latency = round(time.time() - t0, 4)
        judgements = self._parse_and_validate(text, checklist)
        return ProviderResponse(
            judgements=judgements,
            usage=Usage(input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0)),
            latency_sec=latency,
            model_version=model_version,
            raw_text=text,
        )
