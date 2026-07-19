"""プロバイダのファクトリ。設定(conf/models.yaml のエントリ)から実装を生成する。"""

from __future__ import annotations

from pathlib import Path

from .base import VLMProvider

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_prompt(name: str) -> str:
    path = _REPO_ROOT / "conf" / "prompts" / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def build_provider(config: dict) -> VLMProvider:
    """設定 dict からプロバイダを生成する。

    config 例(conf/models.yaml の 1 エントリ):
        {"provider": "gemini", "model": "gemini-2.0-flash", "prompt": "concise"}
        {"provider": "yolo", "weights": "yolov8x.pt", "conf_threshold": 0.25}
        {"provider": "mock"}
    """
    provider = config.get("provider")

    if provider == "mock":
        from .mock import MockProvider
        return MockProvider()

    if provider == "gemini":
        from .gemini import GeminiProvider
        template = _read_prompt(config.get("prompt", "concise"))
        return GeminiProvider(model=config["model"], prompt_template=template,
                              timeout_sec=float(config.get("timeout_sec", 90.0)))

    if provider == "yolo":
        from .yolo import YoloProvider  # M2 で追加(ultralytics 依存)
        return YoloProvider(weights=config["weights"],
                            conf_threshold=float(config.get("conf_threshold", 0.25)))

    raise ValueError(f"未知の provider: {provider!r}")
