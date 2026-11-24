"""実行環境(Python・OS・主なライブラリの版)を結果ファイルに残すための記録。

2026-06 の推論では版を記録しておらず、後から推定した版で推論し直して一致を確かめる
必要があった(README「再現」・locks/inference.lock)。結果を作るたびに、
そのときの環境を結果と同じファイルに書いておく。
"""

from __future__ import annotations

import platform
from importlib import metadata

# 推論・判定の結果に効きうるパッケージ。入っていないものは None で残す
# (例: Gemini だけのランでは ultralytics・torch が無くてもよい)。
TRACKED_PACKAGES = (
    "ultralytics",
    "torch",
    "torchvision",
    "opencv-python",
    "numpy",
    "pillow",
    "google-genai",
)


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def runtime_environment(device: str | None = None) -> dict:
    """今のプロセスの環境を返す。device は推論に使ったデバイス(YOLO の cpu/mps/cuda:0 など)。"""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": device,
        "packages": {p: _version(p) for p in TRACKED_PACKAGES},
    }
