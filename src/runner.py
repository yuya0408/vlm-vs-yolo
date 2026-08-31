"""評価ランナー: 評価セット × プロバイダで判定を実行し、結果を保存する。

実行例:
    python -m src.runner --eval-set data/eval_set_smoke.json --model flash

責務:
- キャッシュ: (provider名, model_version, prompt名, image_id, checklist) のハッシュをキーに
  レスポンスを cache/ に保存。ヒット時は API を呼ばない(再現性 + コスト削減)
- 再開可能: run_id は (provider, model, 評価セットhash) から決定的に決まり、再実行すれば
  未処理の画像のみ実行する
- 結果の確定保存: results/{run_id}.json に確定保存し、モデル版・評価セットの sha256・
  実行日時を記録する(MLflow/DVC は使わず、JSON + シード固定 + キャッシュで再現性を担保)
- コスト記録: Usage を結果に含め、金額換算は src/gate.py(単価表 conf/pricing.yaml)で行う

結果ファイル(results/{run_id}.json):
{
  "run_id": "...", "provider": "mock", "model_version": "mock-model-v1", "prompt": "concise",
  "eval_set_path": "...", "eval_set_sha256": "...", "created_at": "...",
  "records": [
    {"image_id": 139,
     "judgements": [{"category": "person", "judgement": "present", "rationale": "..."}],
     "usage": {"input_tokens": 1234, "output_tokens": 56},
     "latency_sec": 1.2, "cached": false}
  ]
}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path

import yaml

from .providers.base import ProviderResponse
from .providers.factory import build_provider


def cache_key(provider_name: str, model_version: str, prompt_name: str,
              image_id: int, checklist: list[str]) -> str:
    """キャッシュキーを生成する。checklist はソートして正規化し sha256 hex を返す。"""
    payload = json.dumps(
        {"provider": provider_name, "model": model_version, "prompt": prompt_name,
         "image_id": image_id, "checklist": sorted(checklist)},
        sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _response_to_cache(resp: ProviderResponse) -> dict:
    return {
        "judgements": [{"category": j.category, "judgement": j.judgement,
                        "rationale": j.rationale, "count": j.count} for j in resp.judgements],
        "usage": {"input_tokens": resp.usage.input_tokens,
                  "output_tokens": resp.usage.output_tokens},
        "latency_sec": resp.latency_sec,
        "model_version": resp.model_version,
        "raw_text": resp.raw_text,
    }


def _judge_with_retry(provider, image_path: Path, checklist: list[str], max_retries: int = 3):
    """指数バックオフで最大 max_retries 回リトライ。"""
    last_exc = None
    for attempt in range(max_retries):
        try:
            return provider.judge(image_path, checklist)
        except Exception as exc:  # noqa: BLE001 — runner が記録のため握る
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise last_exc


def run_eval(eval_set_path: str, model_config: dict, cache_dir: str = "cache",
             results_dir: str = "results", images_dir: str = "data/coco/val2017") -> str:
    """評価を実行し results/{run_id}.json のパスを返す。"""
    with open(eval_set_path, encoding="utf-8") as f:
        eval_set = json.load(f)

    provider = build_provider(model_config)
    provider_name = model_config.get("provider", "?")
    model_tag = str(model_config.get("model") or model_config.get("weights") or provider_name)
    prompt_name = model_config.get("prompt", "-")
    eval_hash = _sha256_file(eval_set_path)

    # 重みファイル(YOLO の .pt 等)は拡張子を落とす。モデル ID(gemini-3.5-flash 等)は
    # Path().stem だと '.5-flash' が拡張子扱いされ 'gemini-3' に潰れる(版違いで run_id 衝突)ため
    # そのまま使い、ファイル名に使えない文字だけ安全側に置換する。
    _WEIGHT_SUFFIXES = (".pt", ".onnx", ".engine", ".pth")
    model_slug = Path(model_tag).stem if model_tag.lower().endswith(_WEIGHT_SUFFIXES) else model_tag
    model_slug = re.sub(r"[^0-9A-Za-z._-]", "_", model_slug)
    # プロンプトを持つプロバイダ(VLM)は run_id にプロンプト名を含める。含めないと
    # 「同じモデルで prompt だけ変えたラン」が同一 run_id になり、results/ を互いに
    # 上書きする(= プロンプト感度検証が成立しない)。キャッシュキーには元から prompt が
    # 入っているので、既存キャッシュがあれば新 run_id でも API 課金は発生しない。
    # prompt を持たない mock / YOLO の run_id は従来どおり(CI の baseline 名を壊さない)。
    parts = [provider_name, model_slug]
    if model_config.get("prompt"):
        parts.append(re.sub(r"[^0-9A-Za-z._-]", "_", prompt_name))
    run_id = "-".join(parts + [eval_hash[:8]])
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    result_path = Path(results_dir) / f"{run_id}.json"

    # 再開: 既存結果があれば済みの image_id を読み込む
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        records = existing.get("records", [])
    else:
        records = []
    done = {r["image_id"] for r in records}

    model_version = None
    for rec in records:  # 既存から model_version を引き継ぐ
        model_version = model_version or rec.get("_model_version")

    for entry in eval_set:
        image_id = entry["image_id"]
        if image_id in done:
            continue
        checklist = entry["checklist"]
        image_path = Path(images_dir) / entry["file_name"]

        key = cache_key(provider_name, model_tag, prompt_name, image_id, checklist)
        cache_file = Path(cache_dir) / f"{key}.json"
        if cache_file.exists():
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            cached = True
        else:
            t0 = time.time()
            resp = _judge_with_retry(provider, image_path, checklist)
            payload = _response_to_cache(resp)
            payload["latency_sec"] = round(time.time() - t0, 4) if payload["latency_sec"] == 0 \
                else payload["latency_sec"]
            cache_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            cached = False

        model_version = payload["model_version"]
        records.append({
            "image_id": image_id,
            "judgements": payload["judgements"],
            "usage": payload["usage"],
            "latency_sec": payload["latency_sec"],
            "cached": cached,
            "_model_version": payload["model_version"],
        })
        done.add(image_id)
        _save(result_path, run_id, provider_name, model_version, eval_set_path,
              eval_hash, records, prompt_name)

    _save(result_path, run_id, provider_name, model_version, eval_set_path, eval_hash,
          records, prompt_name)
    return str(result_path)


def _save(result_path: Path, run_id: str, provider_name: str, model_version: str | None,
          eval_set_path: str, eval_hash: str, records: list[dict],
          prompt_name: str = "-") -> None:
    # 公開スキーマからは内部用 _model_version を落とす
    clean = [{k: v for k, v in r.items() if k != "_model_version"} for r in records]
    obj = {
        "run_id": run_id,
        "provider": provider_name,
        "model_version": model_version,
        "prompt": prompt_name,
        "eval_set_path": eval_set_path,
        "eval_set_sha256": eval_hash,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "records": clean,
    }
    result_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    # .env から GEMINI_API_KEY 等を読み込む(既存の環境変数は上書きしない)。
    # mock / YOLO ランでは不要だが、Gemini ランで os.environ 経由のキー読込を成立させる。
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--model", required=True, help="conf/models.yaml のエントリ名")
    parser.add_argument("--models-config", default="conf/models.yaml")
    parser.add_argument("--images-dir", default="data/coco/val2017")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    with open(args.models_config, encoding="utf-8") as f:
        models = yaml.safe_load(f)
    if args.model not in models:
        raise SystemExit(f"モデルエントリ {args.model!r} が {args.models_config} に無い")

    path = run_eval(args.eval_set, models[args.model], cache_dir=args.cache_dir,
                    results_dir=args.results_dir, images_dir=args.images_dir)
    print(f"run 完了: {path}")


if __name__ == "__main__":
    main()
