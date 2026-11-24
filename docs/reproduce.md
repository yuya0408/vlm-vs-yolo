# 推論からやり直す

READMEの[動かし方](../README.md#動かし方)は、コミットしてある`results/`から数値と図を作り直す（無料）までを扱う。ここでは、YOLOとVLMの推論そのものをやり直す手順と、そのときの版を書く。

## 要るもの

| もの | 入手先 |
| --- | --- |
| COCO val2017の画像 | [cocodataset.org](https://cocodataset.org/#download) から`data/coco/val2017/`へ。使うのは`data/eval_set.json`の300枚 |
| YOLOの重み`yolo26x.pt` | Ultralyticsの配布物（AGPL-3.0）。このリポジトリには含めない。使ったファイルのsha256は`9fdd44a3…` |
| `GEMINI_API_KEY` | `.env`に書く（`.env.example`を参照）。VLMは1水準あたり約250〜310円（思考を含まない記録の値） |

## 手順

```bash
pip install -r locks/inference.lock   # YOLO の推論に使う版（下の「版」を参照）
python -m src.runner --eval-set data/eval_set.json --model yolo    # → results/yolo-yolo26x-289d6c0e.json
python -m src.analysis.yolo_threshold capture --eval-set data/eval_set.json --weights yolo26x.pt
python -m src.analysis.yolo_threshold sweep --eval-set data/eval_set.json
python -m src.analysis.yolo_threshold export --eval-set data/eval_set.json --threshold 0.075 \
    --base-run results/yolo-yolo26x-289d6c0e.json --out results/yolo-yolo26x-tuned075-289d6c0e.json
python -m src.runner --eval-set data/eval_set.json --model flash   # → results/gemini-gemini-3.5-flash-concise-289d6c0e.json
```

比較のコマンドが読む `results/gemini-gemini-3.5-flash-289d6c0e.json` は、プロンプト名を run_id に入れる前の名前で、判定は `-concise-` の付いたファイルと同じである。
プロンプト3水準のコマンドは [REPORT §2b](../report/REPORT.md#2b-vlm-側のプロンプト感度3-水準実行済み結論不変) にある。VLMの応答は`cache/`にあるので、同じプロンプトならAPIは呼ばれない。

## 版

| 処理 | 版の一覧 | 確かめたこと |
| --- | --- | --- |
| 統計の再計算 | [locks/analysis.lock](../locks/analysis.lock) | 新しいvenvで再計算のコマンドを回し、`results/`と`report/figures/`がコミット済みのものとバイト単位で一致した（Python 3.13・Windows） |
| YOLOの推論 | [locks/inference.lock](../locks/inference.lock)（ultralytics 8.4.75・torch 2.11.0） | 300枚を推論し直すと、検出30,558件の件数、しきい値スイープ、conf=0.075の判定と個数がコミット済みのものと全部一致した（Windows・CPU。信頼度の差は最大1.4×10⁻⁴） |

推論した当時（2026-06-23）は版を記録していなかったため、推論の一覧は、その日より前に公開されていた版から作った。2026-11-24以降の実行は、結果のJSONの`environment`にPython・OS・主なライブラリの版とデバイスが残る（`src/env_info.py`）。
2つの一覧は結果を再現するための記録で、更新しない。このコードを自分の用途で使うときは、版を固定していない`requirements.txt`から入れる。
