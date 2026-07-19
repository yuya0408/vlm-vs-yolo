# 配管施工デモ画像(CC BY / パブリックドメインのみ)

`src/analysis/construction_demo.py` が読む、配管施工ドメインの定性デモ用画像を置く場所。
実務で検出に挑んだ対象(配管・フランジ・チーズ等の配管施工部品)に合わせる。
**顧客の現場写真は置かない。** 公開かつ再掲可能(CC BY / CC0 / PD)な画像のみ。

## 調達手順(ローカルで実施)

1. Wikimedia Commons 等で、配管施工部品が写った画像を 5〜8 枚集める。
   **COCO 80 クラスに概念ごと無い**対象が写っているものを優先:
   `pipe`(配管) / `flange`(フランジ) / `tee fitting`(チーズ) / `elbow fitting`(エルボ) /
   `valve`(バルブ) / `gasket`(ガスケット) / `bolted flange joint`(ボルト締め接合部) /
   `pipe support`(配管サポート) / `pressure gauge`(圧力計)。
   - 検索例(Wikimedia Commons): "pipe flange", "flanged pipe joint", "pipe fittings tee elbow",
     "industrial piping valves", "pipe rack supports", "plumbing rough-in pipes"。
   - 対照として作業員(`person`)が写る配管現場の写真が 1〜2 枚あると、
     「YOLO は person だけ拾えて配管部品は全滅」という対比が一目で出る。
   - ライセンスは各ファイルページで **CC BY 4.0 / CC BY-SA / CC0 / Public domain** を確認。
     CC BY-NC は公開ブログ再掲が灰色なので使わない。

2. 画像をこのフォルダに保存(.jpg/.png/.webp)。ファイル名は英数字推奨。

3. `attributions.json` に各画像の出典を記入(CC BY は帰属表示が必須)。
   `attributions.example.json` をコピーして作成する。

## 実行

```bash
# リポジトリ直下で。GEMINI_API_KEY が .env か環境変数にあること。
source .venv/bin/activate
python -m src.analysis.construction_demo --images data/construction --with-yolo
# 出力: results/construction_demo.json と report/construction_demo_section.md
```

- 既定チェックリストは配管施工部品(`construction_demo.py` の `DEFAULT_CHECKLIST`)。
  画像に合わせ `--checklist person pipe flange "tee fitting" valve ...` で上書き可。
- `--with-yolo` を付けると YOLO も走り、配管部品で「判定不能(absent しか出せない)」様子を併記。
- VLM コールは 1 画像 1 回(数十円規模)。レスポンスはランごとに JSON 保存。

## 注意

- このフォルダの画像と `attributions.json` は **コミットしてよい**(CC BY/PD のため)。
  ライセンス・帰属を `attributions.json` に必ず残すこと。
