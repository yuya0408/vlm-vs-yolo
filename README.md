# vlm-vs-yolo

**写真と「期待される項目のリスト」から各項目の有無を判定するタスクを、汎用の視覚言語モデル(VLM)で解くシステムを考える。そのプロバイダ抽象は専用検出器(YOLO)への差し替えを許容する — では実際に差し替える価値はあるのか。同一タスク・同一評価セットで、精度・コスト・レイテンシのトレードオフと失敗条件を統計的に定量化して判断する比較評価フレームワーク。**

> 「画像 + 期待項目リスト → 各項目の有無(present / absent / uncertain)を判定」というタスクを題材に、
> 汎用 VLM(Gemini Flash)と従来型検出器 YOLO を同じ土俵で突き合わせ、
> ブートストラップ CI・McNemar 検定・誤り要因回帰で「専用検出器に替える価値があるか」を測る。
>
> 題材データは公開の COCO val2017。プロバイダ抽象を介して有無判定を行う VLM システムを、
> 公開データで再現した検証台として位置づける。

## 一番の結果: 「有意差」はベースラインの未調整が作っていた

最初の比較では **VLM の有意勝ち**(macro-F1 0.961 vs 0.906、McNemar **p=0.004**、CI 非重複)が出た。
「専用検出器の本拠地 COCO で汎用 VLM が勝つ」という、きれいな結論に見えた。

ところが YOLO の信頼度しきい値は **ライブラリ既定の conf=0.25 のまま**だった。これは
「調整済みモデル vs 未調整モデル」の比較であって「VLM vs YOLO」の比較ではない。
公平に調整すると:

| | 既定 conf=0.25 | 調整後 conf=0.075 | VLM (Flash) |
|---|---|---|---|
| macro-F1 (strict) | 0.906 | **0.929** | 0.961 |
| McNemar vs VLM | **p=0.004(有意)** | **p=0.71(有意差なし)** | — |
| 誤り要因 gt_present OR | **16.6**(極端な見逃し律速) | **2.43** | 2.25 |

**有意差は消えた。** さらに、誤り要因回帰から読めていた「YOLO は見逃しに偏る」という
*失敗モードの構造* まで既定しきい値のアーティファクトで、調整後は VLM とほぼ同型になった。
集計スコアだけでなく誤り分析の解釈までもが未調整ベースラインに汚染される、というのが本作の主眼。

結論を一点に依存させないため、妥当なしきい値域(conf 0.01〜0.10)**全域**で差が数 pt に収まることも示す。

詳細と図表は **[`report/REPORT.md`](report/REPORT.md)**。生データは [`results/`](results/)。

## 問いの立て方

こうした有無判定を汎用 VLM で解く構成では、アーキテクチャ上は YOLO プロバイダを差し込める。「専用検出器に替えれば精度・コストで割に合うのか / 汎用 VLM で既に十分なのか」は、感想ではなく統計で答えるべき意思決定だ。本作はその判断基盤を作る。

- **精度**: クローズドな語彙(COCO 80 クラス)で専用モデルが強い領域でも、汎用 VLM がどこまで迫れるか / どこで負けるか。差し替える価値があるのは「VLM が有意に劣り、かつ YOLO が安定して上回る」場合に限られる。
- **コスト・レイテンシ**: YOLO はローカル推論で無料・決定論的、VLM は API 課金。精度との引き換えをパレートで可視化する。
- **失敗条件**: 小物体・混同ペア・項目数・crowd など、どの条件でどちらが崩れるかを誤り要因回帰で特定し、差し替え判断の条件分岐に落とす。

## 比較を「有無判定」に揃える設計判断

YOLO の素の出力は bbox 検出だが、素朴に比べると「座標精度の勝負」に引っ張られ、VLM を不当に不利にする。そこで比較を**画像レベルのカテゴリ有無**に還元する。

- YOLO の検出を「そのカテゴリを信頼度しきい値以上で 1 個でも検出したか」で `present` / `absent` に還元する。bbox 座標そのものは比較に使わない。
- VLM は `present` / `absent` / `uncertain` を返す。
- 両者を同じ統計機構で突き合わせる。

YOLO は `uncertain` を持たない 2 値判定とし、「VLM だけが不確実性を表現できる」点自体を比較の観察対象として扱う(詳細は `docs/DESIGN.md`)。

## 特徴

- **統計的に厳密な比較**: 画像単位ブートストラップ 95% CI、対応ありの McNemar 検定、誤り要因のロジスティック回帰
- **ベースラインの公平な調整**: 低 conf で 1 回だけ全検出を保存し、後処理でしきい値スイープ(再推論ゼロ)。操作点は VLM を一切参照しない 3 基準の収束で選ぶ
- **トレードオフ分析**: 精度 / コスト / レイテンシのパレート(YOLO ローカル無料 vs VLM API 課金)
- **回帰ゲート(無料運用)**: PR ごとに mock の smoke 評価(n=50)を回し、ベースライン比で macro-F1 が劣化したら fail
- **再現性**: 評価セットはシード固定、VLM レスポンスは (model, prompt, image, checklist) ハッシュでキャッシュ、各ランは `results/{run_id}.json` に確定保存(MLflow/DVC は使わない)

## 評価設計の要点

- データ: COCO val2017 から N=300 を層化サンプリング(物体数・最小 bbox 面積比で層化)。正解はアノテーションから機械的に決定(アノテーションコスト 0)
- チェックリストの全カテゴリは COCO 80 クラスに収まるため、COCO 学習済み YOLO がそのまま比較対象になる
- negative 項目には混同しやすいカテゴリを意図的に混入(dog↔cat、car↔truck 等)
- localization(bbox/IoU)は意図的にスコープ外(比較を有無判定に揃えるため。詳細は `docs/DESIGN.md`)

## 結論(要約)

- **しきい値を公平に調整すると、汎用 VLM と専用検出器 YOLO の精度差はバンド内で消える。** 妥当な
  しきい値域(conf 0.01〜0.10)で tuned YOLO の macro-F1 は 0.92〜0.945、VLM(0.961)との差は数 pt、
  McNemar で有意差なし(主操作点 conf=0.075 で p=0.71)。**結論は閾値選択に依存しない。**
- 精度が互角なら**決定を支配するのはコストとレイテンシ**: YOLO は無料・約16倍速。
  → **閉語彙・ラベルありなら調整 YOLO で十分。** ただし**語彙外・ラベル無し(例: 配管施工のような専門ドメイン)では
  YOLO は構造的に判定不能で、ゼロショットの VLM が唯一の現実解**(REPORT 付録の定性デモ)。
- VLM は `uncertain` を返せるが実際にはほぼ使わない(0.31%)= 校正された不確実性は表現しない。

## ステータス

- [x] M1: 評価セット構築 + 指標 / ゲートの実装 + mock で E2E 疎通・単体テスト(API 費ゼロ)
- [x] M2: YOLO プロバイダ実装 + YOLO 単独で全指標を算出(ローカル推論・API 費ゼロ)
- [x] M3: Gemini Flash プロバイダ実装 + ベースライン VLM ラン(N=300 を 1 回)
- [x] M4: YOLO vs VLM 比較分析(McNemar + ブートストラップ CI + 誤り要因回帰 + パレート + 閾値チューニング)
- [x] M5: CI(回帰ゲートを mock smoke で無料運用)+ README / REPORT 仕上げ + 建設ドメイン定性デモ

M4 で当初予定していた「過剰主張 ablation」「risk-coverage 曲線」の扱いは `docs/DESIGN.md` §4 を参照。

## セットアップ

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # GEMINI_API_KEY を設定(mock / YOLO のみなら不要)
pytest                 # 単体テスト
```

YOLO の重み(`yolo26x.pt`, 約 113MB)は GitHub の 100MB 制限を超えるためリポジトリに含めない。
`ultralytics` が初回推論時に自動取得する。

## 再現手順

`results/` をコミットしてあるので、**どこから始めるかを選べる**。下に行くほど必要なものが増える。

### A. 何も落とさずに再現する(推奨・数分)

コミット済みの `results/` だけで、主比較・図・誤り taxonomy まで再計算できる。
COCO 画像も YOLO 重みも API キーも不要。

```bash
# しきい値スイープと操作点選定(生検出のみから計算。再推論なし)
python -m src.analysis.yolo_threshold sweep \
    --raw results/yolo_raw_detections.json --eval-set data/eval_set.json \
    --out results/yolo_threshold.json

# 選んだ操作点で tuned YOLO をラン形式に書き出す
python -m src.analysis.yolo_threshold export \
    --raw results/yolo_raw_detections.json --eval-set data/eval_set.json \
    --threshold 0.075 --out results/yolo-tuned-0.075.json

# 主比較(McNemar + ブートストラップ CI + 誤り要因回帰 + パレート)
python -m src.analysis.compare \
    --yolo results/yolo-tuned-0.075.json \
    --vlm  results/gemini-gemini-3.5-flash-289d6c0e.json \
    --eval-set data/eval_set.json --out results/comparison_tuned.json

# 誤り taxonomy と図
python -m src.analysis.error_taxonomy \
    --yolo results/yolo-tuned-0.075.json \
    --vlm  results/gemini-gemini-3.5-flash-289d6c0e.json \
    --eval-set data/eval_set.json --out results/error_taxonomy.json
python -m src.analysis.plots --comparison results/comparison_tuned.json \
    --out report/figures/pareto.png \
    --threshold-result results/yolo_threshold.json \
    --sweep-out report/figures/threshold_sweep.png
```

### B. YOLO 側を推論からやり直す(無料・ローカル GPU/CPU)

COCO val2017 の画像が要る。API 費はかからない。

```bash
# COCO val2017(画像 + アノテーション)を data/coco/ に展開する
mkdir -p data/coco && cd data/coco
curl -O http://images.cocodataset.org/zips/val2017.zip
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q val2017.zip && unzip -q annotations_trainval2017.zip && cd ../..

# 評価セットを再構築(シード固定。コミット済みの data/eval_set.json と一致するはず)
python -m src.data.build_eval_set --config conf/eval.yaml --out data/eval_set.json

# 既定 conf=0.25 での YOLO ラン(= 未調整ベースライン。本作が退けた比較)
python -m src.runner --eval-set data/eval_set.json --model yolo

# 低 conf で全検出を保存 → 以降のスイープは再推論なしで回る
python -m src.analysis.yolo_threshold capture \
    --eval-set data/eval_set.json --weights yolo26x.pt \
    --out results/yolo_raw_detections.json
```

以降は A に合流する。

### C. VLM ランからやり直す(API 費が発生。N=300 で約 251 円)

```bash
python -m src.runner --eval-set data/eval_set.json --model flash
```

レスポンスは `cache/` にハッシュキーで保存され、再実行時は API を呼ばない。
中断しても `results/{run_id}.json` から未処理分だけ再開する。

### 建設ドメインの定性デモ(付録)

```bash
python -m src.analysis.construction_demo \
    --images data/construction --out results/construction_demo.json \
    --report report/construction_demo_section.md --with-yolo
```

## ライセンス・データ

- 画像: [COCO dataset](https://cocodataset.org/)(val2017)。ライセンスは COCO の規約に従う
- 建設デモの画像: Wikimedia Commons の CC BY / CC BY-SA / CC0 素材。帰属は `data/construction/attributions.json` と REPORT 付録に記載
- YOLO: [Ultralytics](https://github.com/ultralytics/ultralytics) の COCO 学習済みモデル
- 本リポジトリのコード: MIT([`LICENSE`](LICENSE))
