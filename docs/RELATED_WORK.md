# 関連研究・将来の活用メモ

> 本作(yolo-vs-vlm)に効きそうな外部の研究・データセットの控え。将来の拡張の参考として残す。
> 確定設計は `docs/DESIGN.md`。

## MEP 要素検出: 専用 YOLO vs オープン語彙(2025)

**論文**: Abdalwhab et al., "Are Open-Vocabulary Models Ready for Detection of MEP Elements
on Construction Sites" (arXiv:2501.09267v2, 2025)
<https://arxiv.org/html/2501.09267v2>

**何をした**: 現場ロボで収集した MEP データセット(8,885 枚 / 10 クラス: boiler, cable tray
fitting, electrical panel, fire alarm, pipe fitting, valve, outlet, generator, light, pump)で、
**ファインチューン済み YOLO11 Nano** と **ゼロショットのオープン語彙"検出器"3 種**
(Grounding DINO / Grounded SAM2 / DETIC)を比較。指標は P/R/F1 @IoU0.5。

**結果(test 分割の F1)**:

| | YOLO11(FT) | GSAM2 | GDINO | DETIC |
|---|---|---|---|---|
| all | **0.89** | 0.018 | 0.032 | 0.014 |

- YOLO11: precision 0.87 / recall 0.90。2.6M params、Jetson Orin Nano で 23 fps。
- オープン語彙勢は params が 63〜350 倍で、精度は桁違いに低い(全クラスで完敗、未検出クラスも)。

**読み方(重要な前提)**:

1. これは「**in-domain データで FT した専用** vs **データ無しのゼロショット汎用**」の比較。
   0.89 は 8,885 枚の現場ラベルがあって出る数字で、**データの有無がそのまま差**になっている。
   ラベルを取り去れば YOLO は学習対象が無く、COCO 学習済みのまま=現場語彙は見えない。
2. 比較対象の「オープン語彙」は **検出器**(GDINO/SAM2/DETIC)であり、**Gemini 系の生成 VLM
   ではない**。よって「生成 VLM がどうか」はこの論文からは言えない(著者も将来課題に残す)。
3. 含意: 「ドメイン学習データがあれば専用が勝つ / ゼロショットのオープン語彙検出器はまだ
   実用域でない」。**学習データが無い現場**では選択肢が「データ収集して FT」か「VLM」に絞られ、
   ゼロショット検出器が落第する以上、VLM を選ぶ判断が裏付けられる。

**本作への活かし方(将来)**:

- 比較軸を 3 すくみに拡張: 「COCO 学習済み YOLO(無 FT)/ 生成 VLM(Gemini)/ オープン語彙
  検出器(YOLO-World 等)」。この論文が埋めていない「生成 VLM」の位置を測れる。
- 「データ量 × 精度」の軸: 少量 FT で専用がどこから VLM を抜くか(損益分岐)を示せれば、
  「データが無いから VLM」を定量で語れる。
- 「MEP 学習済み YOLO を使えば?」への回答根拠: 公開モデルは汎用 10 クラスで粒度が
  合わず、専門ドメインの部品では結局 FT が必要 = データの壁に戻る、をこの論文の数字で補強。

## 建設ドメインの公開データセット / モデル(参考)

- **MEP Elements in Construction Site**(上記論文のデータ。9,125 枚・pipe fitting/valve 等)
  <https://universe.roboflow.com/yolo-nas-frb7l/mep-elements-in-construction-site>
- **SODA: Site Object Detection dAtaset**(汎用の現場物体検出。YOLOv3/v4 で mAP 81.47%)
  <https://arxiv.org/pdf/2202.09554>
- **Construction-PPE**(Ultralytics 公式データセット。ヘルメット/ベスト等の PPE 11 クラス)
  <https://docs.ultralytics.com/datasets/detect/construction-ppe>
- PPE 検出 YOLOv8 一式(学習ノート・重み・ダッシュボード)
  <https://github.com/VoxDroid/Construction-Site-Safety-PPE-Detection>

> 注意: いずれも公開・他者ドメインのデータ。本作は公開 COCO プロキシで IP を出さない方針。
> これらは将来の拡張の参考であり、本作に他者ドメインの固有データを持ち込むものではない。
