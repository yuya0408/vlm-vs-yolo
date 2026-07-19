## 付録: 建設ドメインでの適用可能性デモ(定性)

> 本ベンチ(COCO)は YOLO の本拠地で、調整済み YOLO は VLM と精度互角・無料・高速だった。では専門ドメイン(建設)は? 対象物が検出器の語彙に無く、学習/調整用ラベルも無い。ここでは専用 YOLO は土俵に上がれず、汎用 VLM がゼロショットで動く唯一の選択肢になる。**正解ラベルが無いため精度比較ではなく『適用可能性』の定性実証**(F1・有意差は出さない)。

### YOLO の語彙ギャップ(推論不要・クラス一覧から確定)

建設チェックリスト 10 項目のうち、COCO 80 クラスに**無い**項目は **9 件(90%)**。これらは COCO 学習済み YOLO が conf 値によらず**絶対に present を出せない**= 構造的に判定不能。

- YOLO 語彙内: person
- YOLO 語彙外(判定不能): pipe, flange, tee fitting, elbow fitting, valve, gasket, bolted flange joint, pipe support, pressure gauge

### 画像ごとの判定

#### Flange01.jpg

> 出典: Wikimedia Commons / © 1971markus@wikipedia.de / CC BY-SA 4.0 — https://commons.wikimedia.org/wiki/File:Kokerei_Hansa_(08).jpg

| カテゴリ | 語彙 | YOLO | VLM 判定 | 根拠 |
|---|---|---|---|---|
| person | 内 | absent | absent | no people visible |
| pipe | **外** | 判定不能 | present×4 | multiple rusty metal pipes |
| flange | **外** | 判定不能 | present×6 | circular flanges on pipe joints |
| tee fitting | **外** | 判定不能 | present×1 | Y-branch tee fitting on main pipe |
| elbow fitting | **外** | 判定不能 | present×1 | 90-degree elbow bend on left pipe |
| valve | **外** | 判定不能 | absent | no valves visible |
| gasket | **外** | 判定不能 | uncertain | gaskets inside joints are not clearly visible |
| bolted flange joint | **外** | 判定不能 | present×3 | bolted flange connections visible |
| pipe support | **外** | 判定不能 | uncertain | no clear pipe supports visible |
| pressure gauge | **外** | 判定不能 | absent | no pressure gauges visible |

#### Piping01.jpg

> 出典: Wikimedia Commons / Markus Schweiss / CC BY-SA 3.0 — https://commons.wikimedia.org/wiki/File:Piping01.JPG

| カテゴリ | 語彙 | YOLO | VLM 判定 | 根拠 |
|---|---|---|---|---|
| person | 内 | absent | absent | no people visible in the image |
| pipe | **外** | 判定不能 | present×5 | multiple metal pipe spools on the pallet |
| flange | **外** | 判定不能 | present×9 | multiple circular flanges on the pipe ends |
| tee fitting | **外** | 判定不能 | present×1 | tee branch fitting in the center |
| elbow fitting | **外** | 判定不能 | present×5 | curved elbow sections on the pipes |
| valve | **外** | 判定不能 | absent | no valves visible on the pipes |
| gasket | **外** | 判定不能 | absent | no gaskets visible |
| bolted flange joint | **外** | 判定不能 | absent | no bolted flange connections visible |
| pipe support | **外** | 判定不能 | present×1 | metal support bracket welded to center pipe |
| pressure gauge | **外** | 判定不能 | absent | no pressure gauges visible |

#### Piping02.jpg

> 出典: Wikimedia Commons / U.S. Fish & Wildlife Service - Pacific Region / CC BY 2.0 — https://commons.wikimedia.org/wiki/File:Piping_system_at_Makah_National_Fish_Hatchery_(5837252675).jpg

| カテゴリ | 語彙 | YOLO | VLM 判定 | 根拠 |
|---|---|---|---|---|
| person | 内 | absent | absent | no people visible |
| pipe | **外** | 判定不能 | present×12 | multiple large blue and green pipes |
| flange | **外** | 判定不能 | present×20 | many circular flanges on pipes |
| tee fitting | **外** | 判定不能 | present×4 | tee junctions on vertical pipes |
| elbow fitting | **外** | 判定不能 | present×5 | curved elbow pipes near bottom |
| valve | **外** | 判定不能 | present×4 | valves with blue cylindrical actuators |
| gasket | **外** | 判定不能 | uncertain | gaskets are sandwiched inside flange joints |
| bolted flange joint | **外** | 判定不能 | present×15 | bolted flange connections |
| pipe support | **外** | 判定不能 | present×3 | support feet under bottom pipes |
| pressure gauge | **外** | 判定不能 | present×1 | dial pressure gauge on bottom pipe |

#### Piping03.png

> 出典: Wikimedia Commons / Wikikart99 / CC0 1.0 — https://commons.wikimedia.org/wiki/File:A_UPW_Installation_using_PVDF_Piping.png

| カテゴリ | 語彙 | YOLO | VLM 判定 | 根拠 |
|---|---|---|---|---|
| person | 内 | absent | absent | No people are visible in the image |
| pipe | **外** | 判定不能 | present×50 | Numerous white plastic pipes throughout the system |
| flange | **外** | 判定不能 | present×25 | Many circular flanges connecting pipe segments |
| tee fitting | **外** | 判定不能 | present×12 | T-shaped pipe connectors visible on lines |
| elbow fitting | **外** | 判定不能 | present×10 | 90-degree curved pipe elbows visible |
| valve | **外** | 判定不能 | present×18 | Orange and black control valves on pipes |
| gasket | **外** | 判定不能 | present×15 | Reddish sealing gaskets visible at joints |
| bolted flange joint | **外** | 判定不能 | present×12 | Flanges secured with bolts, especially top right |
| pipe support | **外** | 判定不能 | present×15 | Stainless steel framing supporting the piping network |
| pressure gauge | **外** | 判定不能 | absent | No dial pressure gauges are visible |

#### Piping04.jpg

> 出典: Wikimedia Commons / Audriusa / CC BY-SA 3.0 — https://commons.wikimedia.org/wiki/File:Pipeline_device.jpg

| カテゴリ | 語彙 | YOLO | VLM 判定 | 根拠 |
|---|---|---|---|---|
| person | 内 | absent | absent | no people visible |
| pipe | **外** | 判定不能 | present×5 | multiple metal and flexible pipes |
| flange | **外** | 判定不能 | present×4 | circular pipe flanges visible |
| tee fitting | **外** | 判定不能 | present×2 | T-junctions on the piping system |
| elbow fitting | **外** | 判定不能 | present×2 | curved elbow pipe sections |
| valve | **外** | 判定不能 | present×3 | valve wheel and handles visible |
| gasket | **外** | 判定不能 | uncertain | gaskets inside joints are not clearly visible |
| bolted flange joint | **外** | 判定不能 | present×3 | bolted flange connections visible |
| pipe support | **外** | 判定不能 | present×2 | concrete blocks supporting the pipes |
| pressure gauge | **外** | 判定不能 | absent | no pressure gauges visible |

### 読み

COCO 学習済み YOLO は語彙外カテゴリ(flange/tee fitting/valve 等の配管部品)に対し**原理的に present を返せない**(検出クラスに存在しない)。一方 VLM は同じ画像・同じチェックリストにゼロショットで判定と根拠を返す。COCO ベンチの結論「語彙内・ラベルありなら調整 YOLO で十分」と、本デモの「語彙外・ラベル無しなら VLM 一択」は**矛盾せず、一つの意思決定ルールの両輪**である。
