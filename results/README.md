# results/ — レポートの根拠データ

`report/REPORT.md` が引用する数値は、すべてここのファイルから出ている。
**レポートの主張を自分で検算できるようにするため、生成物をリポジトリにコミットしている**
(API キーは含まれない)。再生成の手順は README の「再現手順」を参照。

| ファイル | 中身 | 生成コマンド |
|---|---|---|
| `yolo_raw_detections.json` | conf=0.001 で 1 回だけ推論した全検出の信頼度 + 画像ごとの実測レイテンシ | `yolo_threshold capture` |
| `yolo_threshold.json` | しきい値スイープ曲線・バンド要約・操作点の選定根拠(3 基準) | `yolo_threshold sweep` |
| `yolo-tuned-0.075.json` | 主比較で使う tuned YOLO の判定(ラン形式) | `yolo_threshold export` |
| `yolo-yolo26x-*.json` | 既定 conf=0.25 の YOLO ラン(= 本作が退けた未調整ベースライン) | `src.runner --model yolo` |
| `gemini-gemini-3.5-flash-*.json` | VLM ラン(N=300, temp=0) | `src.runner --model flash` |
| `comparison_tuned.json` | 主比較: McNemar・ブートストラップ CI・誤り要因回帰・パレート | `src.analysis.compare` |
| `error_taxonomy.json` | 誤りバケット別集計 + 代表誤答例 | `src.analysis.error_taxonomy` |
| `construction_demo.json` | 建設ドメイン定性デモの判定結果(付録) | `src.analysis.construction_demo` |

`yolo_raw_detections.json` があれば、COCO 画像(19GB)も YOLO 重み(113MB)も無しに
しきい値スイープから主比較まで丸ごと再現できる。

VLM の生レスポンスキャッシュ(`cache/`)は再取得可能なのでコミットしていない。
