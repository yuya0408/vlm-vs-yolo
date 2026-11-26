# ADRの一覧

閾値・プロンプト・損益分岐の判断の根拠を、却下した案とともに残している。書き方は[0000-template.md](0000-template.md)。
決めた日付は、2026-07-19より前のものはコミットの履歴に無く、非公開の作業メモによる。

| ADR | 決めたこと | 状態 |
| --- | --- | --- |
| [0001](0001-threshold-operating-point.md) | YOLOの閾値を全点で走査し、VLMのスコアを使わない基準（micro-F1とitem accuracyの最大）で操作点を選ぶ。差の判定は画像単位のペア差CIを主にする | Accepted |
| [0002](0002-prompt-sensitivity-rule.md) | VLMのプロンプトを3水準振り、YOLOの閾値と同じtune/testの手順で選んで、結論が変わるかを確かめる | Accepted |
| [0003](0003-breakeven-assumptions.md) | 損益分岐は、YOLOの固定費をYOLOに有利な値に置き、固定費を上下に振った感度で読む | Accepted |
