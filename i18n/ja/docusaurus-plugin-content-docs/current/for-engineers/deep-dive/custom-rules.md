---
sidebar_position: 10
title: カスタムルールと変則
description: KataGoがサポートする囲碁ルールセット、盤面サイズバリエーション、カスタム設定の詳細
---

# カスタムルールと変則

本記事では、KataGoがサポートする各種囲碁ルール、盤面サイズバリエーション、ルール設定のカスタマイズ方法を紹介します。

---

## ルールセット概要

### 主要ルール比較

| ルールセット | 計算方式 | コミ | 自殺手 | コウ返し |
|--------|---------|------|--------|---------|
| **Chinese** | 地+石 | 7.5 | 禁止 | 禁止 |
| **Japanese** | 地のみ | 6.5 | 禁止 | 禁止 |
| **Korean** | 地のみ | 6.5 | 禁止 | 禁止 |
| **AGA** | 混合 | 7.5 | 禁止 | 禁止 |
| **New Zealand** | 地+石 | 7 | 許可 | 禁止 |
| **Tromp-Taylor** | 地+石 | 7.5 | 許可 | 禁止 |

### KataGo設定

```ini
# config.cfg
rules = chinese           # ルールセット
komi = 7.5               # コミ
boardXSize = 19          # 盤面幅
boardYSize = 19          # 盤面高さ
```

---

## 中国ルール（Chinese）

### 特徴

```
計算方式：地+石（中国式数え方）
コミ：7.5目
自殺手：禁止
コウ返し：禁止（簡易ルール）
```

### 中国式数え方の説明

```
最終得点 = 自分の石の数 + 自分の地

例：
黒石 120個 + 黒地 65点 = 185点
白石 100個 + 白地 75点 + コミ 7.5 = 182.5点
黒の2.5点勝ち
```

### KataGo設定

```ini
rules = chinese
komi = 7.5
```

---

## 日本ルール（Japanese）

### 特徴

```
計算方式：地のみ（日本式数え方）
コミ：6.5目
自殺手：禁止
コウ返し：禁止
死石の判定が必要
```

### 日本式数え方の説明

```
最終得点 = 自分の地 + 取った相手の石

例：
黒地 65点 + アゲハマ 10個 = 75点
白地 75点 + アゲハマ 5個 + コミ 6.5 = 86.5点
白の11.5目勝ち
```

### 死石判定

日本ルールでは両者が死石について合意する必要があります：

```python
def is_dead_by_japanese_rules(group, game_state):
    """日本ルールでの死石判定"""
    # その石群が二眼を作れないことを証明する必要がある
    # これが日本ルールの複雑な点
    pass
```

### KataGo設定

```ini
rules = japanese
komi = 6.5
```

---

## AGAルール

### 特徴

アメリカ囲碁協会（AGA）ルールは中国・日本ルールの利点を組み合わせています：

```
計算方式：混合（地+石でも地のみでも同じ結果）
コミ：7.5目
自殺手：禁止
白番パス時は石を1個差し出す
```

### パスルール

```
黒番パス：石を出す必要なし
白番パス：黒に石を1個渡す必要あり

これにより地+石と地のみの結果が一致する
```

### KataGo設定

```ini
rules = aga
komi = 7.5
```

---

## Tromp-Taylorルール

### 特徴

最も簡潔な囲碁ルールで、プログラム実装に適しています：

```
計算方式：地+石
コミ：7.5目
自殺手：許可
コウ返し：Super Ko（任意の局面の繰り返しを禁止）
死石判定不要
```

### Super Ko

```python
def is_superko_violation(new_state, history):
    """Super Ko違反かどうかをチェック"""
    for past_state in history:
        if new_state == past_state:
            return True
    return False
```

### 終局判定

```
両者の死石合意は不要
対局は以下まで続く：
1. 両者が連続パス
2. その後、探索または実際に打って地を確定
```

### KataGo設定

```ini
rules = tromp-taylor
komi = 7.5
```

---

## 盤面サイズバリエーション

### サポートされるサイズ

KataGoは複数の盤面サイズをサポートしています：

| サイズ | 特徴 | 推奨用途 |
|------|------|---------|
| 9×9 | 約81点 | 入門、早碁 |
| 13×13 | 約169点 | 上級学習 |
| 19×19 | 361点 | 標準対局 |
| カスタム | 任意 | 研究、テスト |

### 設定方法

```ini
# 9×9盤面
boardXSize = 9
boardYSize = 9
komi = 5.5

# 13×13盤面
boardXSize = 13
boardYSize = 13
komi = 6.5

# 非正方形盤面
boardXSize = 19
boardYSize = 9
```

### コミの推奨値

| サイズ | 中国ルール | 日本ルール |
|------|---------|---------|
| 9×9 | 5.5 | 5.5 |
| 13×13 | 6.5 | 6.5 |
| 19×19 | 7.5 | 6.5 |

---

## 置き石設定

### 置き碁

置き石は棋力差を調整する方法です：

```ini
# 2子置き
handicap = 2

# 9子置き
handicap = 9
```

### 置き石の位置

```python
HANDICAP_POSITIONS = {
    2: [(3, 15), (15, 3)],
    3: [(3, 15), (15, 3), (15, 15)],
    4: [(3, 15), (15, 3), (3, 3), (15, 15)],
    # 5-9子は星 + 天元を使用
}
```

### 置き碁時のコミ

```ini
# 伝統的：置き碁ではコミなしまたは半目
komi = 0.5

# 現代的：置き石数に応じて調整
# 1子あたり約10-15目の価値
```

---

## Analysisモードでのルール設定

### GTPコマンド

```gtp
# ルールを設定
kata-set-rules chinese

# コミを設定
komi 7.5

# 盤面サイズを設定
boardsize 19
```

### Analysis API

```json
{
  "id": "query1",
  "moves": [["B", "Q4"], ["W", "D4"]],
  "rules": "chinese",
  "komi": 7.5,
  "boardXSize": 19,
  "boardYSize": 19,
  "overrideSettings": {
    "maxVisits": 1000
  }
}
```

---

## 高度なルールオプション

### 自殺手設定

```ini
# 自殺禁止（デフォルト）
allowSuicide = false

# 自殺許可（Tromp-Taylorスタイル）
allowSuicide = true
```

### コウルール

```ini
# Simple Ko（即座の返しのみ禁止）
koRule = SIMPLE

# Positional Super Ko（手番に関係なく任意の局面の繰り返しを禁止）
koRule = POSITIONAL

# Situational Super Ko（同じ手番での局面の繰り返しを禁止）
koRule = SITUATIONAL
```

### 計算ルール

```ini
# 地+石（中国、AGA）
scoringRule = AREA

# 地のみ（日本、韓国）
scoringRule = TERRITORY
```

### 税ルール

一部のルールではセキ領域に特別な計算があります：

```ini
# 税なし
taxRule = NONE

# セキで目なし
taxRule = SEKI

# すべての眼位で目なし
taxRule = ALL
```

---

## マルチルール訓練

### KataGoの利点

KataGoは単一モデルで複数のルールをサポートします：

```python
def encode_rules(rules):
    """ルールをニューラルネットワーク入力にエンコード"""
    features = np.zeros(RULE_FEATURE_SIZE)

    # 計算方式
    features[0] = 1.0 if rules.scoring == 'area' else 0.0

    # 自殺手
    features[1] = 1.0 if rules.allow_suicide else 0.0

    # コウルール
    features[2:5] = encode_ko_rule(rules.ko)

    # コミ（正規化）
    features[5] = rules.komi / 15.0

    return features
```

### ルール認識入力

```
ニューラルネットワーク入力に含まれる：
- 盤面状態（19×19×N）
- ルール特徴ベクトル（K次元）

これにより同じモデルが異なるルールを理解できる
```

---

## ルール切り替え例

### Pythonコード

```python
from katago import KataGo

engine = KataGo(model_path="kata.bin.gz")

# 中国ルールで分析
result_cn = engine.analyze(
    moves=[("B", "Q4"), ("W", "D4")],
    rules="chinese",
    komi=7.5
)

# 日本ルールで分析（同じ局面）
result_jp = engine.analyze(
    moves=[("B", "Q4"), ("W", "D4")],
    rules="japanese",
    komi=6.5
)

# 差異を比較
print(f"中国ルール黒勝率: {result_cn['winrate']:.1%}")
print(f"日本ルール黒勝率: {result_jp['winrate']:.1%}")
```

### ルールの影響分析

```python
def compare_rules_impact(position, rules_list):
    """異なるルールが局面評価に与える影響を比較"""
    results = {}

    for rules in rules_list:
        analysis = engine.analyze(
            moves=position,
            rules=rules,
            komi=get_default_komi(rules)
        )
        results[rules] = {
            'winrate': analysis['winrate'],
            'score': analysis['scoreLead'],
            'best_move': analysis['moveInfos'][0]['move']
        }

    return results
```

---

## よくある質問

### ルールの違いによる勝敗の変化

```
同じ対局でもルールが異なると結果が変わる可能性：
- 地+石 vs 地のみの計算差
- セキ領域の処理
- パスの影響
```

### どのルールを選ぶべきか？

| シナリオ | 推奨ルール |
|------|---------|
| 初心者 | Chinese（直感的、論争なし） |
| オンライン対局 | プラットフォームのデフォルト（通常Chinese） |
| 日本棋院 | Japanese |
| プログラム実装 | Tromp-Taylor（最も簡潔） |
| 中国プロ戦 | Chinese |

### 特定ルール用にモデルを訓練する必要があるか？

KataGoのマルチルールモデルは既に非常に強力です。しかし、単一ルールのみを使用する場合は以下を検討できます：

```ini
# 固定ルール訓練（特定ルールでの棋力がわずかに向上する可能性）
rules = chinese
```

---

## 関連記事

- [KataGo訓練メカニズム解析](../training) — マルチルール訓練の実装
- [プロジェクトへの統合](../../hands-on/integration) — API使用例
- [評価とベンチマーク](../evaluation) — 異なるルールでの棋力テスト
