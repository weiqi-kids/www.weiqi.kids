---
sidebar_position: 10
title: 自訂規則與變體
description: KataGo 支援的圍棋規則集、變體棋盤與自訂設定詳解
---

# 自訂規則與變體

本文介紹 KataGo 支援的各種圍棋規則、棋盤大小變體，以及如何自訂規則設定。

---

## 規則集總覽

### 主要規則比較

| 規則集 | 計目方式 | 貼目 | 自殺手 | 打劫還原 |
|--------|---------|------|--------|---------|
| **Chinese** | 數子法 | 7.5 | 禁止 | 禁止 |
| **Japanese** | 數目法 | 6.5 | 禁止 | 禁止 |
| **Korean** | 數目法 | 6.5 | 禁止 | 禁止 |
| **AGA** | 混合 | 7.5 | 禁止 | 禁止 |
| **New Zealand** | 數子法 | 7 | 允許 | 禁止 |
| **Tromp-Taylor** | 數子法 | 7.5 | 允許 | 禁止 |

### KataGo 設定

```ini
# config.cfg
rules = chinese           # 規則集
komi = 7.5               # 貼目
boardXSize = 19          # 棋盤寬度
boardYSize = 19          # 棋盤高度
```

---

## 中國規則（Chinese）

### 特點

```
計目方式：數子法（子空皆地）
貼目：7.5 目
自殺手：禁止
打劫還原：禁止（簡易規則）
```

### 數子法說明

```
最終得分 = 己方棋子數 + 己方空點數

範例：
黑子 120 顆 + 黑空 65 點 = 185 點
白子 100 顆 + 白空 75 點 + 貼目 7.5 = 182.5 點
黑勝 2.5 點
```

### KataGo 設定

```ini
rules = chinese
komi = 7.5
```

---

## 日本規則（Japanese）

### 特點

```
計目方式：數目法（只算空點）
貼目：6.5 目
自殺手：禁止
打劫還原：禁止
需要標記死子
```

### 數目法說明

```
最終得分 = 己方空點數 + 提掉的對方棋子數

範例：
黑空 65 點 + 提子 10 顆 = 75 點
白空 75 點 + 提子 5 顆 + 貼目 6.5 = 86.5 點
白勝 11.5 目
```

### 死子判定

日本規則需要雙方同意哪些棋是死子：

```python
def is_dead_by_japanese_rules(group, game_state):
    """日本規則下判定死子"""
    # 需要證明該棋串無法做出兩眼
    # 這是日本規則的複雜之處
    pass
```

### KataGo 設定

```ini
rules = japanese
komi = 6.5
```

---

## AGA 規則

### 特點

美國圍棋協會（AGA）規則結合了中日規則的優點：

```
計目方式：混合（數子或數目皆可，結果相同）
貼目：7.5 目
自殺手：禁止
白方需填一子來 pass
```

### Pass 規則

```
黑方 pass：不需填子
白方 pass：需要向黑方交出一顆子

這讓數子法和數目法結果一致
```

### KataGo 設定

```ini
rules = aga
komi = 7.5
```

---

## Tromp-Taylor 規則

### 特點

最簡潔的圍棋規則，適合程式實作：

```
計目方式：數子法
貼目：7.5 目
自殺手：允許
打劫還原：Super Ko（禁止任何重複局面）
無需判定死子
```

### Super Ko

```python
def is_superko_violation(new_state, history):
    """檢查是否違反 Super Ko"""
    for past_state in history:
        if new_state == past_state:
            return True
    return False
```

### 終局判定

```
無需雙方同意死子
比賽持續直到：
1. 雙方連續 pass
2. 然後用搜索或實際下完來確定領地
```

### KataGo 設定

```ini
rules = tromp-taylor
komi = 7.5
```

---

## 棋盤大小變體

### 支援的尺寸

KataGo 支援多種棋盤大小：

| 尺寸 | 特點 | 建議用途 |
|------|------|---------|
| 9×9 | 約 81 點 | 初學、快棋 |
| 13×13 | 約 169 點 | 進階學習 |
| 19×19 | 361 點 | 標準比賽 |
| 自訂 | 任意 | 研究、測試 |

### 設定方式

```ini
# 9×9 棋盤
boardXSize = 9
boardYSize = 9
komi = 5.5

# 13×13 棋盤
boardXSize = 13
boardYSize = 13
komi = 6.5

# 非正方形棋盤
boardXSize = 19
boardYSize = 9
```

### 貼目建議

| 尺寸 | 中國規則 | 日本規則 |
|------|---------|---------|
| 9×9 | 5.5 | 5.5 |
| 13×13 | 6.5 | 6.5 |
| 19×19 | 7.5 | 6.5 |

---

## 讓子設定

### 讓子棋

讓子是調整棋力差距的方式：

```ini
# 讓 2 子
handicap = 2

# 讓 9 子
handicap = 9
```

### 讓子位置

```python
HANDICAP_POSITIONS = {
    2: [(3, 15), (15, 3)],
    3: [(3, 15), (15, 3), (15, 15)],
    4: [(3, 15), (15, 3), (3, 3), (15, 15)],
    # 5-9 子使用星位 + 天元
}
```

### 讓子時的貼目

```ini
# 傳統：讓子不貼目或貼半目
komi = 0.5

# 現代：根據讓子數調整
# 每子約值 10-15 目
```

---

## Analysis 模式規則設定

### GTP 指令

```gtp
# 設定規則
kata-set-rules chinese

# 設定貼目
komi 7.5

# 設定棋盤大小
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

## 進階規則選項

### 自殺手設定

```ini
# 禁止自殺（預設）
allowSuicide = false

# 允許自殺（Tromp-Taylor 風格）
allowSuicide = true
```

### Ko 規則

```ini
# Simple Ko（僅禁止立即還原）
koRule = SIMPLE

# Positional Super Ko（禁止重複任何局面，不論輪到誰下）
koRule = POSITIONAL

# Situational Super Ko（禁止重複同一方下的局面）
koRule = SITUATIONAL
```

### 計分規則

```ini
# 數子法（中國、AGA）
scoringRule = AREA

# 數目法（日本、韓國）
scoringRule = TERRITORY
```

### 稅金規則

有些規則對雙活區域有特殊計分：

```ini
# 無稅金
taxRule = NONE

# 雙活無目
taxRule = SEKI

# 所有眼位無目
taxRule = ALL
```

---

## 多規則訓練

### KataGo 的優勢

KataGo 使用單一模型支援多種規則：

```python
def encode_rules(rules):
    """將規則編碼為神經網路輸入"""
    features = np.zeros(RULE_FEATURE_SIZE)

    # 計分方式
    features[0] = 1.0 if rules.scoring == 'area' else 0.0

    # 自殺手
    features[1] = 1.0 if rules.allow_suicide else 0.0

    # Ko 規則
    features[2:5] = encode_ko_rule(rules.ko)

    # 貼目（正規化）
    features[5] = rules.komi / 15.0

    return features
```

### 規則感知輸入

```
神經網路輸入包含：
- 棋盤狀態（19×19×N）
- 規則特徵向量（K 維）

這讓同一模型能理解不同規則
```

---

## 規則切換範例

### Python 程式碼

```python
from katago import KataGo

engine = KataGo(model_path="kata.bin.gz")

# 中國規則分析
result_cn = engine.analyze(
    moves=[("B", "Q4"), ("W", "D4")],
    rules="chinese",
    komi=7.5
)

# 日本規則分析（同一局面）
result_jp = engine.analyze(
    moves=[("B", "Q4"), ("W", "D4")],
    rules="japanese",
    komi=6.5
)

# 比較差異
print(f"中國規則黑勝率: {result_cn['winrate']:.1%}")
print(f"日本規則黑勝率: {result_jp['winrate']:.1%}")
```

### 規則影響分析

```python
def compare_rules_impact(position, rules_list):
    """比較不同規則對局面評估的影響"""
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

## 常見問題

### 規則差異造成的勝負不同

```
同一盤棋，不同規則可能導致不同結果：
- 數子法 vs 數目法的計分差異
- 雙活區域的處理
- 虛手（pass）的影響
```

### 選擇哪種規則？

| 場景 | 建議規則 |
|------|---------|
| 初學者 | Chinese（直觀、無爭議） |
| 線上比賽 | 平台預設（通常是 Chinese） |
| 日本棋院 | Japanese |
| 程式實作 | Tromp-Taylor（最簡潔） |
| 中國職業賽 | Chinese |

### 模型是否需要針對特定規則訓練？

KataGo 的多規則模型已經很強。但如果只使用單一規則，可以考慮：

```ini
# 固定規則訓練（可能略微提升特定規則下的棋力）
rules = chinese
```

---

## 延伸閱讀

- [KataGo 訓練機制解析](../training) — 多規則訓練的實作
- [整合到你的專案](../../hands-on/integration) — API 使用範例
- [評估與基準測試](../evaluation) — 不同規則下的棋力測試
