---
sidebar_position: 10
title: 自定义规则与变体
description: KataGo 支持的围棋规则集、变体棋盘与自定义配置详解
---

# 自定义规则与变体

本文介绍 KataGo 支持的各种围棋规则、棋盘大小变体，以及如何自定义规则配置。

---

## 规则集总览

### 主要规则比较

| 规则集 | 计目方式 | 贴目 | 自杀手 | 打劫还原 |
|--------|---------|------|--------|---------|
| **Chinese** | 数子法 | 7.5 | 禁止 | 禁止 |
| **Japanese** | 数目法 | 6.5 | 禁止 | 禁止 |
| **Korean** | 数目法 | 6.5 | 禁止 | 禁止 |
| **AGA** | 混合 | 7.5 | 禁止 | 禁止 |
| **New Zealand** | 数子法 | 7 | 允许 | 禁止 |
| **Tromp-Taylor** | 数子法 | 7.5 | 允许 | 禁止 |

### KataGo 配置

```ini
# config.cfg
rules = chinese           # 规则集
komi = 7.5               # 贴目
boardXSize = 19          # 棋盘宽度
boardYSize = 19          # 棋盘高度
```

---

## 中国规则（Chinese）

### 特点

```
计目方式：数子法（子空皆地）
贴目：7.5 目
自杀手：禁止
打劫还原：禁止（简易规则）
```

### 数子法说明

```
最终得分 = 己方棋子数 + 己方空点数

示例：
黑子 120 颗 + 黑空 65 点 = 185 点
白子 100 颗 + 白空 75 点 + 贴目 7.5 = 182.5 点
黑胜 2.5 点
```

### KataGo 配置

```ini
rules = chinese
komi = 7.5
```

---

## 日本规则（Japanese）

### 特点

```
计目方式：数目法（只算空点）
贴目：6.5 目
自杀手：禁止
打劫还原：禁止
需要标记死子
```

### 数目法说明

```
最终得分 = 己方空点数 + 提掉的对方棋子数

示例：
黑空 65 点 + 提子 10 颗 = 75 点
白空 75 点 + 提子 5 颗 + 贴目 6.5 = 86.5 点
白胜 11.5 目
```

### 死子判定

日本规则需要双方同意哪些棋是死子：

```python
def is_dead_by_japanese_rules(group, game_state):
    """日本规则下判定死子"""
    # 需要证明该棋串无法做出两眼
    # 这是日本规则的复杂之处
    pass
```

### KataGo 配置

```ini
rules = japanese
komi = 6.5
```

---

## AGA 规则

### 特点

美国围棋协会（AGA）规则结合了中日规则的优点：

```
计目方式：混合（数子或数目皆可，结果相同）
贴目：7.5 目
自杀手：禁止
白方需填一子来 pass
```

### Pass 规则

```
黑方 pass：不需填子
白方 pass：需要向黑方交出一颗子

这让数子法和数目法结果一致
```

### KataGo 配置

```ini
rules = aga
komi = 7.5
```

---

## Tromp-Taylor 规则

### 特点

最简洁的围棋规则，适合程序实现：

```
计目方式：数子法
贴目：7.5 目
自杀手：允许
打劫还原：Super Ko（禁止任何重复局面）
无需判定死子
```

### Super Ko

```python
def is_superko_violation(new_state, history):
    """检查是否违反 Super Ko"""
    for past_state in history:
        if new_state == past_state:
            return True
    return False
```

### 终局判定

```
无需双方同意死子
比赛持续直到：
1. 双方连续 pass
2. 然后用搜索或实际下完来确定领地
```

### KataGo 配置

```ini
rules = tromp-taylor
komi = 7.5
```

---

## 棋盘大小变体

### 支持的尺寸

KataGo 支持多种棋盘大小：

| 尺寸 | 特点 | 建议用途 |
|------|------|---------|
| 9×9 | 约 81 点 | 初学、快棋 |
| 13×13 | 约 169 点 | 进阶学习 |
| 19×19 | 361 点 | 标准比赛 |
| 自定义 | 任意 | 研究、测试 |

### 配置方式

```ini
# 9×9 棋盘
boardXSize = 9
boardYSize = 9
komi = 5.5

# 13×13 棋盘
boardXSize = 13
boardYSize = 13
komi = 6.5

# 非正方形棋盘
boardXSize = 19
boardYSize = 9
```

### 贴目建议

| 尺寸 | 中国规则 | 日本规则 |
|------|---------|---------|
| 9×9 | 5.5 | 5.5 |
| 13×13 | 6.5 | 6.5 |
| 19×19 | 7.5 | 6.5 |

---

## 让子配置

### 让子棋

让子是调整棋力差距的方式：

```ini
# 让 2 子
handicap = 2

# 让 9 子
handicap = 9
```

### 让子位置

```python
HANDICAP_POSITIONS = {
    2: [(3, 15), (15, 3)],
    3: [(3, 15), (15, 3), (15, 15)],
    4: [(3, 15), (15, 3), (3, 3), (15, 15)],
    # 5-9 子使用星位 + 天元
}
```

### 让子时的贴目

```ini
# 传统：让子不贴目或贴半目
komi = 0.5

# 现代：根据让子数调整
# 每子约值 10-15 目
```

---

## Analysis 模式规则配置

### GTP 指令

```gtp
# 设置规则
kata-set-rules chinese

# 设置贴目
komi 7.5

# 设置棋盘大小
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

## 进阶规则选项

### 自杀手配置

```ini
# 禁止自杀（默认）
allowSuicide = false

# 允许自杀（Tromp-Taylor 风格）
allowSuicide = true
```

### Ko 规则

```ini
# Simple Ko（仅禁止立即还原）
koRule = SIMPLE

# Positional Super Ko（禁止重复任何局面，不论轮到谁下）
koRule = POSITIONAL

# Situational Super Ko（禁止重复同一方下的局面）
koRule = SITUATIONAL
```

### 计分规则

```ini
# 数子法（中国、AGA）
scoringRule = AREA

# 数目法（日本、韩国）
scoringRule = TERRITORY
```

### 税金规则

有些规则对双活区域有特殊计分：

```ini
# 无税金
taxRule = NONE

# 双活无目
taxRule = SEKI

# 所有眼位无目
taxRule = ALL
```

---

## 多规则训练

### KataGo 的优势

KataGo 使用单一模型支持多种规则：

```python
def encode_rules(rules):
    """将规则编码为神经网络输入"""
    features = np.zeros(RULE_FEATURE_SIZE)

    # 计分方式
    features[0] = 1.0 if rules.scoring == 'area' else 0.0

    # 自杀手
    features[1] = 1.0 if rules.allow_suicide else 0.0

    # Ko 规则
    features[2:5] = encode_ko_rule(rules.ko)

    # 贴目（归一化）
    features[5] = rules.komi / 15.0

    return features
```

### 规则感知输入

```
神经网络输入包含：
- 棋盘状态（19×19×N）
- 规则特征向量（K 维）

这让同一模型能理解不同规则
```

---

## 规则切换示例

### Python 代码

```python
from katago import KataGo

engine = KataGo(model_path="kata.bin.gz")

# 中国规则分析
result_cn = engine.analyze(
    moves=[("B", "Q4"), ("W", "D4")],
    rules="chinese",
    komi=7.5
)

# 日本规则分析（同一局面）
result_jp = engine.analyze(
    moves=[("B", "Q4"), ("W", "D4")],
    rules="japanese",
    komi=6.5
)

# 比较差异
print(f"中国规则黑胜率: {result_cn['winrate']:.1%}")
print(f"日本规则黑胜率: {result_jp['winrate']:.1%}")
```

### 规则影响分析

```python
def compare_rules_impact(position, rules_list):
    """比较不同规则对局面评估的影响"""
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

## 常见问题

### 规则差异造成的胜负不同

```
同一盘棋，不同规则可能导致不同结果：
- 数子法 vs 数目法的计分差异
- 双活区域的处理
- 虚手（pass）的影响
```

### 选择哪种规则？

| 场景 | 建议规则 |
|------|---------|
| 初学者 | Chinese（直观、无争议） |
| 在线比赛 | 平台默认（通常是 Chinese） |
| 日本棋院 | Japanese |
| 程序实现 | Tromp-Taylor（最简洁） |
| 中国职业赛 | Chinese |

### 模型是否需要针对特定规则训练？

KataGo 的多规则模型已经很强。但如果只使用单一规则，可以考虑：

```ini
# 固定规则训练（可能略微提升特定规则下的棋力）
rules = chinese
```

---

## 延伸阅读

- [KataGo 训练机制解析](../training) — 多规则训练的实现
- [集成到你的项目](../../hands-on/integration) — API 使用示例
- [评估与基准测试](../evaluation) — 不同规则下的棋力测试
