---
sidebar_position: 10
title: Custom Rules & Variants
description: Go rulesets supported by KataGo, board size variants, and custom settings
---

# Custom Rules & Variants

This article introduces the various Go rules supported by KataGo, board size variants, and how to customize rule settings.

---

## Ruleset Overview

### Main Rules Comparison

| Ruleset | Scoring Method | Komi | Suicide | Ko Repetition |
|---------|---------------|------|---------|---------------|
| **Chinese** | Area scoring | 7.5 | Forbidden | Forbidden |
| **Japanese** | Territory scoring | 6.5 | Forbidden | Forbidden |
| **Korean** | Territory scoring | 6.5 | Forbidden | Forbidden |
| **AGA** | Hybrid | 7.5 | Forbidden | Forbidden |
| **New Zealand** | Area scoring | 7 | Allowed | Forbidden |
| **Tromp-Taylor** | Area scoring | 7.5 | Allowed | Forbidden |

### KataGo Settings

```ini
# config.cfg
rules = chinese           # Ruleset
komi = 7.5               # Komi
boardXSize = 19          # Board width
boardYSize = 19          # Board height
```

---

## Chinese Rules

### Features

```
Scoring method: Area scoring (stones + territory)
Komi: 7.5 points
Suicide: Forbidden
Ko repetition: Forbidden (simple rule)
```

### Area Scoring Explanation

```
Final score = Own stones + Own empty points

Example:
Black stones 120 + Black territory 65 = 185 points
White stones 100 + White territory 75 + Komi 7.5 = 182.5 points
Black wins by 2.5 points
```

### KataGo Settings

```ini
rules = chinese
komi = 7.5
```

---

## Japanese Rules

### Features

```
Scoring method: Territory scoring (only empty points)
Komi: 6.5 points
Suicide: Forbidden
Ko repetition: Forbidden
Requires marking dead stones
```

### Territory Scoring Explanation

```
Final score = Own empty points + Captured opponent stones

Example:
Black territory 65 + Captures 10 = 75 points
White territory 75 + Captures 5 + Komi 6.5 = 86.5 points
White wins by 11.5 points
```

### Dead Stone Determination

Japanese rules require both players to agree on which stones are dead:

```python
def is_dead_by_japanese_rules(group, game_state):
    """Determine dead stones under Japanese rules"""
    # Need to prove the group cannot make two eyes
    # This is the complexity of Japanese rules
    pass
```

### KataGo Settings

```ini
rules = japanese
komi = 6.5
```

---

## AGA Rules

### Features

American Go Association (AGA) rules combine advantages of Chinese and Japanese rules:

```
Scoring method: Hybrid (area or territory, same result)
Komi: 7.5 points
Suicide: Forbidden
White must fill one stone to pass
```

### Pass Rules

```
Black passes: No stone needed
White passes: Must give one stone to Black

This makes area and territory scoring results identical
```

### KataGo Settings

```ini
rules = aga
komi = 7.5
```

---

## Tromp-Taylor Rules

### Features

The simplest Go rules, suitable for programming:

```
Scoring method: Area scoring
Komi: 7.5 points
Suicide: Allowed
Ko repetition: Super Ko (forbid any repeated position)
No dead stone determination needed
```

### Super Ko

```python
def is_superko_violation(new_state, history):
    """Check if Super Ko is violated"""
    for past_state in history:
        if new_state == past_state:
            return True
    return False
```

### End Game Determination

```
No agreement on dead stones needed
Game continues until:
1. Both players pass consecutively
2. Then use search or actual play to determine territory
```

### KataGo Settings

```ini
rules = tromp-taylor
komi = 7.5
```

---

## Board Size Variants

### Supported Sizes

KataGo supports multiple board sizes:

| Size | Features | Recommended Use |
|------|----------|-----------------|
| 9×9 | ~81 points | Beginners, quick games |
| 13×13 | ~169 points | Advanced learning |
| 19×19 | 361 points | Standard competition |
| Custom | Any | Research, testing |

### Configuration

```ini
# 9×9 board
boardXSize = 9
boardYSize = 9
komi = 5.5

# 13×13 board
boardXSize = 13
boardYSize = 13
komi = 6.5

# Non-square board
boardXSize = 19
boardYSize = 9
```

### Komi Recommendations

| Size | Chinese Rules | Japanese Rules |
|------|--------------|----------------|
| 9×9 | 5.5 | 5.5 |
| 13×13 | 6.5 | 6.5 |
| 19×19 | 7.5 | 6.5 |

---

## Handicap Settings

### Handicap Games

Handicap is a way to adjust for strength differences:

```ini
# 2 stone handicap
handicap = 2

# 9 stone handicap
handicap = 9
```

### Handicap Positions

```python
HANDICAP_POSITIONS = {
    2: [(3, 15), (15, 3)],
    3: [(3, 15), (15, 3), (15, 15)],
    4: [(3, 15), (15, 3), (3, 3), (15, 15)],
    # 5-9 stones use star points + tengen
}
```

### Komi with Handicap

```ini
# Traditional: No komi or half point with handicap
komi = 0.5

# Modern: Adjust based on handicap stones
# Each stone is worth ~10-15 points
```

---

## Analysis Mode Rule Settings

### GTP Commands

```gtp
# Set rules
kata-set-rules chinese

# Set komi
komi 7.5

# Set board size
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

## Advanced Rule Options

### Suicide Settings

```ini
# Forbid suicide (default)
allowSuicide = false

# Allow suicide (Tromp-Taylor style)
allowSuicide = true
```

### Ko Rules

```ini
# Simple Ko (only forbid immediate recapture)
koRule = SIMPLE

# Positional Super Ko (forbid any repeated position, regardless of player)
koRule = POSITIONAL

# Situational Super Ko (forbid repeated positions by same player)
koRule = SITUATIONAL
```

### Scoring Rules

```ini
# Area scoring (Chinese, AGA)
scoringRule = AREA

# Territory scoring (Japanese, Korean)
scoringRule = TERRITORY
```

### Tax Rules

Some rules have special scoring for seki areas:

```ini
# No tax
taxRule = NONE

# Seki has no points
taxRule = SEKI

# All eyes have no points
taxRule = ALL
```

---

## Multi-Rule Training

### KataGo's Advantage

KataGo uses a single model supporting multiple rules:

```python
def encode_rules(rules):
    """Encode rules as neural network input"""
    features = np.zeros(RULE_FEATURE_SIZE)

    # Scoring method
    features[0] = 1.0 if rules.scoring == 'area' else 0.0

    # Suicide
    features[1] = 1.0 if rules.allow_suicide else 0.0

    # Ko rule
    features[2:5] = encode_ko_rule(rules.ko)

    # Komi (normalized)
    features[5] = rules.komi / 15.0

    return features
```

### Rule-Aware Input

```
Neural network input contains:
- Board state (19×19×N)
- Rule feature vector (K dimensions)

This lets the same model understand different rules
```

---

## Rule Switching Examples

### Python Code

```python
from katago import KataGo

engine = KataGo(model_path="kata.bin.gz")

# Chinese rules analysis
result_cn = engine.analyze(
    moves=[("B", "Q4"), ("W", "D4")],
    rules="chinese",
    komi=7.5
)

# Japanese rules analysis (same position)
result_jp = engine.analyze(
    moves=[("B", "Q4"), ("W", "D4")],
    rules="japanese",
    komi=6.5
)

# Compare differences
print(f"Chinese rules Black win rate: {result_cn['winrate']:.1%}")
print(f"Japanese rules Black win rate: {result_jp['winrate']:.1%}")
```

### Rule Impact Analysis

```python
def compare_rules_impact(position, rules_list):
    """Compare impact of different rules on position evaluation"""
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

## Common Questions

### Different Rules Leading to Different Results

```
Same game, different rules may lead to different results:
- Scoring differences between area and territory
- Handling of seki areas
- Impact of passes
```

### Which Rules to Choose?

| Scenario | Recommended Rules |
|----------|-------------------|
| Beginners | Chinese (intuitive, no disputes) |
| Online games | Platform default (usually Chinese) |
| Nihon Ki-in | Japanese |
| Programming | Tromp-Taylor (simplest) |
| Chinese pro tournaments | Chinese |

### Do Models Need Rule-Specific Training?

KataGo's multi-rule model is already strong. But if using only one ruleset, consider:

```ini
# Fixed rules training (may slightly improve strength for specific rules)
rules = chinese
```

---

## Further Reading

- [KataGo Training Mechanism](../training) — Multi-rule training implementation
- [Integrate into Your Project](../../hands-on/integration) — API usage examples
- [Evaluation & Benchmarking](../evaluation) — Strength testing under different rules
