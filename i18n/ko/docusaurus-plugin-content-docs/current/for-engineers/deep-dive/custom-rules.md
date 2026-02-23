---
sidebar_position: 10
title: 사용자 정의 규칙과 변형
description: KataGo가 지원하는 바둑 규칙 세트, 변형 바둑판 및 사용자 정의 설정 상세
---

# 사용자 정의 규칙과 변형

이 문서는 KataGo가 지원하는 다양한 바둑 규칙, 바둑판 크기 변형 및 사용자 정의 규칙 설정 방법을 소개합니다.

---

## 규칙 세트 개요

### 주요 규칙 비교

| 규칙 세트 | 계가 방식 | 덤 | 자살수 | 패 복원 |
|--------|---------|------|--------|---------|
| **Chinese** | 계자법 | 7.5 | 금지 | 금지 |
| **Japanese** | 계목법 | 6.5 | 금지 | 금지 |
| **Korean** | 계목법 | 6.5 | 금지 | 금지 |
| **AGA** | 혼합 | 7.5 | 금지 | 금지 |
| **New Zealand** | 계자법 | 7 | 허용 | 금지 |
| **Tromp-Taylor** | 계자법 | 7.5 | 허용 | 금지 |

### KataGo 설정

```ini
# config.cfg
rules = chinese           # 규칙 세트
komi = 7.5               # 덤
boardXSize = 19          # 바둑판 너비
boardYSize = 19          # 바둑판 높이
```

---

## 중국 규칙 (Chinese)

### 특징

```
계가 방식: 계자법 (돌과 집 모두 계산)
덤: 7.5집
자살수: 금지
패 복원: 금지 (단순 규칙)
```

### 계자법 설명

```
최종 점수 = 내 돌 개수 + 내 집 개수

예시:
흑돌 120개 + 흑집 65점 = 185점
백돌 100개 + 백집 75점 + 덤 7.5 = 182.5점
흑 2.5점 승
```

### KataGo 설정

```ini
rules = chinese
komi = 7.5
```

---

## 일본 규칙 (Japanese)

### 특징

```
계가 방식: 계목법 (집만 계산)
덤: 6.5집
자살수: 금지
패 복원: 금지
사석 표시 필요
```

### 계목법 설명

```
최종 점수 = 내 집 개수 + 잡은 상대 돌 개수

예시:
흑집 65점 + 잡은 돌 10개 = 75점
백집 75점 + 잡은 돌 5개 + 덤 6.5 = 86.5점
백 11.5집 승
```

### 사석 판정

일본 규칙은 양측이 어떤 돌이 사석인지 합의해야 합니다:

```python
def is_dead_by_japanese_rules(group, game_state):
    """일본 규칙에서 사석 판정"""
    # 해당 돌 그룹이 두 눈을 만들 수 없음을 증명해야 함
    # 이것이 일본 규칙의 복잡한 부분
    pass
```

### KataGo 설정

```ini
rules = japanese
komi = 6.5
```

---

## AGA 규칙

### 특징

미국 바둑 협회(AGA) 규칙은 중일 규칙의 장점을 결합합니다:

```
계가 방식: 혼합 (계자 또는 계목 모두 가능, 결과 동일)
덤: 7.5집
자살수: 금지
백이 패스할 때 돌 하나 제출
```

### 패스 규칙

```
흑 패스: 돌 제출 불필요
백 패스: 흑에게 돌 하나 제출 필요

이로 인해 계자법과 계목법 결과가 일치함
```

### KataGo 설정

```ini
rules = aga
komi = 7.5
```

---

## Tromp-Taylor 규칙

### 특징

가장 간결한 바둑 규칙, 프로그램 구현에 적합:

```
계가 방식: 계자법
덤: 7.5집
자살수: 허용
패 복원: Super Ko (모든 반복 국면 금지)
사석 판정 불필요
```

### Super Ko

```python
def is_superko_violation(new_state, history):
    """Super Ko 위반 검사"""
    for past_state in history:
        if new_state == past_state:
            return True
    return False
```

### 종국 판정

```
양측 사석 합의 불필요
다음 시점까지 대국 계속:
1. 양측 연속 패스
2. 그 후 탐색 또는 실제로 두어서 영역 확정
```

### KataGo 설정

```ini
rules = tromp-taylor
komi = 7.5
```

---

## 바둑판 크기 변형

### 지원 크기

KataGo는 다양한 바둑판 크기를 지원합니다:

| 크기 | 특징 | 권장 용도 |
|------|------|---------|
| 9×9 | 약 81점 | 초보, 속기 |
| 13×13 | 약 169점 | 고급 학습 |
| 19×19 | 361점 | 표준 대국 |
| 사용자 정의 | 임의 | 연구, 테스트 |

### 설정 방법

```ini
# 9×9 바둑판
boardXSize = 9
boardYSize = 9
komi = 5.5

# 13×13 바둑판
boardXSize = 13
boardYSize = 13
komi = 6.5

# 비정사각형 바둑판
boardXSize = 19
boardYSize = 9
```

### 덤 권장

| 크기 | 중국 규칙 | 일본 규칙 |
|------|---------|---------|
| 9×9 | 5.5 | 5.5 |
| 13×13 | 6.5 | 6.5 |
| 19×19 | 7.5 | 6.5 |

---

## 접바둑 설정

### 접바둑

접바둑은 기력 차이를 조정하는 방법입니다:

```ini
# 2점 접바둑
handicap = 2

# 9점 접바둑
handicap = 9
```

### 접바둑 위치

```python
HANDICAP_POSITIONS = {
    2: [(3, 15), (15, 3)],
    3: [(3, 15), (15, 3), (15, 15)],
    4: [(3, 15), (15, 3), (3, 3), (15, 15)],
    # 5-9점은 화점 + 천원 사용
}
```

### 접바둑 시 덤

```ini
# 전통: 접바둑 시 덤 없음 또는 반집
komi = 0.5

# 현대: 접바둑 수에 따라 조정
# 각 점당 약 10-15집 가치
```

---

## Analysis 모드 규칙 설정

### GTP 명령

```gtp
# 규칙 설정
kata-set-rules chinese

# 덤 설정
komi 7.5

# 바둑판 크기 설정
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

## 고급 규칙 옵션

### 자살수 설정

```ini
# 자살 금지 (기본값)
allowSuicide = false

# 자살 허용 (Tromp-Taylor 스타일)
allowSuicide = true
```

### Ko 규칙

```ini
# Simple Ko (즉시 복원만 금지)
koRule = SIMPLE

# Positional Super Ko (모든 국면 반복 금지, 누구 차례든)
koRule = POSITIONAL

# Situational Super Ko (같은 측 차례의 국면 반복 금지)
koRule = SITUATIONAL
```

### 계가 규칙

```ini
# 계자법 (중국, AGA)
scoringRule = AREA

# 계목법 (일본, 한국)
scoringRule = TERRITORY
```

### 세금 규칙

일부 규칙은 쌍방생 영역에 특별한 계가 방법이 있습니다:

```ini
# 세금 없음
taxRule = NONE

# 쌍방생 무집
taxRule = SEKI

# 모든 눈 무집
taxRule = ALL
```

---

## 다중 규칙 학습

### KataGo의 장점

KataGo는 단일 모델로 여러 규칙을 지원합니다:

```python
def encode_rules(rules):
    """규칙을 신경망 입력으로 인코딩"""
    features = np.zeros(RULE_FEATURE_SIZE)

    # 계가 방식
    features[0] = 1.0 if rules.scoring == 'area' else 0.0

    # 자살수
    features[1] = 1.0 if rules.allow_suicide else 0.0

    # Ko 규칙
    features[2:5] = encode_ko_rule(rules.ko)

    # 덤 (정규화)
    features[5] = rules.komi / 15.0

    return features
```

### 규칙 인식 입력

```
신경망 입력 포함:
- 바둑판 상태 (19×19×N)
- 규칙 특성 벡터 (K차원)

이로 인해 같은 모델이 다른 규칙을 이해할 수 있음
```

---

## 규칙 전환 예제

### Python 코드

```python
from katago import KataGo

engine = KataGo(model_path="kata.bin.gz")

# 중국 규칙 분석
result_cn = engine.analyze(
    moves=[("B", "Q4"), ("W", "D4")],
    rules="chinese",
    komi=7.5
)

# 일본 규칙 분석 (같은 국면)
result_jp = engine.analyze(
    moves=[("B", "Q4"), ("W", "D4")],
    rules="japanese",
    komi=6.5
)

# 차이 비교
print(f"중국 규칙 흑 승률: {result_cn['winrate']:.1%}")
print(f"일본 규칙 흑 승률: {result_jp['winrate']:.1%}")
```

### 규칙 영향 분석

```python
def compare_rules_impact(position, rules_list):
    """다른 규칙이 국면 평가에 미치는 영향 비교"""
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

## 자주 묻는 질문

### 규칙 차이로 인한 승패 차이

```
같은 대국이 다른 규칙에서 다른 결과를 낼 수 있음:
- 계자법 vs 계목법의 계가 차이
- 쌍방생 영역의 처리
- 패스의 영향
```

### 어떤 규칙을 선택해야 하나?

| 시나리오 | 권장 규칙 |
|------|---------|
| 초보자 | Chinese (직관적, 분쟁 없음) |
| 온라인 대국 | 플랫폼 기본값 (보통 Chinese) |
| 일본 기원 | Japanese |
| 프로그램 구현 | Tromp-Taylor (가장 간결) |
| 중국 프로 대회 | Chinese |

### 특정 규칙에 맞게 모델을 학습해야 하나?

KataGo의 다중 규칙 모델은 이미 매우 강력합니다. 하지만 단일 규칙만 사용한다면 고려해볼 수 있습니다:

```ini
# 고정 규칙 학습 (특정 규칙에서 기력 약간 향상 가능)
rules = chinese
```

---

## 추가 읽기

- [KataGo 학습 메커니즘 분석](../training) — 다중 규칙 학습 구현
- [프로젝트에 통합하기](../../hands-on/integration) — API 사용 예제
- [평가 및 벤치마크 테스트](../evaluation) — 다른 규칙에서의 기력 테스트
