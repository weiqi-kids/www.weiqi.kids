---
sidebar_position: 11
title: 핵심 논문 가이드
description: AlphaGo, AlphaZero, KataGo 등 바둑 AI 이정표 논문의 핵심 분석
---

# 핵심 논문 가이드

이 문서는 바둑 AI 발전사에서 가장 중요한 논문들을 정리하여 빠른 이해를 위한 요약과 기술적 핵심을 제공합니다.

---

## 논문 개요

### 타임라인

```
2006  Coulom - MCTS 바둑에 최초 적용
2016  Silver et al. - AlphaGo (Nature)
2017  Silver et al. - AlphaGo Zero (Nature)
2017  Silver et al. - AlphaZero
2019  Wu - KataGo
2020+ 다양한 개선 및 응용
```

### 읽기 권장

| 목표 | 권장 논문 |
|------|---------|
| 기초 이해 | AlphaGo (2016) |
| 자가 대국 이해 | AlphaGo Zero (2017) |
| 범용 방법 이해 | AlphaZero (2017) |
| 구현 참조 | KataGo (2019) |

---

## 1. MCTS의 탄생 (2006)

### 논문 정보

```
제목: Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search
저자: Rémi Coulom
발표: Computers and Games 2006
```

### 핵심 기여

몬테카를로 방법을 바둑에 체계적으로 최초 적용:

```
이전: 순수 무작위 시뮬레이션, 트리 구조 없음
이후: 탐색 트리 구축 + UCB 선택 + 통계 역전파
```

### 핵심 개념

#### UCB1 공식

```
선택 점수 = 평균 승률 + C × √(ln(N) / n)

여기서:
- N: 부모 노드 방문 횟수
- n: 자식 노드 방문 횟수
- C: 탐색 상수
```

#### MCTS 4단계

```
1. Selection: UCB로 노드 선택
2. Expansion: 새 노드 확장
3. Simulation: 종국까지 무작위 시뮬레이션
4. Backpropagation: 승패 역전파
```

### 영향

- 바둑 AI가 아마추어 단급 수준에 도달
- 이후 모든 바둑 AI의 기초가 됨
- UCB 개념이 PUCT 발전에 영향

---

## 2. AlphaGo (2016)

### 논문 정보

```
제목: Mastering the game of Go with deep neural networks and tree search
저자: Silver, D., Huang, A., Maddison, C.J., et al.
발표: Nature, 2016
DOI: 10.1038/nature16961
```

### 핵심 기여

**딥러닝과 MCTS를 최초로 결합**, 인간 세계 챔피언 격파.

### 시스템 아키텍처

```
┌─────────────────────────────────────────────┐
│              AlphaGo 아키텍처                │
├─────────────────────────────────────────────┤
│                                             │
│   Policy Network (SL)                       │
│   ├── 입력: 바둑판 상태 (48개 특성 평면)      │
│   ├── 구조: 13층 CNN                        │
│   ├── 출력: 361개 위치의 확률                │
│   └── 학습: 3000만 인간 기보                 │
│                                             │
│   Policy Network (RL)                       │
│   ├── SL Policy에서 초기화                   │
│   └── 자가 대국 강화학습                     │
│                                             │
│   Value Network                             │
│   ├── 입력: 바둑판 상태                      │
│   ├── 출력: 단일 승률 값                     │
│   └── 학습: 자가 대국으로 생성된 국면         │
│                                             │
│   MCTS                                      │
│   ├── Policy Network로 탐색 가이드           │
│   └── Value Network + Rollout으로 평가       │
│                                             │
└─────────────────────────────────────────────┘
```

### 기술 요점

#### 1. 지도학습 Policy Network

```python
# 입력 특성 (48개 평면)
- 내 돌 위치
- 상대 돌 위치
- 활로 개수
- 잡은 후 상태
- 합법 착수 위치
- 최근 몇 수 위치
...
```

#### 2. 강화학습 개선

```
SL Policy → 자가 대국 → RL Policy

RL Policy가 SL Policy보다 약 80% 승률로 강함
```

#### 3. Value Network 학습

```
과적합 방지의 핵심:
- 각 대국에서 하나의 국면만 추출
- 유사 국면 중복 방지
```

#### 4. MCTS 통합

```
리프 노드 평가 = 0.5 × Value Network + 0.5 × Rollout

Rollout은 빠른 Policy Network 사용 (정확도 낮지만 속도 빠름)
```

### 주요 데이터

| 항목 | 수치 |
|------|------|
| SL Policy 정확도 | 57% |
| RL Policy 대 SL Policy 승률 | 80% |
| 학습 GPU | 176 |
| 대국 GPU | 48 TPU |

---

## 3. AlphaGo Zero (2017)

### 논문 정보

```
제목: Mastering the game of Go without human knowledge
저자: Silver, D., Schrittwieser, J., Simonyan, K., et al.
발표: Nature, 2017
DOI: 10.1038/nature24270
```

### 핵심 기여

**인간 기보 전혀 필요 없이** 처음부터 자가 학습.

### AlphaGo와의 차이점

| 측면 | AlphaGo | AlphaGo Zero |
|------|---------|--------------|
| 인간 기보 | 필요 | **불필요** |
| 네트워크 수 | 4개 | **1개 듀얼 헤드** |
| 입력 특성 | 48 평면 | **17 평면** |
| Rollout | 사용 | **사용 안 함** |
| 잔차 네트워크 | 없음 | **있음** |
| 학습 시간 | 수 개월 | **3일** |

### 핵심 혁신

#### 1. 단일 듀얼 헤드 네트워크

```
              입력 (17 평면)
                   │
              ┌────┴────┐
              │ 잔차 타워 │
              │ (19 또는 │
              │  39층)   │
              └────┬────┘
           ┌──────┴──────┐
           │             │
        Policy         Value
        (361)          (1)
```

#### 2. 단순화된 입력 특성

```python
# 17개 특성 평면만 필요
features = [
    current_player_stones,      # 내 돌
    opponent_stones,            # 상대 돌
    history_1_player,           # 히스토리 상태 1
    history_1_opponent,
    ...                         # 히스토리 상태 2-7
    color_to_play               # 누구 차례
]
```

#### 3. 순수 Value Network 평가

```
더 이상 Rollout 사용 안 함
리프 노드 평가 = Value Network 출력

더 간결하고 더 빠름
```

#### 4. 학습 흐름

```
무작위 네트워크 초기화
    │
    ▼
┌─────────────────────────────┐
│  자가 대국으로 기보 생성     │ ←─┐
└──────────────┬──────────────┘   │
               │                   │
               ▼                   │
┌─────────────────────────────┐   │
│  신경망 학습                 │   │
│  - Policy: 교차 엔트로피 최소화 │   │
│  - Value: MSE 최소화         │   │
└──────────────┬──────────────┘   │
               │                   │
               ▼                   │
┌─────────────────────────────┐   │
│  새 네트워크 평가            │   │
│  더 강하면 교체              │───┘
└─────────────────────────────┘
```

### 학습 곡선

```
학습 시간    Elo
─────────────────
3시간      초보자
24시간     AlphaGo Lee 초과
72시간     AlphaGo Master 초과
```

---

## 4. AlphaZero (2017)

### 논문 정보

```
제목: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
저자: Silver, D., Hubert, T., Schrittwieser, J., et al.
발표: arXiv:1712.01815 (이후 Science, 2018 발표)
```

### 핵심 기여

**범용화**: 동일 알고리즘을 바둑, 체스, 장기에 적용.

### 범용 아키텍처

```
입력 인코딩 (게임별) → 잔차 네트워크 (범용) → 듀얼 헤드 출력 (범용)
```

### 게임간 적응

| 게임 | 입력 평면 | 행동 공간 | 학습 시간 |
|------|---------|---------|---------|
| 바둑 | 17 | 362 | 40일 |
| 체스 | 119 | 4672 | 9시간 |
| 장기 | 362 | 11259 | 12시간 |

### MCTS 개선

#### PUCT 공식

```
선택 점수 = Q(s,a) + c(s) × P(s,a) × √N(s) / (1 + N(s,a))

c(s) = log((1 + N(s) + c_base) / c_base) + c_init
```

#### 탐색 노이즈

```python
# 루트 노드에 Dirichlet 노이즈 추가
P(s,a) = (1 - ε) × p_a + ε × η_a

η ~ Dir(α)
α = 0.03 (바둑), 0.3 (체스), 0.15 (장기)
```

---

## 5. KataGo (2019)

### 논문 정보

```
제목: Accelerating Self-Play Learning in Go
저자: David J. Wu
발표: arXiv:1902.10565
```

### 핵심 기여

**50배 효율 향상**, 개인 개발자도 강력한 바둑 AI를 학습할 수 있게 함.

### 핵심 혁신

#### 1. 보조 학습 목표

```
총 손실 = Policy Loss + Value Loss +
         Score Loss + Ownership Loss + ...

보조 목표가 네트워크의 빠른 수렴을 도움
```

#### 2. 글로벌 특성

```python
# 글로벌 풀링층
global_features = global_avg_pool(conv_features)
# 로컬 특성과 결합
combined = concat(conv_features, broadcast(global_features))
```

#### 3. Playout Cap 무작위화

```
전통: 매번 고정 N회 탐색
KataGo: N을 특정 분포에서 무작위 샘플링

네트워크가 다양한 탐색 깊이에서 잘 수행하도록 학습
```

#### 4. 점진적 바둑판 크기

```python
if training_step < 1000000:
    board_size = random.choice([9, 13, 19])
else:
    board_size = 19
```

### 효율 비교

| 지표 | AlphaZero | KataGo |
|------|-----------|--------|
| 초인 수준 도달 GPU일 | 5000 | **100** |
| 효율 향상 | 기준 | **50배** |

---

## 6. 확장 논문

### MuZero (2020)

```
제목: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model
기여: 환경 동적 모델 학습, 게임 규칙 불필요
```

### EfficientZero (2021)

```
제목: Mastering Atari Games with Limited Data
기여: 샘플 효율 대폭 향상
```

### Gumbel AlphaZero (2022)

```
제목: Policy Improvement by Planning with Gumbel
기여: 개선된 정책 개선 방법
```

---

## 논문 읽기 제안

### 입문 순서

```
1. AlphaGo (2016) - 기본 아키텍처 이해
2. AlphaGo Zero (2017) - 자가 대국 이해
3. KataGo (2019) - 구현 세부사항 이해
```

### 고급 순서

```
4. AlphaZero (2017) - 범용화
5. MuZero (2020) - 세계 모델 학습
6. MCTS 원본 논문 - 기초 이해
```

### 읽기 팁

1. **먼저 초록과 결론 보기**: 핵심 기여 빠르게 파악
2. **그림과 표 보기**: 전체 아키텍처 이해
3. **방법 섹션 보기**: 기술적 세부사항 이해
4. **부록 보기**: 구현 세부사항과 하이퍼파라미터 찾기

---

## 리소스 링크

### 논문 PDF

| 논문 | 링크 |
|------|------|
| AlphaGo | [Nature](https://www.nature.com/articles/nature16961) |
| AlphaGo Zero | [Nature](https://www.nature.com/articles/nature24270) |
| AlphaZero | [Science](https://www.science.org/doi/10.1126/science.aar6404) |
| KataGo | [arXiv](https://arxiv.org/abs/1902.10565) |

### 오픈소스 구현

| 프로젝트 | 링크 |
|------|------|
| KataGo | [GitHub](https://github.com/lightvector/KataGo) |
| Leela Zero | [GitHub](https://github.com/leela-zero/leela-zero) |
| MiniGo | [GitHub](https://github.com/tensorflow/minigo) |

---

## 추가 읽기

- [신경망 아키텍처 상세 분석](../neural-network) — 네트워크 설계 심층 이해
- [MCTS 구현 세부사항](../mcts-implementation) — 탐색 알고리즘 구현
- [KataGo 학습 메커니즘 분석](../training) — 학습 프로세스 상세
