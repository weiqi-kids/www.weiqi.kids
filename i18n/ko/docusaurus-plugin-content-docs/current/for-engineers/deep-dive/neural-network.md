---
sidebar_position: 4
title: 신경망 아키텍처 상세 분석
description: KataGo의 신경망 설계, 입력 특성 및 다중 헤드 출력 아키텍처 심층 분석
---

# 신경망 아키텍처 상세 분석

이 문서는 KataGo 신경망의 전체 아키텍처를 입력 특성 인코딩부터 다중 헤드 출력 설계까지 심층적으로 분석합니다.

---

## 아키텍처 개요

KataGo는 **단일 신경망, 다중 헤드 출력** 설계를 사용합니다:

```
입력 특성 (19×19×22)
        │
        ▼
┌───────────────────┐
│     초기 합성곱층    │
│   256 filters     │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│     잔차 타워       │
│  20-60개 잔차 블록  │
│  + 글로벌 풀링층    │
└─────────┬─────────┘
          │
    ┌─────┴─────┬─────────┬─────────┐
    │           │         │         │
    ▼           ▼         ▼         ▼
 Policy      Value     Score   Ownership
  Head       Head      Head      Head
    │           │         │         │
    ▼           ▼         ▼         ▼
362 확률    승률      집 차이    361 소유권
(pass 포함)  (-1~+1)    (집)     (-1~+1)
```

---

## 입력 특성 인코딩

### 특성 평면 개요

KataGo는 **22개의 특성 평면** (19×19×22)을 사용하며, 각 평면은 19×19 행렬입니다:

| 평면 | 내용 | 설명 |
|------|------|------|
| 0 | 내 돌 | 1 = 내 돌 있음, 0 = 없음 |
| 1 | 상대 돌 | 1 = 상대 돌 있음, 0 = 없음 |
| 2 | 빈 점 | 1 = 빈 점, 0 = 돌 있음 |
| 3-10 | 히스토리 상태 | 과거 8수의 바둑판 변화 |
| 11 | 패 금지점 | 1 = 패 금지, 0 = 둘 수 있음 |
| 12-17 | 활로 인코딩 | 1활, 2활, 3활... 돌 그룹 |
| 18-21 | 규칙 인코딩 | 중국/일본 규칙, 덤 등 |

### 히스토리 상태 스택

신경망이 국면의 **동적 변화**를 이해할 수 있도록 KataGo는 과거 8수의 바둑판 상태를 스택합니다:

```python
# 히스토리 상태 인코딩 (개념)
def encode_history(game_history, current_player):
    features = []

    for t in range(8):  # 과거 8수
        if t < len(game_history):
            board = game_history[-(t+1)]
            # 해당 시점의 내 돌/상대 돌 인코딩
            features.append(encode_board(board, current_player))
        else:
            # 히스토리 부족, 0으로 채움
            features.append(np.zeros((19, 19)))

    return np.stack(features, axis=0)
```

### 규칙 인코딩

KataGo는 다양한 규칙을 지원하며, 특성 평면을 통해 신경망에 알립니다:

```python
# 규칙 인코딩 (개념)
def encode_rules(rules, komi):
    rule_features = np.zeros((4, 19, 19))

    # 규칙 유형 (one-hot)
    if rules == "chinese":
        rule_features[0] = 1.0
    elif rules == "japanese":
        rule_features[1] = 1.0

    # Komi 정규화
    normalized_komi = komi / 15.0  # [-1, 1]로 정규화
    rule_features[2] = normalized_komi

    # 현재 플레이어
    rule_features[3] = 1.0 if current_player == BLACK else 0.0

    return rule_features
```

---

## 백본 네트워크: 잔차 타워

### 잔차 블록 구조

KataGo는 **Pre-activation ResNet** 구조를 사용합니다:

```
입력 x
    │
    ├────────────────────┐
    │                    │
    ▼                    │
BatchNorm                │
    │                    │
    ▼                    │
ReLU                     │
    │                    │
    ▼                    │
Conv 3×3                 │
    │                    │
    ▼                    │
BatchNorm                │
    │                    │
    ▼                    │
ReLU                     │
    │                    │
    ▼                    │
Conv 3×3                 │
    │                    │
    ▼                    │
    +  ←─────────────────┘ (잔차 연결)
    │
    ▼
출력
```

### 코드 예제

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        return out + residual  # 잔차 연결
```

### 글로벌 풀링층

KataGo의 핵심 혁신 중 하나: 잔차 블록에 **글로벌 풀링**을 추가하여 네트워크가 전역 정보를 볼 수 있게 합니다:

```python
class GlobalPoolingBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.fc = nn.Linear(channels, channels)

    def forward(self, x):
        # 로컬 경로
        local = self.conv(x)

        # 글로벌 경로
        global_pool = x.mean(dim=[2, 3])  # 글로벌 평균 풀링
        global_fc = self.fc(global_pool)
        global_broadcast = global_fc.unsqueeze(2).unsqueeze(3)
        global_broadcast = global_broadcast.expand(-1, -1, 19, 19)

        # 융합
        return local + global_broadcast
```

**왜 글로벌 풀링이 필요한가?**

전통적인 합성곱은 로컬만 봅니다 (3×3 수용 영역), 많은 층을 쌓아도 전역 정보에 대한 인식은 제한적입니다. 글로벌 풀링은 네트워크가 직접 "볼 수 있게" 합니다:
- 전체 바둑판의 돌 수 차이
- 전역적 세력 분포
- 전체적인 형세 판단

---

## 출력 헤드 설계

### Policy Head (정책 헤드)

각 위치의 착수 확률을 출력합니다:

```python
class PolicyHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2, 1)  # 1×1 합성곱
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2 * 19 * 19, 362)  # 361 + pass

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.softmax(out, dim=1)  # 확률 분포
```

**출력 형식**: 362차원 벡터
- 인덱스 0-360: 바둑판 361개 위치의 착수 확률
- 인덱스 361: 패스 확률

### Value Head (가치 헤드)

현재 국면의 승률을 출력합니다:

```python
class ValueHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(19 * 19, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.tanh(self.fc2(out))  # -1에서 +1 출력
        return out
```

**출력 형식**: 단일 값 [-1, +1]
- +1: 내가 필승
- -1: 상대 필승
- 0: 균형

### Score Head (집수 헤드)

KataGo 고유의 최종 집수 차이 예측:

```python
class ScoreHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(19 * 19, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)  # 제한 없는 출력
        return out
```

**출력 형식**: 단일 값 (집수)
- 양수: 내가 앞섬
- 음수: 상대가 앞섬

### Ownership Head (영역 헤드)

각 점의 최종 소유권을 예측합니다:

```python
class OwnershipHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 1)
        self.bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        out = F.relu(self.bn(self.conv1(x)))
        out = torch.tanh(self.conv2(out))  # 각 점 -1에서 +1
        return out.view(out.size(0), -1)  # 361로 평탄화
```

**출력 형식**: 361차원 벡터, 각 값은 [-1, +1]
- +1: 해당 점은 내 영역
- -1: 해당 점은 상대 영역
- 0: 중립 또는 분쟁 지역

---

## AlphaZero와의 차이점

| 측면 | AlphaZero | KataGo |
|------|-----------|--------|
| **출력 헤드** | 2개 (Policy + Value) | **4개** (+ Score + Ownership) |
| **글로벌 풀링** | 없음 | **있음** |
| **입력 특성** | 17 평면 | **22 평면** (규칙 인코딩 포함) |
| **잔차 블록** | 표준 ResNet | **Pre-activation + 글로벌 풀링** |
| **다중 규칙 지원** | 없음 | **있음** (특성 인코딩 통해) |

---

## 모델 규모

KataGo는 다양한 규모의 모델을 제공합니다:

| 모델 | 잔차 블록 수 | 채널 수 | 파라미터 수 | 적용 시나리오 |
|------|---------|--------|--------|---------|
| b10c128 | 10 | 128 | ~5M | CPU, 빠른 테스트 |
| b18c384 | 18 | 384 | ~75M | 일반 GPU |
| b40c256 | 40 | 256 | ~95M | 고급 GPU |
| b60c320 | 60 | 320 | ~200M | 최고급 GPU |

**명명 규칙**: `b{잔차블록수}c{채널수}`

---

## 전체 네트워크 구현

```python
class KataGoNetwork(nn.Module):
    def __init__(self, num_blocks=18, channels=384):
        super().__init__()

        # 초기 합성곱
        self.initial_conv = nn.Conv2d(22, channels, 3, padding=1)
        self.initial_bn = nn.BatchNorm2d(channels)

        # 잔차 타워
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_blocks)
        ])

        # 글로벌 풀링 블록 (매 몇 개 잔차 블록마다 하나 삽입)
        self.global_pooling_blocks = nn.ModuleList([
            GlobalPoolingBlock(channels) for _ in range(num_blocks // 6)
        ])

        # 출력 헤드
        self.policy_head = PolicyHead(channels)
        self.value_head = ValueHead(channels)
        self.score_head = ScoreHead(channels)
        self.ownership_head = OwnershipHead(channels)

    def forward(self, x):
        # 초기 합성곱
        out = F.relu(self.initial_bn(self.initial_conv(x)))

        # 잔차 타워
        gp_idx = 0
        for i, block in enumerate(self.residual_blocks):
            out = block(out)

            # 매 6개 잔차 블록 후 글로벌 풀링 삽입
            if (i + 1) % 6 == 0 and gp_idx < len(self.global_pooling_blocks):
                out = self.global_pooling_blocks[gp_idx](out)
                gp_idx += 1

        # 출력 헤드
        policy = self.policy_head(out)
        value = self.value_head(out)
        score = self.score_head(out)
        ownership = self.ownership_head(out)

        return {
            'policy': policy,
            'value': value,
            'score': score,
            'ownership': ownership
        }
```

---

## 추가 읽기

- [MCTS 구현 세부사항](../mcts-implementation) — 탐색과 신경망의 결합
- [KataGo 학습 메커니즘 분석](../training) — 네트워크 학습 방법
- [핵심 논문 가이드](../papers) — 원본 논문의 수학적 유도
