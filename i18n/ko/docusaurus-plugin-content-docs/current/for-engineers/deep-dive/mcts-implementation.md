---
sidebar_position: 5
title: MCTS 구현 세부사항
description: 몬테카를로 트리 탐색 구현, PUCT 선택 및 병렬화 기술 심층 분석
---

# MCTS 구현 세부사항

이 문서는 KataGo의 몬테카를로 트리 탐색(MCTS) 구현 세부사항을 데이터 구조, 선택 전략 및 병렬화 기술을 포함하여 심층적으로 분석합니다.

---

## MCTS 4단계 복습

```
┌─────────────────────────────────────────────────────┐
│                    MCTS 탐색 루프                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│   1. Selection      선택: 트리를 따라 내려가며 PUCT로 노드 선택  │
│         │                                           │
│         ▼                                           │
│   2. Expansion      확장: 리프 노드에 도달하면 자식 노드 생성    │
│         │                                           │
│         ▼                                           │
│   3. Evaluation     평가: 신경망으로 리프 노드 평가           │
│         │                                           │
│         ▼                                           │
│   4. Backprop       역전파: 경로상의 모든 노드 통계 업데이트    │
│                                                     │
│   수천 번 반복, 방문 횟수가 가장 많은 착수 선택               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 노드 데이터 구조

### 핵심 데이터

각 MCTS 노드는 다음을 저장해야 합니다:

```python
class MCTSNode:
    def __init__(self, state, parent=None, prior=0.0):
        # 기본 정보
        self.state = state              # 바둑판 상태
        self.parent = parent            # 부모 노드
        self.children = {}              # 자식 노드 딕셔너리 {action: node}
        self.action = None              # 이 노드에 도달한 착수

        # 통계 정보
        self.visit_count = 0            # N(s): 방문 횟수
        self.value_sum = 0.0            # W(s): 가치 합계
        self.prior = prior              # P(s,a): 사전 확률

        # 병렬 탐색용
        self.virtual_loss = 0           # 가상 손실
        self.is_expanded = False        # 확장 여부

    @property
    def value(self):
        """Q(s) = W(s) / N(s)"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
```

### 메모리 최적화

KataGo는 메모리 사용량을 줄이기 위해 다양한 기술을 사용합니다:

```python
# Python dict 대신 numpy 배열 사용
class OptimizedNode:
    __slots__ = ['visit_count', 'value_sum', 'prior', 'children_indices']

    def __init__(self):
        self.visit_count = np.int32(0)
        self.value_sum = np.float32(0.0)
        self.prior = np.float32(0.0)
        self.children_indices = None  # 지연 할당
```

---

## Selection: PUCT 선택

### PUCT 공식

```
선택 점수 = Q(s,a) + U(s,a)

여기서:
Q(s,a) = W(s,a) / N(s,a)              # 평균 가치
U(s,a) = c_puct × P(s,a) × √(N(s)) / (1 + N(s,a))  # 탐색 항
```

### 파라미터 설명

| 기호 | 의미 | 전형적인 값 |
|------|------|--------|
| Q(s,a) | 착수 a의 평균 가치 | [-1, +1] |
| P(s,a) | 신경망의 사전 확률 | [0, 1] |
| N(s) | 부모 노드 방문 횟수 | 정수 |
| N(s,a) | 착수 a의 방문 횟수 | 정수 |
| c_puct | 탐색 상수 | 1.0 ~ 2.5 |

### 구현

```python
def select_child(self, c_puct=1.5):
    """PUCT 점수가 가장 높은 자식 노드 선택"""
    best_score = -float('inf')
    best_action = None
    best_child = None

    # 부모 노드 방문 횟수의 제곱근
    sqrt_parent_visits = math.sqrt(self.visit_count)

    for action, child in self.children.items():
        # Q 값 (평균 가치)
        if child.visit_count > 0:
            q_value = child.value_sum / child.visit_count
        else:
            q_value = 0.0

        # U 값 (탐색 항)
        u_value = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)

        # 총 점수
        score = q_value + u_value

        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    return best_action, best_child
```

### 탐색 vs 활용의 균형

```
초기: N(s,a) 작음
├── U(s,a) 큼 → 탐색 위주
└── 높은 사전 확률의 착수가 먼저 탐색됨

후기: N(s,a) 큼
├── U(s,a) 작음 → 활용 위주
└── Q(s,a)가 주도, 이미 알려진 좋은 착수 선택
```

---

## Expansion: 노드 확장

### 확장 조건

리프 노드에 도달하면 신경망을 사용하여 확장합니다:

```python
def expand(self, policy_probs, legal_moves):
    """노드 확장, 모든 합법 착수에 대한 자식 노드 생성"""
    for action in legal_moves:
        if action not in self.children:
            prior = policy_probs[action]  # 신경망 예측 확률
            child_state = self.state.play(action)
            self.children[action] = MCTSNode(
                state=child_state,
                parent=self,
                prior=prior
            )

    self.is_expanded = True
```

### 합법 착수 필터링

```python
def get_legal_moves(state):
    """모든 합법 착수 가져오기"""
    legal = []
    for i in range(361):
        x, y = i // 19, i % 19
        if state.is_legal(x, y):
            legal.append(i)

    # 패스 추가
    legal.append(361)

    return legal
```

---

## Evaluation: 신경망 평가

### 단일 평가

```python
def evaluate(self, state):
    """신경망으로 국면 평가"""
    # 입력 특성 인코딩
    features = encode_state(state)  # (22, 19, 19)
    features = torch.tensor(features).unsqueeze(0)  # (1, 22, 19, 19)

    # 신경망 추론
    with torch.no_grad():
        output = self.network(features)

    policy = output['policy'][0].numpy()  # (362,)
    value = output['value'][0].item()     # scalar

    return policy, value
```

### 배치 평가 (핵심 최적화)

GPU는 배치 추론 시 효율이 가장 높습니다:

```python
class BatchedEvaluator:
    def __init__(self, network, batch_size=8):
        self.network = network
        self.batch_size = batch_size
        self.pending = []  # 평가 대기 중인 (state, callback) 리스트

    def request_evaluation(self, state, callback):
        """평가 요청, 배치가 차면 자동 실행"""
        self.pending.append((state, callback))

        if len(self.pending) >= self.batch_size:
            self.flush()

    def flush(self):
        """배치 평가 실행"""
        if not self.pending:
            return

        # 배치 입력 준비
        states = [s for s, _ in self.pending]
        features = torch.stack([encode_state(s) for s in states])

        # 배치 추론
        with torch.no_grad():
            outputs = self.network(features)

        # 결과 콜백
        for i, (_, callback) in enumerate(self.pending):
            policy = outputs['policy'][i].numpy()
            value = outputs['value'][i].item()
            callback(policy, value)

        self.pending.clear()
```

---

## Backpropagation: 역전파 업데이트

### 기본 역전파

```python
def backpropagate(self, value):
    """리프 노드에서 루트 노드로 역전파, 통계 정보 업데이트"""
    node = self

    while node is not None:
        node.visit_count += 1
        node.value_sum += value

        # 관점 교대: 상대의 가치는 반대
        value = -value

        node = node.parent
```

### 관점 교대의 중요성

```
흑 관점: value = +0.6 (흑 유리)

역전파 경로:
리프 노드 (흑 차례): value_sum += +0.6
    ↑
부모 노드 (백 차례): value_sum += -0.6  ← 백에게는 불리
    ↑
조부모 노드 (흑 차례): value_sum += +0.6
    ↑
...
```

---

## 병렬화: 가상 손실

### 문제

여러 스레드가 동시에 탐색할 때 같은 노드를 선택할 수 있습니다:

```
Thread 1: 노드 A 선택 (Q=0.6, N=100)
Thread 2: 노드 A 선택 (Q=0.6, N=100) ← 중복!
Thread 3: 노드 A 선택 (Q=0.6, N=100) ← 중복!
```

### 해결책: 가상 손실

노드 선택 시 먼저 "가상 손실"을 추가하여 다른 스레드가 선택하지 않도록 합니다:

```python
VIRTUAL_LOSS = 3  # 가상 손실 값

def select_with_virtual_loss(self):
    """가상 손실이 포함된 선택"""
    action, child = self.select_child()

    # 가상 손실 추가
    child.visit_count += VIRTUAL_LOSS
    child.value_sum -= VIRTUAL_LOSS  # 졌다고 가정

    return action, child

def backpropagate_with_virtual_loss(self, value):
    """역전파 시 가상 손실 제거"""
    node = self

    while node is not None:
        # 가상 손실 제거
        node.visit_count -= VIRTUAL_LOSS
        node.value_sum += VIRTUAL_LOSS

        # 정상 업데이트
        node.visit_count += 1
        node.value_sum += value

        value = -value
        node = node.parent
```

### 효과

```
Thread 1: 노드 A 선택, 가상 손실 추가
         A의 Q 값이 일시적으로 감소

Thread 2: 노드 B 선택 (A가 나빠 보이므로)

Thread 3: 노드 C 선택

→ 다른 스레드가 다른 분기 탐색, 효율성 향상
```

---

## 전체 탐색 구현

```python
class MCTS:
    def __init__(self, network, c_puct=1.5, num_simulations=800):
        self.network = network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.evaluator = BatchedEvaluator(network)

    def search(self, root_state):
        """MCTS 탐색 실행"""
        root = MCTSNode(root_state)

        # 루트 노드 확장
        policy, value = self.evaluate(root_state)
        legal_moves = get_legal_moves(root_state)
        root.expand(policy, legal_moves)

        # 시뮬레이션 실행
        for _ in range(self.num_simulations):
            node = root
            path = [node]

            # Selection: 트리를 따라 내려감
            while node.is_expanded and node.children:
                action, node = node.select_child(self.c_puct)
                path.append(node)

            # Expansion + Evaluation
            if not node.is_expanded:
                policy, value = self.evaluate(node.state)
                legal_moves = get_legal_moves(node.state)

                if legal_moves:
                    node.expand(policy, legal_moves)

            # Backpropagation
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += value
                value = -value

        # 방문 횟수가 가장 많은 착수 선택
        best_action = max(root.children.items(),
                         key=lambda x: x[1].visit_count)[0]

        return best_action

    def evaluate(self, state):
        features = encode_state(state)
        features = torch.tensor(features).unsqueeze(0)

        with torch.no_grad():
            output = self.network(features)

        return output['policy'][0].numpy(), output['value'][0].item()
```

---

## 고급 기술

### Dirichlet 노이즈

학습 시 루트 노드에 노이즈를 추가하여 탐색 증가:

```python
def add_dirichlet_noise(root, alpha=0.03, epsilon=0.25):
    """루트 노드에 Dirichlet 노이즈 추가"""
    noise = np.random.dirichlet([alpha] * len(root.children))

    for i, child in enumerate(root.children.values()):
        child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]
```

### 온도 파라미터

착수 선택의 무작위성 제어:

```python
def select_action_with_temperature(root, temperature=1.0):
    """방문 횟수와 온도에 따라 착수 선택"""
    visits = np.array([c.visit_count for c in root.children.values()])
    actions = list(root.children.keys())

    if temperature == 0:
        # 탐욕적 선택
        return actions[np.argmax(visits)]
    else:
        # 방문 횟수의 확률 분포에 따라 선택
        probs = visits ** (1 / temperature)
        probs = probs / probs.sum()
        return np.random.choice(actions, p=probs)
```

### 트리 재사용

새로운 수에서 이전 탐색 트리 재사용:

```python
def reuse_tree(root, action):
    """서브트리 재사용"""
    if action in root.children:
        new_root = root.children[action]
        new_root.parent = None
        return new_root
    else:
        return None  # 새 트리 생성 필요
```

---

## 성능 최적화 요약

| 기술 | 효과 |
|------|------|
| **배치 평가** | GPU 활용률 10% → 80%+ |
| **가상 손실** | 멀티스레드 효율 3-5배 향상 |
| **트리 재사용** | 콜드 스타트 감소, 30%+ 계산 절약 |
| **메모리 풀** | 메모리 할당 오버헤드 감소 |

---

## 추가 읽기

- [신경망 아키텍처 상세 분석](../neural-network) — 평가 함수의 출처
- [GPU 백엔드와 최적화](../gpu-optimization) — 배치 추론의 하드웨어 최적화
- [핵심 논문 가이드](../papers) — PUCT 공식의 이론적 기초
