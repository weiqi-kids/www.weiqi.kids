---
sidebar_position: 3
title: KataGo 학습 메커니즘 분석
description: KataGo의 자가 대국 학습 과정과 핵심 기술에 대한 심층 이해
---

# KataGo 학습 메커니즘 분석

이 문서는 KataGo의 학습 메커니즘을 심층적으로 분석하여 자가 대국 학습의 작동 원리를 이해하는 데 도움을 줍니다.

---

## 학습 개요

### 학습 루프

```
초기 모델 → 자가 대국 → 데이터 수집 → 학습 업데이트 → 더 강한 모델 → 반복
```

**애니메이션 대응**:
- E5 자가 대국 ↔ 고정점 수렴
- E6 기력 곡선 ↔ S 곡선 성장
- H1 MDP ↔ 마르코프 체인

### 하드웨어 요구사항

| 모델 규모 | GPU 메모리 | 학습 시간 |
|---------|-----------|---------|
| b6c96 | 4 GB | 수 시간 |
| b10c128 | 8 GB | 1-2일 |
| b18c384 | 16 GB | 1-2주 |
| b40c256 | 24 GB+ | 수 주 |

---

## 환경 설정

### 의존성 설치

```bash
# Python 환경
conda create -n katago python=3.10
conda activate katago

# PyTorch (CUDA 버전)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 기타 의존성
pip install numpy h5py tqdm tensorboard
```

### 학습 코드 가져오기

```bash
git clone https://github.com/lightvector/KataGo.git
cd KataGo/python
```

---

## 학습 설정

### 설정 파일 구조

```yaml
# configs/train_config.yaml

# 모델 아키텍처
model:
  num_blocks: 10          # 잔차 블록 수
  trunk_channels: 128     # 백본 채널 수
  policy_channels: 32     # Policy 헤드 채널 수
  value_channels: 32      # Value 헤드 채널 수

# 학습 파라미터
training:
  batch_size: 256
  learning_rate: 0.001
  lr_schedule: "cosine"
  weight_decay: 0.0001
  epochs: 100

# 자가 대국 파라미터
selfplay:
  num_games_per_iteration: 1000
  max_visits: 600
  temperature: 1.0
  temperature_drop_move: 20

# 데이터 설정
data:
  max_history_games: 500000
  shuffle_buffer_size: 100000
```

### 모델 규모 대조표

| 이름 | num_blocks | trunk_channels | 파라미터 수 |
|------|-----------|----------------|--------|
| b6c96 | 6 | 96 | ~1M |
| b10c128 | 10 | 128 | ~3M |
| b18c384 | 18 | 384 | ~20M |
| b40c256 | 40 | 256 | ~45M |

**애니메이션 대응**:
- F2 네트워크 크기 vs 기력: 용량 스케일링
- F6 신경 스케일링 법칙: 이중 로그 관계

---

## 학습 프로세스

### 단계 1: 모델 초기화

```python
# init_model.py
import torch
from model import KataGoModel

config = {
    'num_blocks': 10,
    'trunk_channels': 128,
    'input_features': 22,
    'policy_size': 362,  # 361 + pass
}

model = KataGoModel(config)
torch.save(model.state_dict(), 'model_init.pt')
print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
```

### 단계 2: 자가 대국으로 데이터 생성

```bash
# C++ 엔진 컴파일
cd ../cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=CUDA
make -j$(nproc)

# 자가 대국 실행
./katago selfplay \
  -model ../python/model_init.pt \
  -output-dir ../python/selfplay_data \
  -config selfplay.cfg \
  -num-games 1000
```

자가 대국 설정 (selfplay.cfg):

```ini
maxVisits = 600
numSearchThreads = 4

# 온도 설정 (탐색 증가)
chosenMoveTemperature = 1.0
chosenMoveTemperatureEarly = 1.0
chosenMoveTemperatureHalflife = 20

# Dirichlet 노이즈 (다양성 증가)
rootNoiseEnabled = true
rootDirichletNoiseTotalConcentration = 10.83
rootDirichletNoiseWeight = 0.25
```

**애니메이션 대응**:
- C3 탐색 vs 활용: 온도 파라미터
- E10 Dirichlet 노이즈: 루트 노드 탐색

### 단계 3: 신경망 학습

```python
# train.py
import torch
from torch.utils.data import DataLoader
from model import KataGoModel
from dataset import SelfPlayDataset

# 데이터 로드
dataset = SelfPlayDataset('selfplay_data/')
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# 모델 로드
model = KataGoModel(config)
model.load_state_dict(torch.load('model_init.pt'))
model = model.cuda()

# 옵티마이저
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

# 학습률 스케줄러
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=0.00001
)

# 학습 루프
for epoch in range(100):
    model.train()
    total_loss = 0

    for batch in dataloader:
        inputs = batch['inputs'].cuda()
        policy_target = batch['policy'].cuda()
        value_target = batch['value'].cuda()
        ownership_target = batch['ownership'].cuda()

        # 순전파
        policy_pred, value_pred, ownership_pred = model(inputs)

        # 손실 계산
        policy_loss = torch.nn.functional.cross_entropy(
            policy_pred, policy_target
        )
        value_loss = torch.nn.functional.mse_loss(
            value_pred, value_target
        )
        ownership_loss = torch.nn.functional.mse_loss(
            ownership_pred, ownership_target
        )

        loss = policy_loss + value_loss + 0.5 * ownership_loss

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")

    # 체크포인트 저장
    torch.save(model.state_dict(), f'model_epoch{epoch}.pt')
```

**애니메이션 대응**:
- D5 경사 하강: optimizer.step()
- K2 모멘텀: Adam 옵티마이저
- K4 학습률 감소: CosineAnnealingLR
- K5 그래디언트 클리핑: clip_grad_norm_

### 단계 4: 평가 및 반복

```bash
# 새 모델 vs 이전 모델 평가
./katago match \
  -model1 model_epoch99.pt \
  -model2 model_init.pt \
  -num-games 100 \
  -output match_results.txt
```

새 모델 승률 > 55%이면 이전 모델을 대체하고 다음 반복 진행.

---

## 손실 함수 상세

### Policy Loss

```python
# 교차 엔트로피 손실
policy_loss = -sum(target * log(pred))
```

목표: 예측 확률 분포가 MCTS 탐색 결과에 가깝도록.

**애니메이션 대응**:
- J1 정책 엔트로피: 교차 엔트로피
- J2 KL 발산: 분포 거리

### Value Loss

```python
# 평균 제곱 오차
value_loss = (pred - actual_result)^2
```

목표: 대국 최종 결과(승/패/무) 예측.

### Ownership Loss

```python
# 각 점 소유권 예측
ownership_loss = mean((pred - actual_ownership)^2)
```

목표: 각 위치의 최종 소유권 예측.

---

## 고급 기법

### 1. 데이터 증강

바둑판의 대칭성 활용:

```python
def augment_data(board, policy, ownership):
    """D4 군의 8가지 변환으로 데이터 증강"""
    augmented = []

    for rotation in range(4):
        for flip in [False, True]:
            # 회전 및 뒤집기
            aug_board = transform(board, rotation, flip)
            aug_policy = transform(policy, rotation, flip)
            aug_ownership = transform(ownership, rotation, flip)
            augmented.append((aug_board, aug_policy, aug_ownership))

    return augmented
```

**애니메이션 대응**:
- A9 바둑판 대칭성: D4 군
- L4 데이터 증강: 대칭성 활용

### 2. 커리큘럼 학습

쉬운 것부터 복잡한 것으로:

```python
# 먼저 적은 탐색 횟수로 학습
schedule = [
    (100, 10000),   # 100 visits, 10000 games
    (200, 20000),   # 200 visits, 20000 games
    (400, 50000),   # 400 visits, 50000 games
    (600, 100000),  # 600 visits, 100000 games
]
```

**애니메이션 대응**:
- E12 학습 커리큘럼: 커리큘럼 학습

### 3. 혼합 정밀도 학습

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    policy_pred, value_pred, ownership_pred = model(inputs)
    loss = compute_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 4. 다중 GPU 학습

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# 분산 초기화
dist.init_process_group(backend='nccl')

# 모델 래핑
model = DistributedDataParallel(model)
```

---

## 모니터링과 디버깅

### TensorBoard 모니터링

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/training')

# 손실 기록
writer.add_scalar('Loss/policy', policy_loss, step)
writer.add_scalar('Loss/value', value_loss, step)
writer.add_scalar('Loss/total', total_loss, step)

# 학습률 기록
writer.add_scalar('LR', scheduler.get_last_lr()[0], step)
```

```bash
tensorboard --logdir runs
```

### 일반적인 문제

| 문제 | 가능한 원인 | 해결 방법 |
|------|---------|---------|
| 손실이 줄지 않음 | 학습률이 너무 낮거나 높음 | 학습률 조정 |
| 손실 진동 | 배치 크기가 너무 작음 | 배치 크기 증가 |
| 과적합 | 데이터 부족 | 더 많은 자가 대국 데이터 생성 |
| 기력 향상 없음 | 탐색 횟수 부족 | maxVisits 증가 |

**애니메이션 대응**:
- L1 과적합: 과잉 적응
- L2 정규화: weight_decay
- D6 학습률 효과: 튜닝

---

## 소규모 실험 제안

실험만 하려면 다음을 권장합니다:

1. **9×9 바둑판 사용**: 계산량 대폭 감소
2. **소형 모델 사용**: b6c96로 충분
3. **탐색 횟수 감소**: 100-200 visits
4. **사전 학습 모델 미세 조정**: 처음부터 시작하는 것보다 빠름

```bash
# 9×9 바둑판 설정
boardSize = 9
maxVisits = 100
```

---

## 추가 읽기

- [소스 코드 가이드](../source-code) — 코드 구조 이해
- [오픈소스 커뮤니티 참여](../contributing) — 분산 학습 참여
- [KataGo의 핵심 혁신](../../how-it-works/katago-innovations) — 50배 효율의 비밀
