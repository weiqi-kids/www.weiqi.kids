---
sidebar_position: 9
title: 평가 및 벤치마크 테스트
description: 바둑 AI의 Elo 레이팅 시스템, 대국 테스트 및 성능 벤치마크 방법
---

# 평가 및 벤치마크 테스트

이 문서는 바둑 AI의 기력과 성능을 평가하는 방법을 소개하며, Elo 레이팅 시스템, 대국 테스트 방법 및 표준 벤치마크 테스트를 다룹니다.

---

## Elo 레이팅 시스템

### 기본 개념

Elo 레이팅은 상대적 기력을 측정하는 표준 방법입니다:

```
예상 승률 E_A = 1 / (1 + 10^((R_B - R_A) / 400))

새 Elo = 이전 Elo + K × (실제 결과 - 예상 결과)
```

### Elo 차이와 승률 대조

| Elo 차이 | 강자 승률 |
|---------|---------|
| 0 | 50% |
| 100 | 64% |
| 200 | 76% |
| 400 | 91% |
| 800 | 99% |

### 구현

```python
def expected_score(rating_a, rating_b):
    """A 대 B의 예상 점수 계산"""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating, expected, actual, k=32):
    """Elo 레이팅 업데이트"""
    return rating + k * (actual - expected)

def calculate_elo_diff(wins, losses, draws):
    """대전 결과로 Elo 차이 계산"""
    total = wins + losses + draws
    win_rate = (wins + 0.5 * draws) / total

    if win_rate <= 0 or win_rate >= 1:
        return float('inf') if win_rate >= 1 else float('-inf')

    return 400 * math.log10(win_rate / (1 - win_rate))
```

---

## 대국 테스트

### 테스트 프레임워크

```python
class MatchTester:
    def __init__(self, engine_a, engine_b):
        self.engine_a = engine_a
        self.engine_b = engine_b
        self.results = {'a_wins': 0, 'b_wins': 0, 'draws': 0}

    def run_match(self, num_games=400):
        """대전 테스트 실행"""
        for i in range(num_games):
            # 선후수 교대
            if i % 2 == 0:
                black, white = self.engine_a, self.engine_b
                a_is_black = True
            else:
                black, white = self.engine_b, self.engine_a
                a_is_black = False

            # 대국 진행
            result = self.play_game(black, white)

            # 결과 기록
            if result == 'black':
                if a_is_black:
                    self.results['a_wins'] += 1
                else:
                    self.results['b_wins'] += 1
            elif result == 'white':
                if a_is_black:
                    self.results['b_wins'] += 1
                else:
                    self.results['a_wins'] += 1
            else:
                self.results['draws'] += 1

        return self.results

    def play_game(self, black_engine, white_engine):
        """한 판 대국 진행"""
        game = Game()

        while not game.is_terminal():
            if game.current_player == 'black':
                move = black_engine.get_move(game.state)
            else:
                move = white_engine.get_move(game.state)

            game.play(move)

        return game.get_winner()
```

### 통계적 유의성

테스트 결과가 통계적으로 의미있도록 보장:

```python
from scipy import stats

def calculate_confidence_interval(wins, total, confidence=0.95):
    """승률의 신뢰 구간 계산"""
    p = wins / total
    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * math.sqrt(p * (1 - p) / total)

    return (p - margin, p + margin)

# 예제
wins, total = 220, 400
ci_low, ci_high = calculate_confidence_interval(wins, total)
print(f"승률: {wins/total:.1%}, 95% CI: [{ci_low:.1%}, {ci_high:.1%}]")
```

### 권장 테스트 대국 수

| 예상 Elo 차이 | 권장 대국 수 | 신뢰도 |
|------------|---------|--------|
| \>100 | 100 | 95% |
| 50-100 | 200 | 95% |
| 20-50 | 400 | 95% |
| \<20 | 1000+ | 95% |

---

## SPRT (순차 확률비 검정)

### 개념

고정된 대국 수 없이 누적 결과에 따라 동적으로 중단 결정:

```python
def sprt(wins, losses, elo0=0, elo1=10, alpha=0.05, beta=0.05):
    """
    순차 확률비 검정

    elo0: 귀무 가설의 Elo 차이 (보통 0)
    elo1: 대립 가설의 Elo 차이 (보통 5-20)
    alpha: 거짓 양성률
    beta: 거짓 음성률
    """
    if wins + losses == 0:
        return 'continue'

    # 로그 우도비 계산
    p0 = expected_score(elo1, 0)  # H1에서의 예상 승률
    p1 = expected_score(elo0, 0)  # H0에서의 예상 승률

    llr = (
        wins * math.log(p0 / p1) +
        losses * math.log((1 - p0) / (1 - p1))
    )

    # 결정 경계
    lower = math.log(beta / (1 - alpha))
    upper = math.log((1 - beta) / alpha)

    if llr <= lower:
        return 'reject'  # H0 기각, 새 모델이 더 약함
    elif llr >= upper:
        return 'accept'  # H0 채택, 새 모델이 더 강함
    else:
        return 'continue'  # 테스트 계속
```

---

## KataGo 벤치마크 테스트

### 벤치마크 실행

```bash
# 기본 테스트
katago benchmark -model model.bin.gz

# 방문 횟수 지정
katago benchmark -model model.bin.gz -v 1000

# 상세 출력
katago benchmark -model model.bin.gz -v 1000 -t 8
```

### 출력 해석

```
KataGo Benchmark Results
========================

Configuration:
  Model: kata-b18c384.bin.gz
  Backend: CUDA
  Threads: 8
  Visits: 1000

Performance:
  NN evals/second: 2847.3
  Playouts/second: 4521.8
  Avg time per move: 0.221 seconds

Memory:
  GPU memory usage: 2.1 GB
  System memory: 1.3 GB

Quality metrics:
  Policy accuracy: 0.612
  Value accuracy: 0.891
```

### 주요 지표

| 지표 | 설명 | 좋은 수치 |
|------|------|---------|
| NN evals/sec | 신경망 평가 속도 | >1000 |
| Playouts/sec | MCTS 시뮬레이션 속도 | >2000 |
| GPU 활용률 | GPU 사용 효율 | >80% |

---

## 기력 평가

### 인간 기력 대조

| AI Elo | 인간 기력 |
|--------|---------|
| ~1500 | 아마 1단 |
| ~2000 | 아마 5단 |
| ~2500 | 프로 초단 |
| ~3000 | 프로 5단 |
| ~3500 | 세계 챔피언급 |
| ~4000+ | 인간 초월 |

### 주요 AI의 Elo

| AI | Elo (추정) |
|----|-----------|
| KataGo (최신) | ~5000 |
| AlphaGo Zero | ~5000 |
| Leela Zero | ~4500 |
| 절예 | ~4800 |

### 테스트 대조

```python
def estimate_human_rank(ai_model, test_positions):
    """AI가 해당하는 인간 기력 추정"""
    # 표준 테스트 문제 사용
    correct = 0
    for pos in test_positions:
        ai_move = ai_model.get_best_move(pos['state'])
        if ai_move == pos['best_move']:
            correct += 1

    accuracy = correct / len(test_positions)

    # 정확도 대조표
    if accuracy > 0.9:
        return "프로급"
    elif accuracy > 0.7:
        return "아마 5단+"
    elif accuracy > 0.5:
        return "아마 1-5단"
    else:
        return "아마급 이하"
```

---

## 성능 모니터링

### 지속적 모니터링

```python
import time
import psutil
import GPUtil

class PerformanceMonitor:
    def __init__(self):
        self.metrics = []

    def sample(self):
        """현재 성능 지표 샘플링"""
        gpus = GPUtil.getGPUs()

        self.metrics.append({
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_util': gpus[0].load * 100 if gpus else 0,
            'gpu_memory': gpus[0].memoryUsed if gpus else 0,
        })

    def report(self):
        """리포트 생성"""
        if not self.metrics:
            return

        avg_cpu = sum(m['cpu_percent'] for m in self.metrics) / len(self.metrics)
        avg_gpu = sum(m['gpu_util'] for m in self.metrics) / len(self.metrics)

        print(f"평균 CPU 사용률: {avg_cpu:.1f}%")
        print(f"평균 GPU 사용률: {avg_gpu:.1f}%")
```

### 성능 병목 진단

| 증상 | 가능한 원인 | 해결 방안 |
|------|---------|---------|
| CPU 100%, GPU 낮음 | 탐색 스레드 부족 | numSearchThreads 증가 |
| GPU 100%, 출력 느림 | 배치 너무 작음 | nnMaxBatchSize 증가 |
| 메모리 부족 | 모델 너무 큼 | 더 작은 모델 사용 |
| 속도 불안정 | 온도 과열 | 냉각 개선 |

---

## 자동화 테스트

### CI/CD 통합

```yaml
# .github/workflows/benchmark.yml
name: Benchmark

on:
  push:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Run benchmark
        run: |
          ./katago benchmark -model model.bin.gz -v 500 > results.txt

      - name: Check performance
        run: |
          playouts=$(grep "Playouts/second" results.txt | awk '{print $2}')
          if (( $(echo "$playouts < 1000" | bc -l) )); then
            echo "Performance regression detected!"
            exit 1
          fi
```

### 회귀 테스트

```python
def regression_test(new_model, baseline_model, threshold=0.95):
    """새 모델의 성능 회귀 검사"""
    # 정확도 테스트
    new_accuracy = test_accuracy(new_model)
    baseline_accuracy = test_accuracy(baseline_model)

    if new_accuracy < baseline_accuracy * threshold:
        raise Exception(f"정확도 회귀: {new_accuracy:.3f} < {baseline_accuracy:.3f}")

    # 속도 테스트
    new_speed = benchmark_speed(new_model)
    baseline_speed = benchmark_speed(baseline_model)

    if new_speed < baseline_speed * threshold:
        raise Exception(f"속도 회귀: {new_speed:.1f} < {baseline_speed:.1f}")

    print("회귀 테스트 통과")
```

---

## 추가 읽기

- [KataGo 학습 메커니즘 분석](../training) — 모델이 어떻게 학습되는지
- [분산 학습 아키텍처](../distributed-training) — 대규모 평가
- [GPU 백엔드와 최적화](../gpu-optimization) — 성능 튜닝
