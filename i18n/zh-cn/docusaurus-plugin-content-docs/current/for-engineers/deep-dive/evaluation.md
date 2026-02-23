---
sidebar_position: 9
title: 评估与基准测试
description: 围棋 AI 的 Elo 评分系统、对局测试与性能基准测试方法
---

# 评估与基准测试

本文介绍如何评估围棋 AI 的棋力与性能，包括 Elo 评分系统、对局测试方法与标准基准测试。

---

## Elo 评分系统

### 基本概念

Elo 评分是衡量相对棋力的标准方法：

```
预期胜率 E_A = 1 / (1 + 10^((R_B - R_A) / 400))

新 Elo = 旧 Elo + K × (实际结果 - 预期结果)
```

### Elo 差距与胜率对照

| Elo 差距 | 强者胜率 |
|---------|---------|
| 0 | 50% |
| 100 | 64% |
| 200 | 76% |
| 400 | 91% |
| 800 | 99% |

### 实现

```python
def expected_score(rating_a, rating_b):
    """计算 A 对 B 的预期得分"""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating, expected, actual, k=32):
    """更新 Elo 评分"""
    return rating + k * (actual - expected)

def calculate_elo_diff(wins, losses, draws):
    """从对战结果计算 Elo 差距"""
    total = wins + losses + draws
    win_rate = (wins + 0.5 * draws) / total

    if win_rate <= 0 or win_rate >= 1:
        return float('inf') if win_rate >= 1 else float('-inf')

    return 400 * math.log10(win_rate / (1 - win_rate))
```

---

## 对局测试

### 测试框架

```python
class MatchTester:
    def __init__(self, engine_a, engine_b):
        self.engine_a = engine_a
        self.engine_b = engine_b
        self.results = {'a_wins': 0, 'b_wins': 0, 'draws': 0}

    def run_match(self, num_games=400):
        """执行对战测试"""
        for i in range(num_games):
            # 交替先后手
            if i % 2 == 0:
                black, white = self.engine_a, self.engine_b
                a_is_black = True
            else:
                black, white = self.engine_b, self.engine_a
                a_is_black = False

            # 进行对局
            result = self.play_game(black, white)

            # 记录结果
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
        """进行一局对局"""
        game = Game()

        while not game.is_terminal():
            if game.current_player == 'black':
                move = black_engine.get_move(game.state)
            else:
                move = white_engine.get_move(game.state)

            game.play(move)

        return game.get_winner()
```

### 统计显著性

确保测试结果具有统计意义：

```python
from scipy import stats

def calculate_confidence_interval(wins, total, confidence=0.95):
    """计算胜率的置信区间"""
    p = wins / total
    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * math.sqrt(p * (1 - p) / total)

    return (p - margin, p + margin)

# 示例
wins, total = 220, 400
ci_low, ci_high = calculate_confidence_interval(wins, total)
print(f"胜率: {wins/total:.1%}, 95% CI: [{ci_low:.1%}, {ci_high:.1%}]")
```

### 建议测试局数

| 预期 Elo 差 | 建议局数 | 置信度 |
|------------|---------|--------|
| \>100 | 100 | 95% |
| 50-100 | 200 | 95% |
| 20-50 | 400 | 95% |
| \<20 | 1000+ | 95% |

---

## SPRT（序贯概率比检验）

### 概念

无需固定局数，根据累积结果动态决定是否停止：

```python
def sprt(wins, losses, elo0=0, elo1=10, alpha=0.05, beta=0.05):
    """
    序贯概率比检验

    elo0: 零假设的 Elo 差（通常为 0）
    elo1: 备择假设的 Elo 差（通常为 5-20）
    alpha: 假阳性率
    beta: 假阴性率
    """
    if wins + losses == 0:
        return 'continue'

    # 计算对数似然比
    p0 = expected_score(elo1, 0)  # H1 下的预期胜率
    p1 = expected_score(elo0, 0)  # H0 下的预期胜率

    llr = (
        wins * math.log(p0 / p1) +
        losses * math.log((1 - p0) / (1 - p1))
    )

    # 决策边界
    lower = math.log(beta / (1 - alpha))
    upper = math.log((1 - beta) / alpha)

    if llr <= lower:
        return 'reject'  # H0 被拒绝，新模型较差
    elif llr >= upper:
        return 'accept'  # H0 被接受，新模型较好
    else:
        return 'continue'  # 继续测试
```

---

## KataGo 基准测试

### 执行基准测试

```bash
# 基本测试
katago benchmark -model model.bin.gz

# 指定访问次数
katago benchmark -model model.bin.gz -v 1000

# 详细输出
katago benchmark -model model.bin.gz -v 1000 -t 8
```

### 输出解读

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

### 关键指标

| 指标 | 说明 | 良好数值 |
|------|------|---------|
| NN evals/sec | 神经网络评估速度 | >1000 |
| Playouts/sec | MCTS 模拟速度 | >2000 |
| GPU 利用率 | GPU 使用效率 | >80% |

---

## 棋力评估

### 人类棋力对照

| AI Elo | 人类棋力 |
|--------|---------|
| ~1500 | 业余 1 段 |
| ~2000 | 业余 5 段 |
| ~2500 | 职业初段 |
| ~3000 | 职业五段 |
| ~3500 | 世界冠军级 |
| ~4000+ | 超越人类 |

### 主要 AI 的 Elo

| AI | Elo（估计） |
|----|-----------|
| KataGo (最新) | ~5000 |
| AlphaGo Zero | ~5000 |
| Leela Zero | ~4500 |
| 绝艺 | ~4800 |

### 测试对照

```python
def estimate_human_rank(ai_model, test_positions):
    """估计 AI 相当于的人类棋力"""
    # 使用标准测试题目
    correct = 0
    for pos in test_positions:
        ai_move = ai_model.get_best_move(pos['state'])
        if ai_move == pos['best_move']:
            correct += 1

    accuracy = correct / len(test_positions)

    # 准确率对照表
    if accuracy > 0.9:
        return "职业级"
    elif accuracy > 0.7:
        return "业余 5 段+"
    elif accuracy > 0.5:
        return "业余 1-5 段"
    else:
        return "业余级以下"
```

---

## 性能监控

### 持续监控

```python
import time
import psutil
import GPUtil

class PerformanceMonitor:
    def __init__(self):
        self.metrics = []

    def sample(self):
        """采样当前性能指标"""
        gpus = GPUtil.getGPUs()

        self.metrics.append({
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_util': gpus[0].load * 100 if gpus else 0,
            'gpu_memory': gpus[0].memoryUsed if gpus else 0,
        })

    def report(self):
        """生成报告"""
        if not self.metrics:
            return

        avg_cpu = sum(m['cpu_percent'] for m in self.metrics) / len(self.metrics)
        avg_gpu = sum(m['gpu_util'] for m in self.metrics) / len(self.metrics)

        print(f"平均 CPU 使用率: {avg_cpu:.1f}%")
        print(f"平均 GPU 使用率: {avg_gpu:.1f}%")
```

### 性能瓶颈诊断

| 症状 | 可能原因 | 解决方案 |
|------|---------|---------|
| CPU 100%，GPU 低 | 搜索线程不足 | 增加 numSearchThreads |
| GPU 100%，输出慢 | 批量太小 | 增加 nnMaxBatchSize |
| 内存不足 | 模型太大 | 使用较小模型 |
| 速度不稳定 | 温度过高 | 改善散热 |

---

## 自动化测试

### CI/CD 集成

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

### 回归测试

```python
def regression_test(new_model, baseline_model, threshold=0.95):
    """检查新模型是否有性能回归"""
    # 测试准确率
    new_accuracy = test_accuracy(new_model)
    baseline_accuracy = test_accuracy(baseline_model)

    if new_accuracy < baseline_accuracy * threshold:
        raise Exception(f"准确率回归: {new_accuracy:.3f} < {baseline_accuracy:.3f}")

    # 测试速度
    new_speed = benchmark_speed(new_model)
    baseline_speed = benchmark_speed(baseline_model)

    if new_speed < baseline_speed * threshold:
        raise Exception(f"速度回归: {new_speed:.1f} < {baseline_speed:.1f}")

    print("回归测试通过")
```

---

## 延伸阅读

- [KataGo 训练机制解析](../training) — 模型是如何训练的
- [分布式训练架构](../distributed-training) — 大规模评估
- [GPU 后端与优化](../gpu-optimization) — 性能调优
