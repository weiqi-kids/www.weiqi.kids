---
sidebar_position: 9
title: 評估與基準測試
description: 圍棋 AI 嘅 Elo 評分系統、對局測試同效能基準測試方法
---

# 評估與基準測試

本文介紹點樣評估圍棋 AI 嘅棋力同效能，包括 Elo 評分系統、對局測試方法同標準基準測試。

---

## Elo 評分系統

### 基本概念

Elo 評分係衡量相對棋力嘅標準方法：

```
預期勝率 E_A = 1 / (1 + 10^((R_B - R_A) / 400))

新 Elo = 舊 Elo + K × (實際結果 - 預期結果)
```

### Elo 差距同勝率對照

| Elo 差距 | 強者勝率 |
|---------|---------|
| 0 | 50% |
| 100 | 64% |
| 200 | 76% |
| 400 | 91% |
| 800 | 99% |

### 實作

```python
def expected_score(rating_a, rating_b):
    """計算 A 對 B 嘅預期得分"""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating, expected, actual, k=32):
    """更新 Elo 評分"""
    return rating + k * (actual - expected)

def calculate_elo_diff(wins, losses, draws):
    """由對戰結果計算 Elo 差距"""
    total = wins + losses + draws
    win_rate = (wins + 0.5 * draws) / total

    if win_rate <= 0 or win_rate >= 1:
        return float('inf') if win_rate >= 1 else float('-inf')

    return 400 * math.log10(win_rate / (1 - win_rate))
```

---

## 對局測試

### 測試框架

```python
class MatchTester:
    def __init__(self, engine_a, engine_b):
        self.engine_a = engine_a
        self.engine_b = engine_b
        self.results = {'a_wins': 0, 'b_wins': 0, 'draws': 0}

    def run_match(self, num_games=400):
        """執行對戰測試"""
        for i in range(num_games):
            # 交替先後手
            if i % 2 == 0:
                black, white = self.engine_a, self.engine_b
                a_is_black = True
            else:
                black, white = self.engine_b, self.engine_a
                a_is_black = False

            # 進行對局
            result = self.play_game(black, white)

            # 記錄結果
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
        """進行一局對局"""
        game = Game()

        while not game.is_terminal():
            if game.current_player == 'black':
                move = black_engine.get_move(game.state)
            else:
                move = white_engine.get_move(game.state)

            game.play(move)

        return game.get_winner()
```

### 統計顯著性

確保測試結果有統計意義：

```python
from scipy import stats

def calculate_confidence_interval(wins, total, confidence=0.95):
    """計算勝率嘅信賴區間"""
    p = wins / total
    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * math.sqrt(p * (1 - p) / total)

    return (p - margin, p + margin)

# 範例
wins, total = 220, 400
ci_low, ci_high = calculate_confidence_interval(wins, total)
print(f"勝率: {wins/total:.1%}, 95% CI: [{ci_low:.1%}, {ci_high:.1%}]")
```

### 建議測試局數

| 預期 Elo 差 | 建議局數 | 信賴度 |
|------------|---------|--------|
| \>100 | 100 | 95% |
| 50-100 | 200 | 95% |
| 20-50 | 400 | 95% |
| \<20 | 1000+ | 95% |

---

## SPRT（序貫機率比檢定）

### 概念

唔使固定局數，根據累積結果動態決定係咪停止：

```python
def sprt(wins, losses, elo0=0, elo1=10, alpha=0.05, beta=0.05):
    """
    序貫機率比檢定

    elo0: 虛無假設嘅 Elo 差（通常為 0）
    elo1: 對立假設嘅 Elo 差（通常為 5-20）
    alpha: 假陽性率
    beta: 假陰性率
    """
    if wins + losses == 0:
        return 'continue'

    # 計算對數似然比
    p0 = expected_score(elo1, 0)  # H1 下嘅預期勝率
    p1 = expected_score(elo0, 0)  # H0 下嘅預期勝率

    llr = (
        wins * math.log(p0 / p1) +
        losses * math.log((1 - p0) / (1 - p1))
    )

    # 決策邊界
    lower = math.log(beta / (1 - alpha))
    upper = math.log((1 - beta) / alpha)

    if llr <= lower:
        return 'reject'  # H0 被拒絕，新模型較差
    elif llr >= upper:
        return 'accept'  # H0 被接受，新模型較好
    else:
        return 'continue'  # 繼續測試
```

---

## KataGo 基準測試

### 執行基準測試

```bash
# 基本測試
katago benchmark -model model.bin.gz

# 指定訪問次數
katago benchmark -model model.bin.gz -v 1000

# 詳細輸出
katago benchmark -model model.bin.gz -v 1000 -t 8
```

### 輸出解讀

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

### 關鍵指標

| 指標 | 說明 | 好嘅數值 |
|------|------|---------|
| NN evals/sec | 神經網絡評估速度 | >1000 |
| Playouts/sec | MCTS 模擬速度 | >2000 |
| GPU 利用率 | GPU 使用效率 | >80% |

---

## 棋力評估

### 人類棋力對照

| AI Elo | 人類棋力 |
|--------|---------|
| ~1500 | 業餘 1 段 |
| ~2000 | 業餘 5 段 |
| ~2500 | 職業初段 |
| ~3000 | 職業五段 |
| ~3500 | 世界冠軍級 |
| ~4000+ | 超越人類 |

### 主要 AI 嘅 Elo

| AI | Elo（估計） |
|----|-----------|
| KataGo (最新) | ~5000 |
| AlphaGo Zero | ~5000 |
| Leela Zero | ~4500 |
| 絕藝 | ~4800 |

### 測試對照

```python
def estimate_human_rank(ai_model, test_positions):
    """估計 AI 相當於嘅人類棋力"""
    # 用標準測試題目
    correct = 0
    for pos in test_positions:
        ai_move = ai_model.get_best_move(pos['state'])
        if ai_move == pos['best_move']:
            correct += 1

    accuracy = correct / len(test_positions)

    # 準確率對照表
    if accuracy > 0.9:
        return "職業級"
    elif accuracy > 0.7:
        return "業餘 5 段+"
    elif accuracy > 0.5:
        return "業餘 1-5 段"
    else:
        return "業餘級以下"
```

---

## 效能監控

### 持續監控

```python
import time
import psutil
import GPUtil

class PerformanceMonitor:
    def __init__(self):
        self.metrics = []

    def sample(self):
        """取樣當前效能指標"""
        gpus = GPUtil.getGPUs()

        self.metrics.append({
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_util': gpus[0].load * 100 if gpus else 0,
            'gpu_memory': gpus[0].memoryUsed if gpus else 0,
        })

    def report(self):
        """產生報告"""
        if not self.metrics:
            return

        avg_cpu = sum(m['cpu_percent'] for m in self.metrics) / len(self.metrics)
        avg_gpu = sum(m['gpu_util'] for m in self.metrics) / len(self.metrics)

        print(f"平均 CPU 使用率: {avg_cpu:.1f}%")
        print(f"平均 GPU 使用率: {avg_gpu:.1f}%")
```

### 效能瓶頸診斷

| 症狀 | 可能原因 | 解決方案 |
|------|---------|---------|
| CPU 100%，GPU 低 | 搜索執行緒唔夠 | 增加 numSearchThreads |
| GPU 100%，輸出慢 | 批次太細 | 增加 nnMaxBatchSize |
| 記憶體唔夠 | 模型太大 | 用較細模型 |
| 速度唔穩定 | 溫度過高 | 改善散熱 |

---

## 自動化測試

### CI/CD 整合

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

### 回歸測試

```python
def regression_test(new_model, baseline_model, threshold=0.95):
    """檢查新模型係咪有效能回歸"""
    # 測試準確率
    new_accuracy = test_accuracy(new_model)
    baseline_accuracy = test_accuracy(baseline_model)

    if new_accuracy < baseline_accuracy * threshold:
        raise Exception(f"準確率回歸: {new_accuracy:.3f} < {baseline_accuracy:.3f}")

    # 測試速度
    new_speed = benchmark_speed(new_model)
    baseline_speed = benchmark_speed(baseline_model)

    if new_speed < baseline_speed * threshold:
        raise Exception(f"速度回歸: {new_speed:.1f} < {baseline_speed:.1f}")

    print("回歸測試通過")
```

---

## 延伸閱讀

- [KataGo 訓練機制解析](../training) — 模型係點樣訓練嘅
- [分散式訓練架構](../distributed-training) — 大規模評估
- [GPU 後端與優化](../gpu-optimization) — 效能調校
