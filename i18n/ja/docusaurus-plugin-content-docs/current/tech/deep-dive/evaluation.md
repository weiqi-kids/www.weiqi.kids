---
sidebar_position: 9
title: 評価とベンチマーク
description: 囲碁AIのEloレーティングシステム、対局テスト、パフォーマンスベンチマーク手法
---

# 評価とベンチマーク

本記事では、囲碁AIの棋力とパフォーマンスの評価方法を、Eloレーティングシステム、対局テスト手法、標準ベンチマークを含めて紹介します。

---

## Eloレーティングシステム

### 基本概念

Eloレーティングは相対的な棋力を測定する標準的な方法です：

```
期待勝率 E_A = 1 / (1 + 10^((R_B - R_A) / 400))

新Elo = 旧Elo + K × (実際の結果 - 期待結果)
```

### Elo差と勝率対照表

| Elo差 | 強者の勝率 |
|---------|---------|
| 0 | 50% |
| 100 | 64% |
| 200 | 76% |
| 400 | 91% |
| 800 | 99% |

### 実装

```python
def expected_score(rating_a, rating_b):
    """AのBに対する期待スコアを計算"""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating, expected, actual, k=32):
    """Eloレーティングを更新"""
    return rating + k * (actual - expected)

def calculate_elo_diff(wins, losses, draws):
    """対戦結果からElo差を計算"""
    total = wins + losses + draws
    win_rate = (wins + 0.5 * draws) / total

    if win_rate <= 0 or win_rate >= 1:
        return float('inf') if win_rate >= 1 else float('-inf')

    return 400 * math.log10(win_rate / (1 - win_rate))
```

---

## 対局テスト

### テストフレームワーク

```python
class MatchTester:
    def __init__(self, engine_a, engine_b):
        self.engine_a = engine_a
        self.engine_b = engine_b
        self.results = {'a_wins': 0, 'b_wins': 0, 'draws': 0}

    def run_match(self, num_games=400):
        """対戦テストを実行"""
        for i in range(num_games):
            # 先後を交代
            if i % 2 == 0:
                black, white = self.engine_a, self.engine_b
                a_is_black = True
            else:
                black, white = self.engine_b, self.engine_a
                a_is_black = False

            # 対局を実施
            result = self.play_game(black, white)

            # 結果を記録
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
        """1局の対局を実施"""
        game = Game()

        while not game.is_terminal():
            if game.current_player == 'black':
                move = black_engine.get_move(game.state)
            else:
                move = white_engine.get_move(game.state)

            game.play(move)

        return game.get_winner()
```

### 統計的有意性

テスト結果に統計的意味があることを確認：

```python
from scipy import stats

def calculate_confidence_interval(wins, total, confidence=0.95):
    """勝率の信頼区間を計算"""
    p = wins / total
    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * math.sqrt(p * (1 - p) / total)

    return (p - margin, p + margin)

# 例
wins, total = 220, 400
ci_low, ci_high = calculate_confidence_interval(wins, total)
print(f"勝率: {wins/total:.1%}, 95% CI: [{ci_low:.1%}, {ci_high:.1%}]")
```

### 推奨対局数

| 予想Elo差 | 推奨対局数 | 信頼度 |
|------------|---------|--------|
| \>100 | 100 | 95% |
| 50-100 | 200 | 95% |
| 20-50 | 400 | 95% |
| \<20 | 1000以上 | 95% |

---

## SPRT（逐次確率比検定）

### 概念

対局数を固定せず、累積結果に基づいて動的に停止を判断：

```python
def sprt(wins, losses, elo0=0, elo1=10, alpha=0.05, beta=0.05):
    """
    逐次確率比検定

    elo0: 帰無仮説のElo差（通常0）
    elo1: 対立仮説のElo差（通常5-20）
    alpha: 偽陽性率
    beta: 偽陰性率
    """
    if wins + losses == 0:
        return 'continue'

    # 対数尤度比を計算
    p0 = expected_score(elo1, 0)  # H1での期待勝率
    p1 = expected_score(elo0, 0)  # H0での期待勝率

    llr = (
        wins * math.log(p0 / p1) +
        losses * math.log((1 - p0) / (1 - p1))
    )

    # 決定境界
    lower = math.log(beta / (1 - alpha))
    upper = math.log((1 - beta) / alpha)

    if llr <= lower:
        return 'reject'  # H0が棄却、新モデルは弱い
    elif llr >= upper:
        return 'accept'  # H0が受容、新モデルは強い
    else:
        return 'continue'  # テストを継続
```

---

## KataGoベンチマーク

### ベンチマークの実行

```bash
# 基本テスト
katago benchmark -model model.bin.gz

# 訪問回数を指定
katago benchmark -model model.bin.gz -v 1000

# 詳細出力
katago benchmark -model model.bin.gz -v 1000 -t 8
```

### 出力の解釈

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

### 主要指標

| 指標 | 説明 | 良好な値 |
|------|------|---------|
| NN evals/sec | ニューラルネットワーク評価速度 | >1000 |
| Playouts/sec | MCTSシミュレーション速度 | >2000 |
| GPU使用率 | GPU使用効率 | >80% |

---

## 棋力評価

### 人間の棋力対照

| AI Elo | 人間の棋力 |
|--------|---------|
| ~1500 | アマチュア初段 |
| ~2000 | アマチュア5段 |
| ~2500 | プロ初段 |
| ~3000 | プロ五段 |
| ~3500 | 世界チャンピオン級 |
| ~4000以上 | 人間を超越 |

### 主要AIのElo

| AI | Elo（推定） |
|----|-----------|
| KataGo (最新) | ~5000 |
| AlphaGo Zero | ~5000 |
| Leela Zero | ~4500 |
| 絶芸 | ~4800 |

### テスト対照

```python
def estimate_human_rank(ai_model, test_positions):
    """AIの人間相当の棋力を推定"""
    # 標準テスト問題を使用
    correct = 0
    for pos in test_positions:
        ai_move = ai_model.get_best_move(pos['state'])
        if ai_move == pos['best_move']:
            correct += 1

    accuracy = correct / len(test_positions)

    # 正確率対照表
    if accuracy > 0.9:
        return "プロ級"
    elif accuracy > 0.7:
        return "アマチュア5段以上"
    elif accuracy > 0.5:
        return "アマチュア1-5段"
    else:
        return "アマチュア級以下"
```

---

## パフォーマンス監視

### 継続的監視

```python
import time
import psutil
import GPUtil

class PerformanceMonitor:
    def __init__(self):
        self.metrics = []

    def sample(self):
        """現在のパフォーマンス指標をサンプリング"""
        gpus = GPUtil.getGPUs()

        self.metrics.append({
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_util': gpus[0].load * 100 if gpus else 0,
            'gpu_memory': gpus[0].memoryUsed if gpus else 0,
        })

    def report(self):
        """レポートを生成"""
        if not self.metrics:
            return

        avg_cpu = sum(m['cpu_percent'] for m in self.metrics) / len(self.metrics)
        avg_gpu = sum(m['gpu_util'] for m in self.metrics) / len(self.metrics)

        print(f"平均CPU使用率: {avg_cpu:.1f}%")
        print(f"平均GPU使用率: {avg_gpu:.1f}%")
```

### パフォーマンスボトルネック診断

| 症状 | 考えられる原因 | 解決策 |
|------|---------|---------|
| CPU 100%、GPU低い | 探索スレッド不足 | numSearchThreadsを増やす |
| GPU 100%、出力が遅い | バッチが小さすぎる | nnMaxBatchSizeを増やす |
| メモリ不足 | モデルが大きすぎる | より小さいモデルを使用 |
| 速度が不安定 | 温度が高すぎる | 冷却を改善 |

---

## 自動化テスト

### CI/CD統合

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

### 回帰テスト

```python
def regression_test(new_model, baseline_model, threshold=0.95):
    """新モデルにパフォーマンス回帰がないかチェック"""
    # 精度をテスト
    new_accuracy = test_accuracy(new_model)
    baseline_accuracy = test_accuracy(baseline_model)

    if new_accuracy < baseline_accuracy * threshold:
        raise Exception(f"精度回帰: {new_accuracy:.3f} < {baseline_accuracy:.3f}")

    # 速度をテスト
    new_speed = benchmark_speed(new_model)
    baseline_speed = benchmark_speed(baseline_model)

    if new_speed < baseline_speed * threshold:
        raise Exception(f"速度回帰: {new_speed:.1f} < {baseline_speed:.1f}")

    print("回帰テスト合格")
```

---

## 関連記事

- [KataGo訓練メカニズム解析](../training) — モデルの訓練方法
- [分散訓練アーキテクチャ](../distributed-training) — 大規模評価
- [GPUバックエンドと最適化](../gpu-optimization) — パフォーマンスチューニング
