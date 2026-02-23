---
sidebar_position: 9
title: Evaluation & Benchmarking
description: Go AI Elo rating system, match testing, and performance benchmarking methods
---

# Evaluation & Benchmarking

This article introduces how to evaluate Go AI strength and performance, including the Elo rating system, match testing methods, and standard benchmarks.

---

## Elo Rating System

### Basic Concept

Elo rating is the standard method for measuring relative playing strength:

```
Expected win rate E_A = 1 / (1 + 10^((R_B - R_A) / 400))

New Elo = Old Elo + K × (Actual result - Expected result)
```

### Elo Difference vs Win Rate

| Elo Difference | Stronger Player Win Rate |
|----------------|-------------------------|
| 0 | 50% |
| 100 | 64% |
| 200 | 76% |
| 400 | 91% |
| 800 | 99% |

### Implementation

```python
def expected_score(rating_a, rating_b):
    """Calculate A's expected score against B"""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating, expected, actual, k=32):
    """Update Elo rating"""
    return rating + k * (actual - expected)

def calculate_elo_diff(wins, losses, draws):
    """Calculate Elo difference from match results"""
    total = wins + losses + draws
    win_rate = (wins + 0.5 * draws) / total

    if win_rate <= 0 or win_rate >= 1:
        return float('inf') if win_rate >= 1 else float('-inf')

    return 400 * math.log10(win_rate / (1 - win_rate))
```

---

## Match Testing

### Testing Framework

```python
class MatchTester:
    def __init__(self, engine_a, engine_b):
        self.engine_a = engine_a
        self.engine_b = engine_b
        self.results = {'a_wins': 0, 'b_wins': 0, 'draws': 0}

    def run_match(self, num_games=400):
        """Run match test"""
        for i in range(num_games):
            # Alternate colors
            if i % 2 == 0:
                black, white = self.engine_a, self.engine_b
                a_is_black = True
            else:
                black, white = self.engine_b, self.engine_a
                a_is_black = False

            # Play game
            result = self.play_game(black, white)

            # Record result
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
        """Play one game"""
        game = Game()

        while not game.is_terminal():
            if game.current_player == 'black':
                move = black_engine.get_move(game.state)
            else:
                move = white_engine.get_move(game.state)

            game.play(move)

        return game.get_winner()
```

### Statistical Significance

Ensure test results are statistically meaningful:

```python
from scipy import stats

def calculate_confidence_interval(wins, total, confidence=0.95):
    """Calculate confidence interval for win rate"""
    p = wins / total
    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * math.sqrt(p * (1 - p) / total)

    return (p - margin, p + margin)

# Example
wins, total = 220, 400
ci_low, ci_high = calculate_confidence_interval(wins, total)
print(f"Win rate: {wins/total:.1%}, 95% CI: [{ci_low:.1%}, {ci_high:.1%}]")
```

### Recommended Game Counts

| Expected Elo Diff | Recommended Games | Confidence |
|-------------------|-------------------|------------|
| \>100 | 100 | 95% |
| 50-100 | 200 | 95% |
| 20-50 | 400 | 95% |
| \<20 | 1000+ | 95% |

---

## SPRT (Sequential Probability Ratio Test)

### Concept

No fixed game count needed; dynamically decide whether to stop based on accumulated results:

```python
def sprt(wins, losses, elo0=0, elo1=10, alpha=0.05, beta=0.05):
    """
    Sequential Probability Ratio Test

    elo0: Null hypothesis Elo difference (usually 0)
    elo1: Alternative hypothesis Elo difference (usually 5-20)
    alpha: False positive rate
    beta: False negative rate
    """
    if wins + losses == 0:
        return 'continue'

    # Calculate log likelihood ratio
    p0 = expected_score(elo1, 0)  # Expected win rate under H1
    p1 = expected_score(elo0, 0)  # Expected win rate under H0

    llr = (
        wins * math.log(p0 / p1) +
        losses * math.log((1 - p0) / (1 - p1))
    )

    # Decision boundaries
    lower = math.log(beta / (1 - alpha))
    upper = math.log((1 - beta) / alpha)

    if llr <= lower:
        return 'reject'  # H0 rejected, new model is worse
    elif llr >= upper:
        return 'accept'  # H0 accepted, new model is better
    else:
        return 'continue'  # Continue testing
```

---

## KataGo Benchmarking

### Run Benchmark

```bash
# Basic test
katago benchmark -model model.bin.gz

# Specify visit count
katago benchmark -model model.bin.gz -v 1000

# Detailed output
katago benchmark -model model.bin.gz -v 1000 -t 8
```

### Output Interpretation

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

### Key Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| NN evals/sec | Neural network evaluation speed | >1000 |
| Playouts/sec | MCTS simulation speed | >2000 |
| GPU utilization | GPU usage efficiency | >80% |

---

## Strength Evaluation

### Human Strength Reference

| AI Elo | Human Strength |
|--------|----------------|
| ~1500 | Amateur 1 dan |
| ~2000 | Amateur 5 dan |
| ~2500 | Professional 1 dan |
| ~3000 | Professional 5 dan |
| ~3500 | World champion level |
| ~4000+ | Beyond human |

### Major AI Elo Ratings

| AI | Elo (estimated) |
|----|-----------------|
| KataGo (latest) | ~5000 |
| AlphaGo Zero | ~5000 |
| Leela Zero | ~4500 |
| Fine Art | ~4800 |

### Test Comparison

```python
def estimate_human_rank(ai_model, test_positions):
    """Estimate AI's equivalent human rank"""
    # Use standard test problems
    correct = 0
    for pos in test_positions:
        ai_move = ai_model.get_best_move(pos['state'])
        if ai_move == pos['best_move']:
            correct += 1

    accuracy = correct / len(test_positions)

    # Accuracy reference table
    if accuracy > 0.9:
        return "Professional level"
    elif accuracy > 0.7:
        return "Amateur 5 dan+"
    elif accuracy > 0.5:
        return "Amateur 1-5 dan"
    else:
        return "Below amateur level"
```

---

## Performance Monitoring

### Continuous Monitoring

```python
import time
import psutil
import GPUtil

class PerformanceMonitor:
    def __init__(self):
        self.metrics = []

    def sample(self):
        """Sample current performance metrics"""
        gpus = GPUtil.getGPUs()

        self.metrics.append({
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_util': gpus[0].load * 100 if gpus else 0,
            'gpu_memory': gpus[0].memoryUsed if gpus else 0,
        })

    def report(self):
        """Generate report"""
        if not self.metrics:
            return

        avg_cpu = sum(m['cpu_percent'] for m in self.metrics) / len(self.metrics)
        avg_gpu = sum(m['gpu_util'] for m in self.metrics) / len(self.metrics)

        print(f"Average CPU usage: {avg_cpu:.1f}%")
        print(f"Average GPU usage: {avg_gpu:.1f}%")
```

### Performance Bottleneck Diagnosis

| Symptom | Possible Cause | Solution |
|---------|----------------|----------|
| CPU 100%, GPU low | Insufficient search threads | Increase numSearchThreads |
| GPU 100%, slow output | Batch too small | Increase nnMaxBatchSize |
| Out of memory | Model too large | Use smaller model |
| Unstable speed | Overheating | Improve cooling |

---

## Automated Testing

### CI/CD Integration

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

### Regression Testing

```python
def regression_test(new_model, baseline_model, threshold=0.95):
    """Check if new model has performance regression"""
    # Test accuracy
    new_accuracy = test_accuracy(new_model)
    baseline_accuracy = test_accuracy(baseline_model)

    if new_accuracy < baseline_accuracy * threshold:
        raise Exception(f"Accuracy regression: {new_accuracy:.3f} < {baseline_accuracy:.3f}")

    # Test speed
    new_speed = benchmark_speed(new_model)
    baseline_speed = benchmark_speed(baseline_model)

    if new_speed < baseline_speed * threshold:
        raise Exception(f"Speed regression: {new_speed:.1f} < {baseline_speed:.1f}")

    print("Regression test passed")
```

---

## Further Reading

- [KataGo Training Mechanism](../training) — How models are trained
- [Distributed Training Architecture](../distributed-training) — Large-scale evaluation
- [GPU Backend & Optimization](../gpu-optimization) — Performance tuning
