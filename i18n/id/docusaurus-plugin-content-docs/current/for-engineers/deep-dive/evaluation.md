---
sidebar_position: 9
title: Evaluasi dan Benchmark
description: Sistem rating Elo AI Go, pengujian pertandingan, dan metode benchmark performa
---

# Evaluasi dan Benchmark

Artikel ini memperkenalkan cara mengevaluasi kekuatan dan performa AI Go, termasuk sistem rating Elo, metode pengujian pertandingan, dan benchmark standar.

---

## Sistem Rating Elo

### Konsep Dasar

Rating Elo adalah metode standar untuk mengukur kekuatan relatif:

```
Expected winrate E_A = 1 / (1 + 10^((R_B - R_A) / 400))

Elo baru = Elo lama + K × (hasil aktual - hasil yang diharapkan)
```

### Tabel Konversi Selisih Elo dan Winrate

| Selisih Elo | Winrate yang Lebih Kuat |
|-------------|------------------------|
| 0 | 50% |
| 100 | 64% |
| 200 | 76% |
| 400 | 91% |
| 800 | 99% |

### Implementasi

```python
def expected_score(rating_a, rating_b):
    """Hitung expected score A terhadap B"""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating, expected, actual, k=32):
    """Update rating Elo"""
    return rating + k * (actual - expected)

def calculate_elo_diff(wins, losses, draws):
    """Hitung selisih Elo dari hasil pertandingan"""
    total = wins + losses + draws
    win_rate = (wins + 0.5 * draws) / total

    if win_rate <= 0 or win_rate >= 1:
        return float('inf') if win_rate >= 1 else float('-inf')

    return 400 * math.log10(win_rate / (1 - win_rate))
```

---

## Pengujian Pertandingan

### Framework Pengujian

```python
class MatchTester:
    def __init__(self, engine_a, engine_b):
        self.engine_a = engine_a
        self.engine_b = engine_b
        self.results = {'a_wins': 0, 'b_wins': 0, 'draws': 0}

    def run_match(self, num_games=400):
        """Jalankan pengujian pertandingan"""
        for i in range(num_games):
            # Bergantian giliran pertama
            if i % 2 == 0:
                black, white = self.engine_a, self.engine_b
                a_is_black = True
            else:
                black, white = self.engine_b, self.engine_a
                a_is_black = False

            # Mainkan pertandingan
            result = self.play_game(black, white)

            # Catat hasil
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
        """Mainkan satu pertandingan"""
        game = Game()

        while not game.is_terminal():
            if game.current_player == 'black':
                move = black_engine.get_move(game.state)
            else:
                move = white_engine.get_move(game.state)

            game.play(move)

        return game.get_winner()
```

### Signifikansi Statistik

Pastikan hasil pengujian memiliki makna statistik:

```python
from scipy import stats

def calculate_confidence_interval(wins, total, confidence=0.95):
    """Hitung interval kepercayaan winrate"""
    p = wins / total
    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * math.sqrt(p * (1 - p) / total)

    return (p - margin, p + margin)

# Contoh
wins, total = 220, 400
ci_low, ci_high = calculate_confidence_interval(wins, total)
print(f"Winrate: {wins/total:.1%}, 95% CI: [{ci_low:.1%}, {ci_high:.1%}]")
```

### Jumlah Pertandingan yang Disarankan

| Perkiraan Selisih Elo | Jumlah Pertandingan yang Disarankan | Tingkat Kepercayaan |
|-----------------------|-------------------------------------|---------------------|
| \>100 | 100 | 95% |
| 50-100 | 200 | 95% |
| 20-50 | 400 | 95% |
| \<20 | 1000+ | 95% |

---

## SPRT (Sequential Probability Ratio Test)

### Konsep

Tidak perlu jumlah pertandingan tetap, keputusan berhenti berdasarkan hasil kumulatif secara dinamis:

```python
def sprt(wins, losses, elo0=0, elo1=10, alpha=0.05, beta=0.05):
    """
    Sequential Probability Ratio Test

    elo0: Selisih Elo hipotesis null (biasanya 0)
    elo1: Selisih Elo hipotesis alternatif (biasanya 5-20)
    alpha: Tingkat false positive
    beta: Tingkat false negative
    """
    if wins + losses == 0:
        return 'continue'

    # Hitung log likelihood ratio
    p0 = expected_score(elo1, 0)  # Expected winrate di bawah H1
    p1 = expected_score(elo0, 0)  # Expected winrate di bawah H0

    llr = (
        wins * math.log(p0 / p1) +
        losses * math.log((1 - p0) / (1 - p1))
    )

    # Batas keputusan
    lower = math.log(beta / (1 - alpha))
    upper = math.log((1 - beta) / alpha)

    if llr <= lower:
        return 'reject'  # H0 ditolak, model baru lebih lemah
    elif llr >= upper:
        return 'accept'  # H0 diterima, model baru lebih kuat
    else:
        return 'continue'  # Lanjutkan pengujian
```

---

## Benchmark KataGo

### Menjalankan Benchmark

```bash
# Pengujian dasar
katago benchmark -model model.bin.gz

# Tentukan jumlah kunjungan
katago benchmark -model model.bin.gz -v 1000

# Output detail
katago benchmark -model model.bin.gz -v 1000 -t 8
```

### Interpretasi Output

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

### Metrik Kunci

| Metrik | Deskripsi | Nilai Bagus |
|--------|-----------|-------------|
| NN evals/sec | Kecepatan evaluasi neural network | >1000 |
| Playouts/sec | Kecepatan simulasi MCTS | >2000 |
| Utilisasi GPU | Efisiensi penggunaan GPU | >80% |

---

## Evaluasi Kekuatan

### Padanan Kekuatan Manusia

| Elo AI | Kekuatan Manusia |
|--------|------------------|
| ~1500 | Amatir 1 dan |
| ~2000 | Amatir 5 dan |
| ~2500 | Profesional shodan |
| ~3000 | Profesional 5 dan |
| ~3500 | Level juara dunia |
| ~4000+ | Melampaui manusia |

### Elo AI Utama

| AI | Elo (perkiraan) |
|----|-----------------|
| KataGo (terbaru) | ~5000 |
| AlphaGo Zero | ~5000 |
| Leela Zero | ~4500 |
| Fine Art | ~4800 |

### Pengujian Perbandingan

```python
def estimate_human_rank(ai_model, test_positions):
    """Perkirakan kekuatan manusia yang setara dengan AI"""
    # Gunakan soal tes standar
    correct = 0
    for pos in test_positions:
        ai_move = ai_model.get_best_move(pos['state'])
        if ai_move == pos['best_move']:
            correct += 1

    accuracy = correct / len(test_positions)

    # Tabel padanan akurasi
    if accuracy > 0.9:
        return "Level profesional"
    elif accuracy > 0.7:
        return "Amatir 5 dan+"
    elif accuracy > 0.5:
        return "Amatir 1-5 dan"
    else:
        return "Di bawah level amatir"
```

---

## Monitoring Performa

### Monitoring Berkelanjutan

```python
import time
import psutil
import GPUtil

class PerformanceMonitor:
    def __init__(self):
        self.metrics = []

    def sample(self):
        """Sampel metrik performa saat ini"""
        gpus = GPUtil.getGPUs()

        self.metrics.append({
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_util': gpus[0].load * 100 if gpus else 0,
            'gpu_memory': gpus[0].memoryUsed if gpus else 0,
        })

    def report(self):
        """Buat laporan"""
        if not self.metrics:
            return

        avg_cpu = sum(m['cpu_percent'] for m in self.metrics) / len(self.metrics)
        avg_gpu = sum(m['gpu_util'] for m in self.metrics) / len(self.metrics)

        print(f"Rata-rata penggunaan CPU: {avg_cpu:.1f}%")
        print(f"Rata-rata penggunaan GPU: {avg_gpu:.1f}%")
```

### Diagnosis Bottleneck Performa

| Gejala | Kemungkinan Penyebab | Solusi |
|--------|---------------------|--------|
| CPU 100%, GPU rendah | Thread pencarian tidak cukup | Tingkatkan numSearchThreads |
| GPU 100%, output lambat | Batch terlalu kecil | Tingkatkan nnMaxBatchSize |
| Memori tidak cukup | Model terlalu besar | Gunakan model lebih kecil |
| Kecepatan tidak stabil | Suhu terlalu tinggi | Perbaiki pendinginan |

---

## Pengujian Otomatis

### Integrasi CI/CD

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

### Pengujian Regresi

```python
def regression_test(new_model, baseline_model, threshold=0.95):
    """Periksa apakah model baru memiliki regresi performa"""
    # Uji akurasi
    new_accuracy = test_accuracy(new_model)
    baseline_accuracy = test_accuracy(baseline_model)

    if new_accuracy < baseline_accuracy * threshold:
        raise Exception(f"Regresi akurasi: {new_accuracy:.3f} < {baseline_accuracy:.3f}")

    # Uji kecepatan
    new_speed = benchmark_speed(new_model)
    baseline_speed = benchmark_speed(baseline_model)

    if new_speed < baseline_speed * threshold:
        raise Exception(f"Regresi kecepatan: {new_speed:.1f} < {baseline_speed:.1f}")

    print("Pengujian regresi berhasil")
```

---

## Bacaan Lanjutan

- [Analisis Mekanisme Pelatihan KataGo](../training) — Bagaimana model dilatih
- [Arsitektur Pelatihan Terdistribusi](../distributed-training) — Evaluasi skala besar
- [Backend GPU dan Optimasi](../gpu-optimization) — Tuning performa
