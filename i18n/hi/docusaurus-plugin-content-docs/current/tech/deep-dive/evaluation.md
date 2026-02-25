---
sidebar_position: 9
title: मूल्यांकन और बेंचमार्क टेस्टिंग
description: गो AI का Elo रेटिंग सिस्टम, मैच टेस्टिंग और प्रदर्शन बेंचमार्क विधियाँ
---

# मूल्यांकन और बेंचमार्क टेस्टिंग

यह लेख गो AI की खेल शक्ति और प्रदर्शन का मूल्यांकन करने का परिचय देता है, जिसमें Elo रेटिंग सिस्टम, मैच टेस्टिंग विधियाँ और मानक बेंचमार्क टेस्ट शामिल हैं।

---

## Elo रेटिंग सिस्टम

### मूल अवधारणा

Elo रेटिंग सापेक्ष खेल शक्ति मापने की मानक विधि है:

```
अपेक्षित जीत दर E_A = 1 / (1 + 10^((R_B - R_A) / 400))

नया Elo = पुराना Elo + K × (वास्तविक परिणाम - अपेक्षित परिणाम)
```

### Elo अंतर और जीत दर तुलना

| Elo अंतर | मजबूत खिलाड़ी की जीत दर |
|---------|---------|
| 0 | 50% |
| 100 | 64% |
| 200 | 76% |
| 400 | 91% |
| 800 | 99% |

### कार्यान्वयन

```python
def expected_score(rating_a, rating_b):
    """A का B के खिलाफ अपेक्षित स्कोर गणना करें"""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating, expected, actual, k=32):
    """Elo रेटिंग अपडेट करें"""
    return rating + k * (actual - expected)

def calculate_elo_diff(wins, losses, draws):
    """मैच परिणामों से Elo अंतर गणना करें"""
    total = wins + losses + draws
    win_rate = (wins + 0.5 * draws) / total

    if win_rate <= 0 or win_rate >= 1:
        return float('inf') if win_rate >= 1 else float('-inf')

    return 400 * math.log10(win_rate / (1 - win_rate))
```

---

## मैच टेस्टिंग

### टेस्टिंग फ्रेमवर्क

```python
class MatchTester:
    def __init__(self, engine_a, engine_b):
        self.engine_a = engine_a
        self.engine_b = engine_b
        self.results = {'a_wins': 0, 'b_wins': 0, 'draws': 0}

    def run_match(self, num_games=400):
        """मैच टेस्ट निष्पादित करें"""
        for i in range(num_games):
            # बारी-बारी से काला/सफेद
            if i % 2 == 0:
                black, white = self.engine_a, self.engine_b
                a_is_black = True
            else:
                black, white = self.engine_b, self.engine_a
                a_is_black = False

            # गेम खेलें
            result = self.play_game(black, white)

            # परिणाम रिकॉर्ड करें
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
        """एक गेम खेलें"""
        game = Game()

        while not game.is_terminal():
            if game.current_player == 'black':
                move = black_engine.get_move(game.state)
            else:
                move = white_engine.get_move(game.state)

            game.play(move)

        return game.get_winner()
```

### सांख्यिकीय महत्व

सुनिश्चित करें परीक्षण परिणाम सांख्यिकीय रूप से अर्थपूर्ण हैं:

```python
from scipy import stats

def calculate_confidence_interval(wins, total, confidence=0.95):
    """जीत दर का कॉन्फिडेंस इंटरवल गणना करें"""
    p = wins / total
    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * math.sqrt(p * (1 - p) / total)

    return (p - margin, p + margin)

# उदाहरण
wins, total = 220, 400
ci_low, ci_high = calculate_confidence_interval(wins, total)
print(f"जीत दर: {wins/total:.1%}, 95% CI: [{ci_low:.1%}, {ci_high:.1%}]")
```

### सुझाई गेम संख्या

| अपेक्षित Elo अंतर | सुझाई गेम संख्या | कॉन्फिडेंस |
|------------|---------|--------|
| \>100 | 100 | 95% |
| 50-100 | 200 | 95% |
| 20-50 | 400 | 95% |
| \<20 | 1000+ | 95% |

---

## SPRT (सीक्वेंशियल प्रोबेबिलिटी रेशियो टेस्ट)

### अवधारणा

निश्चित गेम संख्या की आवश्यकता नहीं, संचित परिणामों के आधार पर गतिशील रूप से रुकने का निर्णय:

```python
def sprt(wins, losses, elo0=0, elo1=10, alpha=0.05, beta=0.05):
    """
    सीक्वेंशियल प्रोबेबिलिटी रेशियो टेस्ट

    elo0: नल परिकल्पना का Elo अंतर (आमतौर पर 0)
    elo1: वैकल्पिक परिकल्पना का Elo अंतर (आमतौर पर 5-20)
    alpha: फॉल्स पॉज़िटिव दर
    beta: फॉल्स नेगेटिव दर
    """
    if wins + losses == 0:
        return 'continue'

    # लॉग लाइकलीहुड रेशियो गणना करें
    p0 = expected_score(elo1, 0)  # H1 के तहत अपेक्षित जीत दर
    p1 = expected_score(elo0, 0)  # H0 के तहत अपेक्षित जीत दर

    llr = (
        wins * math.log(p0 / p1) +
        losses * math.log((1 - p0) / (1 - p1))
    )

    # निर्णय सीमाएं
    lower = math.log(beta / (1 - alpha))
    upper = math.log((1 - beta) / alpha)

    if llr <= lower:
        return 'reject'  # H0 अस्वीकृत, नया मॉडल कमजोर
    elif llr >= upper:
        return 'accept'  # H0 स्वीकृत, नया मॉडल बेहतर
    else:
        return 'continue'  # परीक्षण जारी रखें
```

---

## KataGo बेंचमार्क टेस्टिंग

### बेंचमार्क चलाएं

```bash
# मूल परीक्षण
katago benchmark -model model.bin.gz

# विज़िट संख्या निर्दिष्ट करें
katago benchmark -model model.bin.gz -v 1000

# विस्तृत आउटपुट
katago benchmark -model model.bin.gz -v 1000 -t 8
```

### आउटपुट व्याख्या

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

### मुख्य मेट्रिक्स

| मेट्रिक | विवरण | अच्छा मान |
|------|------|---------|
| NN evals/sec | न्यूरल नेटवर्क मूल्यांकन गति | >1000 |
| Playouts/sec | MCTS सिमुलेशन गति | >2000 |
| GPU उपयोग | GPU उपयोग दक्षता | >80% |

---

## खेल शक्ति मूल्यांकन

### मानव खेल शक्ति तुलना

| AI Elo | मानव खेल शक्ति |
|--------|---------|
| ~1500 | एमेच्योर 1 दान |
| ~2000 | एमेच्योर 5 दान |
| ~2500 | प्रोफेशनल 1 दान |
| ~3000 | प्रोफेशनल 5 दान |
| ~3500 | विश्व चैंपियन स्तर |
| ~4000+ | मानव से परे |

### प्रमुख AI की Elo

| AI | Elo (अनुमानित) |
|----|-----------|
| KataGo (नवीनतम) | ~5000 |
| AlphaGo Zero | ~5000 |
| Leela Zero | ~4500 |
| 绝艺 | ~4800 |

### परीक्षण तुलना

```python
def estimate_human_rank(ai_model, test_positions):
    """AI की मानव खेल शक्ति का अनुमान"""
    # मानक परीक्षण प्रश्न उपयोग करें
    correct = 0
    for pos in test_positions:
        ai_move = ai_model.get_best_move(pos['state'])
        if ai_move == pos['best_move']:
            correct += 1

    accuracy = correct / len(test_positions)

    # सटीकता तुलना तालिका
    if accuracy > 0.9:
        return "प्रोफेशनल स्तर"
    elif accuracy > 0.7:
        return "एमेच्योर 5 दान+"
    elif accuracy > 0.5:
        return "एमेच्योर 1-5 दान"
    else:
        return "एमेच्योर स्तर से नीचे"
```

---

## प्रदर्शन मॉनिटरिंग

### निरंतर मॉनिटरिंग

```python
import time
import psutil
import GPUtil

class PerformanceMonitor:
    def __init__(self):
        self.metrics = []

    def sample(self):
        """वर्तमान प्रदर्शन मेट्रिक्स सैंपल करें"""
        gpus = GPUtil.getGPUs()

        self.metrics.append({
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_util': gpus[0].load * 100 if gpus else 0,
            'gpu_memory': gpus[0].memoryUsed if gpus else 0,
        })

    def report(self):
        """रिपोर्ट जनरेट करें"""
        if not self.metrics:
            return

        avg_cpu = sum(m['cpu_percent'] for m in self.metrics) / len(self.metrics)
        avg_gpu = sum(m['gpu_util'] for m in self.metrics) / len(self.metrics)

        print(f"औसत CPU उपयोग: {avg_cpu:.1f}%")
        print(f"औसत GPU उपयोग: {avg_gpu:.1f}%")
```

### प्रदर्शन बॉटलनेक निदान

| लक्षण | संभावित कारण | समाधान |
|------|---------|---------|
| CPU 100%, GPU कम | सर्च थ्रेड्स अपर्याप्त | numSearchThreads बढ़ाएं |
| GPU 100%, आउटपुट धीमा | बैच बहुत छोटा | nnMaxBatchSize बढ़ाएं |
| मेमोरी अपर्याप्त | मॉडल बहुत बड़ा | छोटा मॉडल उपयोग करें |
| गति अस्थिर | तापमान अधिक | कूलिंग सुधारें |

---

## ऑटोमेटेड टेस्टिंग

### CI/CD इंटीग्रेशन

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

### रिग्रेशन टेस्टिंग

```python
def regression_test(new_model, baseline_model, threshold=0.95):
    """जांचें नए मॉडल में प्रदर्शन रिग्रेशन है या नहीं"""
    # सटीकता परीक्षण
    new_accuracy = test_accuracy(new_model)
    baseline_accuracy = test_accuracy(baseline_model)

    if new_accuracy < baseline_accuracy * threshold:
        raise Exception(f"सटीकता रिग्रेशन: {new_accuracy:.3f} < {baseline_accuracy:.3f}")

    # गति परीक्षण
    new_speed = benchmark_speed(new_model)
    baseline_speed = benchmark_speed(baseline_model)

    if new_speed < baseline_speed * threshold:
        raise Exception(f"गति रिग्रेशन: {new_speed:.1f} < {baseline_speed:.1f}")

    print("रिग्रेशन टेस्ट पास")
```

---

## आगे पढ़ें

- [KataGo प्रशिक्षण तंत्र विश्लेषण](../training) — मॉडल कैसे प्रशिक्षित होता है
- [वितरित प्रशिक्षण आर्किटेक्चर](../distributed-training) — बड़े पैमाने पर मूल्यांकन
- [GPU बैकएंड और अनुकूलन](../gpu-optimization) — प्रदर्शन ट्यूनिंग
