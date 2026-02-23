---
sidebar_position: 9
title: التقييم والاختبار المعياري
description: نظام تقييم Elo للذكاء الاصطناعي في الغو، اختبار المباريات وطرق الاختبار المعياري
---

# التقييم والاختبار المعياري

يقدم هذا المقال كيفية تقييم قوة لعب الذكاء الاصطناعي في الغو وأدائه، بما في ذلك نظام تقييم Elo وطرق اختبار المباريات والاختبارات المعيارية القياسية.

---

## نظام تقييم Elo

### المفهوم الأساسي

تقييم Elo هو الطريقة القياسية لقياس قوة اللعب النسبية:

```
معدل الفوز المتوقع E_A = 1 / (1 + 10^((R_B - R_A) / 400))

Elo الجديد = Elo القديم + K × (النتيجة الفعلية - النتيجة المتوقعة)
```

### جدول فرق Elo ومعدل الفوز

| فرق Elo | معدل فوز الأقوى |
|---------|-----------------|
| 0 | 50% |
| 100 | 64% |
| 200 | 76% |
| 400 | 91% |
| 800 | 99% |

### التنفيذ

```python
def expected_score(rating_a, rating_b):
    """حساب النقاط المتوقعة لـ A ضد B"""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating, expected, actual, k=32):
    """تحديث تقييم Elo"""
    return rating + k * (actual - expected)

def calculate_elo_diff(wins, losses, draws):
    """حساب فرق Elo من نتائج المباريات"""
    total = wins + losses + draws
    win_rate = (wins + 0.5 * draws) / total

    if win_rate <= 0 or win_rate >= 1:
        return float('inf') if win_rate >= 1 else float('-inf')

    return 400 * math.log10(win_rate / (1 - win_rate))
```

---

## اختبار المباريات

### إطار الاختبار

```python
class MatchTester:
    def __init__(self, engine_a, engine_b):
        self.engine_a = engine_a
        self.engine_b = engine_b
        self.results = {'a_wins': 0, 'b_wins': 0, 'draws': 0}

    def run_match(self, num_games=400):
        """تنفيذ اختبار المباريات"""
        for i in range(num_games):
            # تبديل البداية
            if i % 2 == 0:
                black, white = self.engine_a, self.engine_b
                a_is_black = True
            else:
                black, white = self.engine_b, self.engine_a
                a_is_black = False

            # إجراء المباراة
            result = self.play_game(black, white)

            # تسجيل النتيجة
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
        """إجراء مباراة واحدة"""
        game = Game()

        while not game.is_terminal():
            if game.current_player == 'black':
                move = black_engine.get_move(game.state)
            else:
                move = white_engine.get_move(game.state)

            game.play(move)

        return game.get_winner()
```

### الأهمية الإحصائية

التأكد من أن نتائج الاختبار ذات معنى إحصائي:

```python
from scipy import stats

def calculate_confidence_interval(wins, total, confidence=0.95):
    """حساب فترة الثقة لمعدل الفوز"""
    p = wins / total
    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * math.sqrt(p * (1 - p) / total)

    return (p - margin, p + margin)

# مثال
wins, total = 220, 400
ci_low, ci_high = calculate_confidence_interval(wins, total)
print(f"معدل الفوز: {wins/total:.1%}، فترة ثقة 95%: [{ci_low:.1%}، {ci_high:.1%}]")
```

### عدد المباريات المقترح

| فرق Elo المتوقع | عدد المباريات المقترح | مستوى الثقة |
|----------------|----------------------|-------------|
| \>100 | 100 | 95% |
| 50-100 | 200 | 95% |
| 20-50 | 400 | 95% |
| \<20 | 1000+ | 95% |

---

## SPRT (اختبار نسبة الاحتمالية التسلسلية)

### المفهوم

لا حاجة لعدد مباريات ثابت، القرار يعتمد على النتائج المتراكمة:

```python
def sprt(wins, losses, elo0=0, elo1=10, alpha=0.05, beta=0.05):
    """
    اختبار نسبة الاحتمالية التسلسلية

    elo0: فرق Elo للفرضية الصفرية (عادة 0)
    elo1: فرق Elo للفرضية البديلة (عادة 5-20)
    alpha: معدل الإيجابية الكاذبة
    beta: معدل السلبية الكاذبة
    """
    if wins + losses == 0:
        return 'continue'

    # حساب نسبة الاحتمالية اللوغاريتمية
    p0 = expected_score(elo1, 0)  # معدل الفوز المتوقع تحت H1
    p1 = expected_score(elo0, 0)  # معدل الفوز المتوقع تحت H0

    llr = (
        wins * math.log(p0 / p1) +
        losses * math.log((1 - p0) / (1 - p1))
    )

    # حدود القرار
    lower = math.log(beta / (1 - alpha))
    upper = math.log((1 - beta) / alpha)

    if llr <= lower:
        return 'reject'  # H0 مرفوضة، النموذج الجديد أضعف
    elif llr >= upper:
        return 'accept'  # H0 مقبولة، النموذج الجديد أفضل
    else:
        return 'continue'  # استمر في الاختبار
```

---

## اختبار KataGo المعياري

### تشغيل الاختبار المعياري

```bash
# اختبار أساسي
katago benchmark -model model.bin.gz

# تحديد عدد الزيارات
katago benchmark -model model.bin.gz -v 1000

# إخراج مفصل
katago benchmark -model model.bin.gz -v 1000 -t 8
```

### تفسير المخرجات

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

### المؤشرات الرئيسية

| المؤشر | الوصف | القيمة الجيدة |
|--------|-------|---------------|
| NN evals/sec | سرعة تقييم الشبكة العصبية | >1000 |
| Playouts/sec | سرعة محاكاة MCTS | >2000 |
| استخدام GPU | كفاءة استخدام GPU | >80% |

---

## تقييم قوة اللعب

### مقارنة مع قوة اللعب البشرية

| AI Elo | قوة اللعب البشرية |
|--------|-------------------|
| ~1500 | هاوٍ 1 دان |
| ~2000 | هاوٍ 5 دان |
| ~2500 | محترف مبتدئ |
| ~3000 | محترف 5 دان |
| ~3500 | مستوى بطل العالم |
| ~4000+ | يتجاوز البشر |

### Elo للذكاء الاصطناعي الرئيسي

| AI | Elo (تقدير) |
|----|-------------|
| KataGo (الأحدث) | ~5000 |
| AlphaGo Zero | ~5000 |
| Leela Zero | ~4500 |
| 絶藝 | ~4800 |

### اختبار المقارنة

```python
def estimate_human_rank(ai_model, test_positions):
    """تقدير قوة اللعب البشرية المعادلة للـ AI"""
    # استخدام مسائل اختبار قياسية
    correct = 0
    for pos in test_positions:
        ai_move = ai_model.get_best_move(pos['state'])
        if ai_move == pos['best_move']:
            correct += 1

    accuracy = correct / len(test_positions)

    # جدول مقارنة الدقة
    if accuracy > 0.9:
        return "مستوى محترف"
    elif accuracy > 0.7:
        return "هاوٍ 5 دان+"
    elif accuracy > 0.5:
        return "هاوٍ 1-5 دان"
    else:
        return "أقل من مستوى الهواة"
```

---

## مراقبة الأداء

### المراقبة المستمرة

```python
import time
import psutil
import GPUtil

class PerformanceMonitor:
    def __init__(self):
        self.metrics = []

    def sample(self):
        """أخذ عينة من مؤشرات الأداء الحالية"""
        gpus = GPUtil.getGPUs()

        self.metrics.append({
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_util': gpus[0].load * 100 if gpus else 0,
            'gpu_memory': gpus[0].memoryUsed if gpus else 0,
        })

    def report(self):
        """إنشاء تقرير"""
        if not self.metrics:
            return

        avg_cpu = sum(m['cpu_percent'] for m in self.metrics) / len(self.metrics)
        avg_gpu = sum(m['gpu_util'] for m in self.metrics) / len(self.metrics)

        print(f"متوسط استخدام CPU: {avg_cpu:.1f}%")
        print(f"متوسط استخدام GPU: {avg_gpu:.1f}%")
```

### تشخيص عنق الزجاجة في الأداء

| الأعراض | السبب المحتمل | الحل |
|---------|--------------|------|
| CPU 100%، GPU منخفض | خيوط بحث غير كافية | زيادة numSearchThreads |
| GPU 100%، إخراج بطيء | حجم الدفعة صغير | زيادة nnMaxBatchSize |
| ذاكرة غير كافية | النموذج كبير جداً | استخدام نموذج أصغر |
| سرعة غير مستقرة | درجة حرارة عالية | تحسين التبريد |

---

## الاختبار الآلي

### تكامل CI/CD

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

### اختبار التراجع

```python
def regression_test(new_model, baseline_model, threshold=0.95):
    """التحقق من عدم تراجع أداء النموذج الجديد"""
    # اختبار الدقة
    new_accuracy = test_accuracy(new_model)
    baseline_accuracy = test_accuracy(baseline_model)

    if new_accuracy < baseline_accuracy * threshold:
        raise Exception(f"تراجع الدقة: {new_accuracy:.3f} < {baseline_accuracy:.3f}")

    # اختبار السرعة
    new_speed = benchmark_speed(new_model)
    baseline_speed = benchmark_speed(baseline_model)

    if new_speed < baseline_speed * threshold:
        raise Exception(f"تراجع السرعة: {new_speed:.1f} < {baseline_speed:.1f}")

    print("اختبار التراجع ناجح")
```

---

## قراءات إضافية

- [تحليل آلية تدريب KataGo](../training) — كيف يتم تدريب النماذج
- [بنية التدريب الموزع](../distributed-training) — التقييم واسع النطاق
- [واجهات GPU والتحسين](../gpu-optimization) — ضبط الأداء
