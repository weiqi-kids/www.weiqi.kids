---
sidebar_position: 9
title: Evaluación y benchmarking
description: Sistema de puntuación Elo para IA de Go, pruebas de partidas y métodos de benchmarking de rendimiento
---

# Evaluación y benchmarking

Este artículo presenta cómo evaluar la fuerza de juego y el rendimiento de una IA de Go, incluyendo el sistema de puntuación Elo, métodos de prueba de partidas y benchmarks estándar.

---

## Sistema de puntuación Elo

### Concepto básico

La puntuación Elo es el método estándar para medir la fuerza relativa:

```
Tasa de victoria esperada E_A = 1 / (1 + 10^((R_B - R_A) / 400))

Nuevo Elo = Elo antiguo + K × (resultado real - resultado esperado)
```

### Correspondencia entre diferencia de Elo y tasa de victoria

| Diferencia de Elo | Tasa de victoria del más fuerte |
|-------------------|--------------------------------|
| 0 | 50% |
| 100 | 64% |
| 200 | 76% |
| 400 | 91% |
| 800 | 99% |

### Implementación

```python
def expected_score(rating_a, rating_b):
    """Calcular la puntuación esperada de A contra B"""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating, expected, actual, k=32):
    """Actualizar puntuación Elo"""
    return rating + k * (actual - expected)

def calculate_elo_diff(wins, losses, draws):
    """Calcular diferencia de Elo a partir de resultados de partidas"""
    total = wins + losses + draws
    win_rate = (wins + 0.5 * draws) / total

    if win_rate <= 0 or win_rate >= 1:
        return float('inf') if win_rate >= 1 else float('-inf')

    return 400 * math.log10(win_rate / (1 - win_rate))
```

---

## Pruebas de partidas

### Marco de pruebas

```python
class MatchTester:
    def __init__(self, engine_a, engine_b):
        self.engine_a = engine_a
        self.engine_b = engine_b
        self.results = {'a_wins': 0, 'b_wins': 0, 'draws': 0}

    def run_match(self, num_games=400):
        """Ejecutar prueba de enfrentamiento"""
        for i in range(num_games):
            # Alternar colores
            if i % 2 == 0:
                black, white = self.engine_a, self.engine_b
                a_is_black = True
            else:
                black, white = self.engine_b, self.engine_a
                a_is_black = False

            # Jugar partida
            result = self.play_game(black, white)

            # Registrar resultado
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
        """Jugar una partida"""
        game = Game()

        while not game.is_terminal():
            if game.current_player == 'black':
                move = black_engine.get_move(game.state)
            else:
                move = white_engine.get_move(game.state)

            game.play(move)

        return game.get_winner()
```

### Significancia estadística

Asegurar que los resultados de prueba tengan significado estadístico:

```python
from scipy import stats

def calculate_confidence_interval(wins, total, confidence=0.95):
    """Calcular intervalo de confianza de la tasa de victoria"""
    p = wins / total
    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * math.sqrt(p * (1 - p) / total)

    return (p - margin, p + margin)

# Ejemplo
wins, total = 220, 400
ci_low, ci_high = calculate_confidence_interval(wins, total)
print(f"Tasa de victoria: {wins/total:.1%}, IC 95%: [{ci_low:.1%}, {ci_high:.1%}]")
```

### Número de partidas recomendado

| Diferencia de Elo esperada | Partidas recomendadas | Confianza |
|----------------------------|----------------------|-----------|
| \>100 | 100 | 95% |
| 50-100 | 200 | 95% |
| 20-50 | 400 | 95% |
| \<20 | 1000+ | 95% |

---

## SPRT (Prueba de razón de probabilidad secuencial)

### Concepto

Sin necesidad de fijar el número de partidas, decide dinámicamente si parar según los resultados acumulados:

```python
def sprt(wins, losses, elo0=0, elo1=10, alpha=0.05, beta=0.05):
    """
    Prueba de razón de probabilidad secuencial

    elo0: Diferencia de Elo de la hipótesis nula (normalmente 0)
    elo1: Diferencia de Elo de la hipótesis alternativa (normalmente 5-20)
    alpha: Tasa de falsos positivos
    beta: Tasa de falsos negativos
    """
    if wins + losses == 0:
        return 'continue'

    # Calcular razón de log-verosimilitud
    p0 = expected_score(elo1, 0)  # Tasa de victoria esperada bajo H1
    p1 = expected_score(elo0, 0)  # Tasa de victoria esperada bajo H0

    llr = (
        wins * math.log(p0 / p1) +
        losses * math.log((1 - p0) / (1 - p1))
    )

    # Límites de decisión
    lower = math.log(beta / (1 - alpha))
    upper = math.log((1 - beta) / alpha)

    if llr <= lower:
        return 'reject'  # H0 rechazada, nuevo modelo es peor
    elif llr >= upper:
        return 'accept'  # H0 aceptada, nuevo modelo es mejor
    else:
        return 'continue'  # Continuar prueba
```

---

## Benchmarking de KataGo

### Ejecutar benchmark

```bash
# Prueba básica
katago benchmark -model model.bin.gz

# Especificar número de visitas
katago benchmark -model model.bin.gz -v 1000

# Salida detallada
katago benchmark -model model.bin.gz -v 1000 -t 8
```

### Interpretación de salida

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

### Métricas clave

| Métrica | Descripción | Buen valor |
|---------|-------------|------------|
| NN evals/seg | Velocidad de evaluación de red neuronal | >1000 |
| Playouts/seg | Velocidad de simulación MCTS | >2000 |
| Utilización GPU | Eficiencia de uso de GPU | >80% |

---

## Evaluación de fuerza de juego

### Correspondencia con fuerza humana

| Elo de IA | Fuerza humana |
|-----------|---------------|
| ~1500 | Amateur 1 dan |
| ~2000 | Amateur 5 dan |
| ~2500 | Profesional shodan |
| ~3000 | Profesional 5 dan |
| ~3500 | Nivel campeón mundial |
| ~4000+ | Supera a los humanos |

### Elo de principales IAs

| IA | Elo (estimado) |
|----|----------------|
| KataGo (última) | ~5000 |
| AlphaGo Zero | ~5000 |
| Leela Zero | ~4500 |
| Fine Art | ~4800 |

### Prueba de referencia

```python
def estimate_human_rank(ai_model, test_positions):
    """Estimar la fuerza humana equivalente de la IA"""
    # Usar problemas de prueba estándar
    correct = 0
    for pos in test_positions:
        ai_move = ai_model.get_best_move(pos['state'])
        if ai_move == pos['best_move']:
            correct += 1

    accuracy = correct / len(test_positions)

    # Tabla de correspondencia de precisión
    if accuracy > 0.9:
        return "Nivel profesional"
    elif accuracy > 0.7:
        return "Amateur 5 dan+"
    elif accuracy > 0.5:
        return "Amateur 1-5 dan"
    else:
        return "Inferior a amateur"
```

---

## Monitoreo de rendimiento

### Monitoreo continuo

```python
import time
import psutil
import GPUtil

class PerformanceMonitor:
    def __init__(self):
        self.metrics = []

    def sample(self):
        """Muestrear métricas de rendimiento actuales"""
        gpus = GPUtil.getGPUs()

        self.metrics.append({
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_util': gpus[0].load * 100 if gpus else 0,
            'gpu_memory': gpus[0].memoryUsed if gpus else 0,
        })

    def report(self):
        """Generar informe"""
        if not self.metrics:
            return

        avg_cpu = sum(m['cpu_percent'] for m in self.metrics) / len(self.metrics)
        avg_gpu = sum(m['gpu_util'] for m in self.metrics) / len(self.metrics)

        print(f"Uso promedio de CPU: {avg_cpu:.1f}%")
        print(f"Uso promedio de GPU: {avg_gpu:.1f}%")
```

### Diagnóstico de cuellos de botella

| Síntoma | Causa posible | Solución |
|---------|---------------|----------|
| CPU 100%, GPU baja | Hilos de búsqueda insuficientes | Aumentar numSearchThreads |
| GPU 100%, salida lenta | Lote muy pequeño | Aumentar nnMaxBatchSize |
| Memoria insuficiente | Modelo muy grande | Usar modelo más pequeño |
| Velocidad inestable | Temperatura muy alta | Mejorar refrigeración |

---

## Pruebas automatizadas

### Integración CI/CD

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

### Prueba de regresión

```python
def regression_test(new_model, baseline_model, threshold=0.95):
    """Verificar si el nuevo modelo tiene regresión de rendimiento"""
    # Probar precisión
    new_accuracy = test_accuracy(new_model)
    baseline_accuracy = test_accuracy(baseline_model)

    if new_accuracy < baseline_accuracy * threshold:
        raise Exception(f"Regresión de precisión: {new_accuracy:.3f} < {baseline_accuracy:.3f}")

    # Probar velocidad
    new_speed = benchmark_speed(new_model)
    baseline_speed = benchmark_speed(baseline_model)

    if new_speed < baseline_speed * threshold:
        raise Exception(f"Regresión de velocidad: {new_speed:.1f} < {baseline_speed:.1f}")

    print("Prueba de regresión aprobada")
```

---

## Lectura adicional

- [Análisis del mecanismo de entrenamiento de KataGo](../training) — Cómo se entrena el modelo
- [Arquitectura de entrenamiento distribuido](../distributed-training) — Evaluación a gran escala
- [Backend GPU y optimización](../gpu-optimization) — Ajuste de rendimiento
