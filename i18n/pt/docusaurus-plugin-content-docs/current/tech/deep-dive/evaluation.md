---
sidebar_position: 9
title: Avaliacao e Benchmarks
description: Sistema de classificacao Elo para IA de Go, testes de partidas e metodos de benchmark de desempenho
---

# Avaliacao e Benchmarks

Este artigo apresenta como avaliar a forca de jogo e o desempenho de uma IA de Go, incluindo o sistema de classificacao Elo, metodos de teste de partidas e benchmarks padrao.

---

## Sistema de Classificacao Elo

### Conceito Basico

A classificacao Elo e o metodo padrao para medir forca de jogo relativa:

```
Taxa de vitoria esperada E_A = 1 / (1 + 10^((R_B - R_A) / 400))

Novo Elo = Elo antigo + K × (resultado real - resultado esperado)
```

### Diferenca de Elo vs Taxa de Vitoria

| Diferenca de Elo | Taxa de Vitoria do Mais Forte |
|------------------|------------------------------|
| 0 | 50% |
| 100 | 64% |
| 200 | 76% |
| 400 | 91% |
| 800 | 99% |

### Implementacao

```python
def expected_score(rating_a, rating_b):
    """Calcula pontuacao esperada de A contra B"""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating, expected, actual, k=32):
    """Atualiza classificacao Elo"""
    return rating + k * (actual - expected)

def calculate_elo_diff(wins, losses, draws):
    """Calcula diferenca de Elo a partir dos resultados"""
    total = wins + losses + draws
    win_rate = (wins + 0.5 * draws) / total

    if win_rate <= 0 or win_rate >= 1:
        return float('inf') if win_rate >= 1 else float('-inf')

    return 400 * math.log10(win_rate / (1 - win_rate))
```

---

## Testes de Partidas

### Framework de Teste

```python
class MatchTester:
    def __init__(self, engine_a, engine_b):
        self.engine_a = engine_a
        self.engine_b = engine_b
        self.results = {'a_wins': 0, 'b_wins': 0, 'draws': 0}

    def run_match(self, num_games=400):
        """Executa teste de partidas"""
        for i in range(num_games):
            # Alterna primeiro a jogar
            if i % 2 == 0:
                black, white = self.engine_a, self.engine_b
                a_is_black = True
            else:
                black, white = self.engine_b, self.engine_a
                a_is_black = False

            # Jogar partida
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
        """Joga uma partida"""
        game = Game()

        while not game.is_terminal():
            if game.current_player == 'black':
                move = black_engine.get_move(game.state)
            else:
                move = white_engine.get_move(game.state)

            game.play(move)

        return game.get_winner()
```

### Significancia Estatistica

Garanta que os resultados do teste sejam estatisticamente significativos:

```python
from scipy import stats

def calculate_confidence_interval(wins, total, confidence=0.95):
    """Calcula intervalo de confianca da taxa de vitoria"""
    p = wins / total
    z = stats.norm.ppf((1 + confidence) / 2)
    margin = z * math.sqrt(p * (1 - p) / total)

    return (p - margin, p + margin)

# Exemplo
wins, total = 220, 400
ci_low, ci_high = calculate_confidence_interval(wins, total)
print(f"Taxa de vitoria: {wins/total:.1%}, IC 95%: [{ci_low:.1%}, {ci_high:.1%}]")
```

### Numero de Partidas Recomendado

| Diferenca de Elo Esperada | Partidas Recomendadas | Confianca |
|---------------------------|----------------------|-----------|
| \>100 | 100 | 95% |
| 50-100 | 200 | 95% |
| 20-50 | 400 | 95% |
| \<20 | 1000+ | 95% |

---

## SPRT (Teste de Razao de Probabilidade Sequencial)

### Conceito

Nao precisa de numero fixo de partidas, decide dinamicamente quando parar baseado nos resultados acumulados:

```python
def sprt(wins, losses, elo0=0, elo1=10, alpha=0.05, beta=0.05):
    """
    Teste de Razao de Probabilidade Sequencial

    elo0: Diferenca de Elo da hipotese nula (geralmente 0)
    elo1: Diferenca de Elo da hipotese alternativa (geralmente 5-20)
    alpha: Taxa de falso positivo
    beta: Taxa de falso negativo
    """
    if wins + losses == 0:
        return 'continue'

    # Calcular log-verossimilhanca
    p0 = expected_score(elo1, 0)  # Taxa de vitoria esperada sob H1
    p1 = expected_score(elo0, 0)  # Taxa de vitoria esperada sob H0

    llr = (
        wins * math.log(p0 / p1) +
        losses * math.log((1 - p0) / (1 - p1))
    )

    # Limites de decisao
    lower = math.log(beta / (1 - alpha))
    upper = math.log((1 - beta) / alpha)

    if llr <= lower:
        return 'reject'  # H0 rejeitada, novo modelo e pior
    elif llr >= upper:
        return 'accept'  # H0 aceita, novo modelo e melhor
    else:
        return 'continue'  # Continuar testando
```

---

## Benchmark do KataGo

### Executar Benchmark

```bash
# Teste basico
katago benchmark -model model.bin.gz

# Especificar numero de visitas
katago benchmark -model model.bin.gz -v 1000

# Saida detalhada
katago benchmark -model model.bin.gz -v 1000 -t 8
```

### Interpretacao da Saida

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

### Metricas Principais

| Metrica | Descricao | Bom Valor |
|---------|-----------|-----------|
| NN evals/seg | Velocidade de avaliacao da rede neural | >1000 |
| Playouts/seg | Velocidade de simulacao MCTS | >2000 |
| Utilizacao GPU | Eficiencia de uso da GPU | >80% |

---

## Avaliacao de Forca de Jogo

### Correspondencia com Forca Humana

| Elo IA | Forca Humana |
|--------|--------------|
| ~1500 | Amador 1 dan |
| ~2000 | Amador 5 dan |
| ~2500 | Profissional iniciante |
| ~3000 | Profissional 5 dan |
| ~3500 | Nivel campeonato mundial |
| ~4000+ | Acima do nivel humano |

### Elo das Principais IAs

| IA | Elo (estimado) |
|----|----------------|
| KataGo (mais recente) | ~5000 |
| AlphaGo Zero | ~5000 |
| Leela Zero | ~4500 |
| FineArt (Jueyi) | ~4800 |

### Teste de Referencia

```python
def estimate_human_rank(ai_model, test_positions):
    """Estima forca humana equivalente da IA"""
    # Usar questoes de teste padrao
    correct = 0
    for pos in test_positions:
        ai_move = ai_model.get_best_move(pos['state'])
        if ai_move == pos['best_move']:
            correct += 1

    accuracy = correct / len(test_positions)

    # Tabela de correspondencia de precisao
    if accuracy > 0.9:
        return "Nivel profissional"
    elif accuracy > 0.7:
        return "Amador 5 dan+"
    elif accuracy > 0.5:
        return "Amador 1-5 dan"
    else:
        return "Abaixo do nivel amador"
```

---

## Monitoramento de Desempenho

### Monitoramento Continuo

```python
import time
import psutil
import GPUtil

class PerformanceMonitor:
    def __init__(self):
        self.metrics = []

    def sample(self):
        """Amostra metricas de desempenho atuais"""
        gpus = GPUtil.getGPUs()

        self.metrics.append({
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_util': gpus[0].load * 100 if gpus else 0,
            'gpu_memory': gpus[0].memoryUsed if gpus else 0,
        })

    def report(self):
        """Gera relatorio"""
        if not self.metrics:
            return

        avg_cpu = sum(m['cpu_percent'] for m in self.metrics) / len(self.metrics)
        avg_gpu = sum(m['gpu_util'] for m in self.metrics) / len(self.metrics)

        print(f"Uso medio de CPU: {avg_cpu:.1f}%")
        print(f"Uso medio de GPU: {avg_gpu:.1f}%")
```

### Diagnostico de Gargalos de Desempenho

| Sintoma | Causa Possivel | Solucao |
|---------|----------------|---------|
| CPU 100%, GPU baixa | Threads de busca insuficientes | Aumentar numSearchThreads |
| GPU 100%, saida lenta | Lote muito pequeno | Aumentar nnMaxBatchSize |
| Memoria insuficiente | Modelo muito grande | Usar modelo menor |
| Velocidade instavel | Temperatura muito alta | Melhorar refrigeracao |

---

## Testes Automatizados

### Integracao CI/CD

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

### Teste de Regressao

```python
def regression_test(new_model, baseline_model, threshold=0.95):
    """Verifica se novo modelo tem regressao de desempenho"""
    # Testar precisao
    new_accuracy = test_accuracy(new_model)
    baseline_accuracy = test_accuracy(baseline_model)

    if new_accuracy < baseline_accuracy * threshold:
        raise Exception(f"Regressao de precisao: {new_accuracy:.3f} < {baseline_accuracy:.3f}")

    # Testar velocidade
    new_speed = benchmark_speed(new_model)
    baseline_speed = benchmark_speed(baseline_model)

    if new_speed < baseline_speed * threshold:
        raise Exception(f"Regressao de velocidade: {new_speed:.1f} < {baseline_speed:.1f}")

    print("Teste de regressao passou")
```

---

## Leitura Adicional

- [Analise do Mecanismo de Treinamento do KataGo](../training) — Como os modelos sao treinados
- [Arquitetura de Treinamento Distribuido](../distributed-training) — Avaliacao em larga escala
- [Backend GPU e Otimizacao](../gpu-optimization) — Ajuste de desempenho
