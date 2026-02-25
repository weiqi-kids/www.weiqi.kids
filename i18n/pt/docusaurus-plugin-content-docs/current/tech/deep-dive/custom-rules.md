---
sidebar_position: 10
title: Regras Personalizadas e Variantes
description: Conjuntos de regras de Go suportados pelo KataGo, variantes de tabuleiro e configuracoes personalizadas detalhadas
---

# Regras Personalizadas e Variantes

Este artigo apresenta os diversos conjuntos de regras de Go suportados pelo KataGo, variantes de tamanho de tabuleiro e como personalizar configuracoes de regras.

---

## Visao Geral dos Conjuntos de Regras

### Comparacao das Principais Regras

| Conjunto de Regras | Metodo de Contagem | Komi | Suicidio | Reversao de Ko |
|-------------------|-------------------|------|----------|----------------|
| **Chinese** | Contagem de area | 7.5 | Proibido | Proibido |
| **Japanese** | Contagem de territorio | 6.5 | Proibido | Proibido |
| **Korean** | Contagem de territorio | 6.5 | Proibido | Proibido |
| **AGA** | Misto | 7.5 | Proibido | Proibido |
| **New Zealand** | Contagem de area | 7 | Permitido | Proibido |
| **Tromp-Taylor** | Contagem de area | 7.5 | Permitido | Proibido |

### Configuracao do KataGo

```ini
# config.cfg
rules = chinese           # Conjunto de regras
komi = 7.5               # Komi
boardXSize = 19          # Largura do tabuleiro
boardYSize = 19          # Altura do tabuleiro
```

---

## Regras Chinesas (Chinese)

### Caracteristicas

```
Metodo de contagem: Contagem de area (pedras + territorio)
Komi: 7.5 pontos
Suicidio: Proibido
Reversao de Ko: Proibido (regras simplificadas)
```

### Explicacao da Contagem de Area

```
Pontuacao final = Pedras proprias + Pontos vazios proprios

Exemplo:
Preto 120 pedras + 65 pontos de territorio = 185 pontos
Branco 100 pedras + 75 pontos + komi 7.5 = 182.5 pontos
Preto vence por 2.5 pontos
```

### Configuracao do KataGo

```ini
rules = chinese
komi = 7.5
```

---

## Regras Japonesas (Japanese)

### Caracteristicas

```
Metodo de contagem: Contagem de territorio (apenas pontos vazios)
Komi: 6.5 pontos
Suicidio: Proibido
Reversao de Ko: Proibido
Requer marcacao de pedras mortas
```

### Explicacao da Contagem de Territorio

```
Pontuacao final = Pontos vazios proprios + Pedras capturadas do oponente

Exemplo:
Territorio preto 65 pontos + 10 capturas = 75 pontos
Territorio branco 75 pontos + 5 capturas + komi 6.5 = 86.5 pontos
Branco vence por 11.5 pontos
```

### Determinacao de Pedras Mortas

Regras japonesas requerem acordo de ambos os jogadores sobre quais pedras estao mortas:

```python
def is_dead_by_japanese_rules(group, game_state):
    """Determina pedras mortas sob regras japonesas"""
    # Precisa provar que o grupo nao pode fazer dois olhos
    # Esta e a complexidade das regras japonesas
    pass
```

### Configuracao do KataGo

```ini
rules = japanese
komi = 6.5
```

---

## Regras AGA

### Caracteristicas

As regras da American Go Association (AGA) combinam vantagens das regras chinesas e japonesas:

```
Metodo de contagem: Misto (area ou territorio, resultado igual)
Komi: 7.5 pontos
Suicidio: Proibido
Branco precisa preencher um ponto para passar
```

### Regra de Pass

```
Preto passa: Nao precisa preencher
Branco passa: Precisa entregar uma pedra ao Preto

Isso faz com que contagem de area e territorio deem o mesmo resultado
```

### Configuracao do KataGo

```ini
rules = aga
komi = 7.5
```

---

## Regras Tromp-Taylor

### Caracteristicas

As regras de Go mais concisas, adequadas para implementacao em programas:

```
Metodo de contagem: Contagem de area
Komi: 7.5 pontos
Suicidio: Permitido
Reversao de Ko: Super Ko (proibe qualquer posicao repetida)
Nao precisa determinar pedras mortas
```

### Super Ko

```python
def is_superko_violation(new_state, history):
    """Verifica se viola Super Ko"""
    for past_state in history:
        if new_state == past_state:
            return True
    return False
```

### Determinacao de Fim de Jogo

```
Nao precisa de acordo sobre pedras mortas
O jogo continua ate:
1. Ambos passarem consecutivamente
2. Depois usa busca ou joga ate o fim para determinar territorio
```

### Configuracao do KataGo

```ini
rules = tromp-taylor
komi = 7.5
```

---

## Variantes de Tamanho de Tabuleiro

### Tamanhos Suportados

KataGo suporta varios tamanhos de tabuleiro:

| Tamanho | Caracteristicas | Uso Recomendado |
|---------|-----------------|-----------------|
| 9x9 | ~81 pontos | Iniciantes, partidas rapidas |
| 13x13 | ~169 pontos | Aprendizado avancado |
| 19x19 | 361 pontos | Competicoes padrao |
| Personalizado | Qualquer | Pesquisa, testes |

### Metodo de Configuracao

```ini
# Tabuleiro 9x9
boardXSize = 9
boardYSize = 9
komi = 5.5

# Tabuleiro 13x13
boardXSize = 13
boardYSize = 13
komi = 6.5

# Tabuleiro nao quadrado
boardXSize = 19
boardYSize = 9
```

### Komi Recomendado

| Tamanho | Regras Chinesas | Regras Japonesas |
|---------|-----------------|------------------|
| 9x9 | 5.5 | 5.5 |
| 13x13 | 6.5 | 6.5 |
| 19x19 | 7.5 | 6.5 |

---

## Configuracao de Handicap

### Partidas com Handicap

Handicap e uma forma de ajustar a diferenca de forca:

```ini
# Handicap de 2 pedras
handicap = 2

# Handicap de 9 pedras
handicap = 9
```

### Posicoes de Handicap

```python
HANDICAP_POSITIONS = {
    2: [(3, 15), (15, 3)],
    3: [(3, 15), (15, 3), (15, 15)],
    4: [(3, 15), (15, 3), (3, 3), (15, 15)],
    # 5-9 pedras usam pontos estrela + tengen
}
```

### Komi com Handicap

```ini
# Tradicional: Sem komi ou meio ponto com handicap
komi = 0.5

# Moderno: Ajustar conforme numero de pedras
# Cada pedra vale aproximadamente 10-15 pontos
```

---

## Configuracao de Regras no Modo Analysis

### Comandos GTP

```gtp
# Definir regras
kata-set-rules chinese

# Definir komi
komi 7.5

# Definir tamanho do tabuleiro
boardsize 19
```

### API de Analysis

```json
{
  "id": "query1",
  "moves": [["B", "Q4"], ["W", "D4"]],
  "rules": "chinese",
  "komi": 7.5,
  "boardXSize": 19,
  "boardYSize": 19,
  "overrideSettings": {
    "maxVisits": 1000
  }
}
```

---

## Opcoes de Regras Avancadas

### Configuracao de Suicidio

```ini
# Proibir suicidio (padrao)
allowSuicide = false

# Permitir suicidio (estilo Tromp-Taylor)
allowSuicide = true
```

### Regras de Ko

```ini
# Simple Ko (proibe apenas reversao imediata)
koRule = SIMPLE

# Positional Super Ko (proibe repeticao de qualquer posicao, independente de quem joga)
koRule = POSITIONAL

# Situational Super Ko (proibe repeticao da posicao jogada pelo mesmo lado)
koRule = SITUATIONAL
```

### Regras de Pontuacao

```ini
# Contagem de area (Chinesa, AGA)
scoringRule = AREA

# Contagem de territorio (Japonesa, Coreana)
scoringRule = TERRITORY
```

### Regras de Imposto

Algumas regras tem pontuacao especial para areas de seki:

```ini
# Sem imposto
taxRule = NONE

# Seki sem pontos
taxRule = SEKI

# Todos os olhos sem pontos
taxRule = ALL
```

---

## Treinamento Multi-Regras

### Vantagem do KataGo

KataGo usa um unico modelo para suportar multiplas regras:

```python
def encode_rules(rules):
    """Codifica regras como entrada da rede neural"""
    features = np.zeros(RULE_FEATURE_SIZE)

    # Metodo de pontuacao
    features[0] = 1.0 if rules.scoring == 'area' else 0.0

    # Suicidio
    features[1] = 1.0 if rules.allow_suicide else 0.0

    # Regra de Ko
    features[2:5] = encode_ko_rule(rules.ko)

    # Komi (normalizado)
    features[5] = rules.komi / 15.0

    return features
```

### Entrada Consciente de Regras

```
Entrada da rede neural inclui:
- Estado do tabuleiro (19x19xN)
- Vetor de caracteristicas de regras (K dimensoes)

Isso permite que o mesmo modelo entenda diferentes regras
```

---

## Exemplo de Troca de Regras

### Codigo Python

```python
from katago import KataGo

engine = KataGo(model_path="kata.bin.gz")

# Analise com regras chinesas
result_cn = engine.analyze(
    moves=[("B", "Q4"), ("W", "D4")],
    rules="chinese",
    komi=7.5
)

# Analise com regras japonesas (mesma posicao)
result_jp = engine.analyze(
    moves=[("B", "Q4"), ("W", "D4")],
    rules="japanese",
    komi=6.5
)

# Comparar diferencas
print(f"Taxa de vitoria do Preto (regras chinesas): {result_cn['winrate']:.1%}")
print(f"Taxa de vitoria do Preto (regras japonesas): {result_jp['winrate']:.1%}")
```

### Analise de Impacto das Regras

```python
def compare_rules_impact(position, rules_list):
    """Compara impacto de diferentes regras na avaliacao da posicao"""
    results = {}

    for rules in rules_list:
        analysis = engine.analyze(
            moves=position,
            rules=rules,
            komi=get_default_komi(rules)
        )
        results[rules] = {
            'winrate': analysis['winrate'],
            'score': analysis['scoreLead'],
            'best_move': analysis['moveInfos'][0]['move']
        }

    return results
```

---

## Perguntas Frequentes

### Diferencas de Regras Causando Resultados Diferentes

```
O mesmo jogo pode ter resultados diferentes com regras diferentes:
- Diferencas de pontuacao entre contagem de area e territorio
- Tratamento de areas de seki
- Impacto de jogadas virtuais (pass)
```

### Qual Regra Escolher?

| Cenario | Regra Recomendada |
|---------|-------------------|
| Iniciantes | Chinese (intuitiva, sem controversias) |
| Competicoes online | Padrao da plataforma (geralmente Chinese) |
| Associacao Japonesa | Japanese |
| Implementacao em programas | Tromp-Taylor (mais concisa) |
| Competicoes profissionais chinesas | Chinese |

### O Modelo Precisa Ser Treinado para Regras Especificas?

O modelo multi-regras do KataGo ja e muito forte. Mas se usar apenas uma regra, pode considerar:

```ini
# Treinamento com regra fixa (pode melhorar levemente a forca sob essa regra especifica)
rules = chinese
```

---

## Leitura Adicional

- [Analise do Mecanismo de Treinamento do KataGo](../training) — Implementacao de treinamento multi-regras
- [Integracao no Seu Projeto](../../hands-on/integration) — Exemplos de uso da API
- [Avaliacao e Benchmarks](../evaluation) — Testes de forca sob diferentes regras
