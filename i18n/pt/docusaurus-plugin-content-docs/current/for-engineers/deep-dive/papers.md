---
sidebar_position: 11
title: Guia de Artigos Importantes
description: "Analise dos pontos-chave dos artigos marcos da IA de Go: AlphaGo, AlphaZero, KataGo"
---

# Guia de Artigos Importantes

Este artigo organiza os artigos mais importantes na historia do desenvolvimento da IA de Go, fornecendo resumos e pontos tecnicos para compreensao rapida.

---

## Visao Geral dos Artigos

### Linha do Tempo

```
2006  Coulom - MCTS aplicado ao Go pela primeira vez
2016  Silver et al. - AlphaGo (Nature)
2017  Silver et al. - AlphaGo Zero (Nature)
2017  Silver et al. - AlphaZero
2019  Wu - KataGo
2020+ Varias melhorias e aplicacoes
```

### Sugestoes de Leitura

| Objetivo | Artigo Recomendado |
|----------|-------------------|
| Entender o basico | AlphaGo (2016) |
| Entender auto-jogo | AlphaGo Zero (2017) |
| Entender metodo geral | AlphaZero (2017) |
| Referencia de implementacao | KataGo (2019) |

---

## 1. O Nascimento do MCTS (2006)

### Informacoes do Artigo

```
Titulo: Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search
Autor: Remi Coulom
Publicacao: Computers and Games 2006
```

### Contribuicao Principal

Primeira aplicacao sistematica de metodos de Monte Carlo ao Go:

```
Antes: Simulacao puramente aleatoria, sem estrutura de arvore
Depois: Construcao de arvore de busca + selecao UCB + retropropagacao de estatisticas
```

### Conceitos-Chave

#### Formula UCB1

```
Pontuacao de selecao = Taxa media de vitoria + C × sqrt(ln(N) / n)

Onde:
- N: Contagem de visitas do no pai
- n: Contagem de visitas do no filho
- C: Constante de exploracao
```

#### Quatro Etapas do MCTS

```
1. Selection: Seleciona nos usando UCB
2. Expansion: Expande novos nos
3. Simulation: Simula aleatoriamente ate o fim
4. Backpropagation: Retropropaga vitoria/derrota
```

### Impacto

- Elevou a IA de Go ao nivel amador dan
- Tornou-se a base de todas as IAs de Go subsequentes
- Conceito UCB influenciou o desenvolvimento do PUCT

---

## 2. AlphaGo (2016)

### Informacoes do Artigo

```
Titulo: Mastering the game of Go with deep neural networks and tree search
Autores: Silver, D., Huang, A., Maddison, C.J., et al.
Publicacao: Nature, 2016
DOI: 10.1038/nature16961
```

### Contribuicao Principal

**Primeira combinacao de deep learning com MCTS**, derrotando o campeao mundial humano.

### Arquitetura do Sistema

```
┌─────────────────────────────────────────────┐
│              Arquitetura AlphaGo            │
├─────────────────────────────────────────────┤
│                                             │
│   Policy Network (SL)                       │
│   ├── Entrada: Estado do tabuleiro          │
│   │   (48 planos de recursos)               │
│   ├── Arquitetura: CNN de 13 camadas        │
│   ├── Saida: Probabilidade de 361 posicoes  │
│   └── Treinamento: 30 milhoes de registros  │
│       humanos                               │
│                                             │
│   Policy Network (RL)                       │
│   ├── Inicializado a partir de SL Policy    │
│   └── Aprendizado por reforco via auto-jogo │
│                                             │
│   Value Network                             │
│   ├── Entrada: Estado do tabuleiro          │
│   ├── Saida: Valor unico de taxa de vitoria │
│   └── Treinamento: Posicoes geradas por     │
│       auto-jogo                             │
│                                             │
│   MCTS                                      │
│   ├── Usa Policy Network para guiar busca   │
│   └── Usa Value Network + Rollout para      │
│       avaliacao                             │
│                                             │
└─────────────────────────────────────────────┘
```

### Pontos Tecnicos

#### 1. Policy Network com Aprendizado Supervisionado

```python
# Recursos de entrada (48 planos)
- Posicao das pedras proprias
- Posicao das pedras do oponente
- Numero de liberdades
- Estado apos captura
- Posicoes de jogadas legais
- Posicoes das ultimas jogadas
...
```

#### 2. Melhoria com Aprendizado por Reforco

```
SL Policy → Auto-jogo → RL Policy

RL Policy e ~80% mais forte que SL Policy em taxa de vitoria
```

#### 3. Treinamento da Value Network

```
Chave para evitar overfitting:
- Pegar apenas uma posicao de cada jogo
- Evitar repeticao de posicoes similares
```

#### 4. Integracao MCTS

```
Avaliacao de no folha = 0.5 × Value Network + 0.5 × Rollout

Rollout usa Policy Network rapida (menor precisao mas mais velocidade)
```

### Dados-Chave

| Item | Valor |
|------|-------|
| Precisao SL Policy | 57% |
| Taxa de vitoria RL Policy vs SL Policy | 80% |
| GPUs de treinamento | 176 |
| TPUs de jogo | 48 |

---

## 3. AlphaGo Zero (2017)

### Informacoes do Artigo

```
Titulo: Mastering the game of Go without human knowledge
Autores: Silver, D., Schrittwieser, J., Simonyan, K., et al.
Publicacao: Nature, 2017
DOI: 10.1038/nature24270
```

### Contribuicao Principal

**Nao precisa de registros humanos**, aprende do zero por auto-aprendizado.

### Diferencas em Relacao ao AlphaGo

| Aspecto | AlphaGo | AlphaGo Zero |
|---------|---------|--------------|
| Registros humanos | Precisa | **Nao precisa** |
| Numero de redes | 4 | **1 com duas cabecas** |
| Planos de entrada | 48 | **17** |
| Rollout | Usa | **Nao usa** |
| Rede residual | Nao | **Sim** |
| Tempo de treinamento | Meses | **3 dias** |

### Inovacoes-Chave

#### 1. Rede Unica com Duas Cabecas

```
              Entrada (17 planos)
                   │
              ┌────┴────┐
              │ Torre    │
              │ Residual │
              │ (19 ou   │
              │  39 cam.)│
              └────┬────┘
           ┌──────┴──────┐
           │             │
        Policy         Value
        (361)          (1)
```

#### 2. Recursos de Entrada Simplificados

```python
# Apenas 17 planos de recursos necessarios
features = [
    current_player_stones,      # Pedras proprias
    opponent_stones,            # Pedras do oponente
    history_1_player,           # Estado historico 1
    history_1_opponent,
    ...                         # Estados historicos 2-7
    color_to_play               # De quem e a vez
]
```

#### 3. Avaliacao Pura com Value Network

```
Nao usa mais Rollout
Avaliacao de no folha = Saida da Value Network

Mais simples e rapido
```

#### 4. Fluxo de Treinamento

```
Inicializar rede aleatoria
    │
    ▼
┌─────────────────────────────┐
│  Auto-jogo gera registros   │ ←─┐
└──────────────┬──────────────┘   │
               │                   │
               ▼                   │
┌─────────────────────────────┐   │
│  Treinar rede neural        │   │
│  - Policy: minimizar        │   │
│    entropia cruzada         │   │
│  - Value: minimizar MSE     │   │
└──────────────┬──────────────┘   │
               │                   │
               ▼                   │
┌─────────────────────────────┐   │
│  Avaliar nova rede          │   │
│  Se mais forte, substitui   │───┘
└─────────────────────────────┘
```

### Curva de Aprendizado

```
Tempo de treinamento    Elo
─────────────────────────────
3 horas                 Iniciante
24 horas                Supera AlphaGo Lee
72 horas                Supera AlphaGo Master
```

---

## 4. AlphaZero (2017)

### Informacoes do Artigo

```
Titulo: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
Autores: Silver, D., Hubert, T., Schrittwieser, J., et al.
Publicacao: arXiv:1712.01815 (depois publicado na Science, 2018)
```

### Contribuicao Principal

**Generalizacao**: Mesmo algoritmo aplicado a Go, xadrez e shogi.

### Arquitetura Geral

```
Codificacao de entrada (especifica do jogo) → Rede residual (geral) → Saida com duas cabecas (geral)
```

### Adaptacao Entre Jogos

| Jogo | Planos de Entrada | Espaco de Acoes | Tempo de Treinamento |
|------|-------------------|-----------------|---------------------|
| Go | 17 | 362 | 40 dias |
| Xadrez | 119 | 4672 | 9 horas |
| Shogi | 362 | 11259 | 12 horas |

### Melhorias no MCTS

#### Formula PUCT

```
Pontuacao de selecao = Q(s,a) + c(s) × P(s,a) × sqrt(N(s)) / (1 + N(s,a))

c(s) = log((1 + N(s) + c_base) / c_base) + c_init
```

#### Ruido de Exploracao

```python
# Adicionar ruido de Dirichlet no no raiz
P(s,a) = (1 - epsilon) × p_a + epsilon × eta_a

eta ~ Dir(alpha)
alpha = 0.03 (Go), 0.3 (xadrez), 0.15 (shogi)
```

---

## 5. KataGo (2019)

### Informacoes do Artigo

```
Titulo: Accelerating Self-Play Learning in Go
Autor: David J. Wu
Publicacao: arXiv:1902.10565
```

### Contribuicao Principal

**Aumento de eficiencia de 50x**, permitindo que desenvolvedores individuais treinem IAs de Go poderosas.

### Inovacoes-Chave

#### 1. Objetivos de Treinamento Auxiliares

```
Perda total = Policy Loss + Value Loss +
              Score Loss + Ownership Loss + ...

Objetivos auxiliares fazem a rede convergir mais rapido
```

#### 2. Recursos Globais

```python
# Camada de pooling global
global_features = global_avg_pool(conv_features)
# Combinar com recursos locais
combined = concat(conv_features, broadcast(global_features))
```

#### 3. Randomizacao de Playout Cap

```
Tradicional: Busca fixa de N vezes
KataGo: N amostrado de uma distribuicao

Permite que a rede tenha bom desempenho em varias profundidades de busca
```

#### 4. Tamanho de Tabuleiro Progressivo

```python
if training_step < 1000000:
    board_size = random.choice([9, 13, 19])
else:
    board_size = 19
```

### Comparacao de Eficiencia

| Metrica | AlphaZero | KataGo |
|---------|-----------|--------|
| Dias de GPU para nivel sobre-humano | 5000 | **100** |
| Aumento de eficiencia | Base | **50x** |

---

## 6. Artigos Relacionados

### MuZero (2020)

```
Titulo: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model
Contribuicao: Aprende modelo de dinamica do ambiente, nao precisa de regras do jogo
```

### EfficientZero (2021)

```
Titulo: Mastering Atari Games with Limited Data
Contribuicao: Grande melhoria na eficiencia de amostragem
```

### Gumbel AlphaZero (2022)

```
Titulo: Policy Improvement by Planning with Gumbel
Contribuicao: Metodo melhorado de melhoria de politica
```

---

## Sugestoes de Leitura de Artigos

### Ordem para Iniciantes

```
1. AlphaGo (2016) - Entender arquitetura basica
2. AlphaGo Zero (2017) - Entender auto-jogo
3. KataGo (2019) - Entender detalhes de implementacao
```

### Ordem Avancada

```
4. AlphaZero (2017) - Generalizacao
5. MuZero (2020) - Aprender modelo do mundo
6. Artigo original do MCTS - Entender fundamentos
```

### Tecnicas de Leitura

1. **Ver resumo e conclusao primeiro**: Captar rapidamente a contribuicao principal
2. **Ver figuras e tabelas**: Entender arquitetura geral
3. **Ver secao de metodos**: Entender detalhes tecnicos
4. **Ver apendice**: Encontrar detalhes de implementacao e hiperparametros

---

## Links de Recursos

### PDFs dos Artigos

| Artigo | Link |
|--------|------|
| AlphaGo | [Nature](https://www.nature.com/articles/nature16961) |
| AlphaGo Zero | [Nature](https://www.nature.com/articles/nature24270) |
| AlphaZero | [Science](https://www.science.org/doi/10.1126/science.aar6404) |
| KataGo | [arXiv](https://arxiv.org/abs/1902.10565) |

### Implementacoes Open Source

| Projeto | Link |
|---------|------|
| KataGo | [GitHub](https://github.com/lightvector/KataGo) |
| Leela Zero | [GitHub](https://github.com/leela-zero/leela-zero) |
| MiniGo | [GitHub](https://github.com/tensorflow/minigo) |

---

## Leitura Adicional

- [Arquitetura de Rede Neural Detalhada](../neural-network) — Entendimento profundo do design de redes
- [Detalhes de Implementacao do MCTS](../mcts-implementation) — Implementacao do algoritmo de busca
- [Analise do Mecanismo de Treinamento do KataGo](../training) — Detalhes do fluxo de treinamento
