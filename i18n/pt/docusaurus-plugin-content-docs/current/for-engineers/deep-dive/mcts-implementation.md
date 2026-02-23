---
sidebar_position: 5
title: Detalhes de Implementação do MCTS
description: Análise aprofundada da implementação da Busca em Árvore de Monte Carlo, seleção PUCT e técnicas de paralelização
---

# Detalhes de Implementação do MCTS

Este artigo analisa em profundidade os detalhes de implementação da Busca em Árvore de Monte Carlo (MCTS) no KataGo, incluindo estruturas de dados, estratégias de seleção e técnicas de paralelização.

---

## Revisão das Quatro Etapas do MCTS

```
┌─────────────────────────────────────────────────────┐
│                  Ciclo de Busca MCTS                │
├─────────────────────────────────────────────────────┤
│                                                     │
│   1. Selection      Seleção: Desce pela árvore      │
│         │           usando PUCT para selecionar nós │
│         ▼                                           │
│   2. Expansion      Expansão: Ao chegar em nó folha │
│         │           cria nós filhos                 │
│         ▼                                           │
│   3. Evaluation     Avaliação: Avalia o nó folha    │
│         │           usando a rede neural            │
│         ▼                                           │
│   4. Backprop       Retropropagação: Atualiza       │
│                     estatísticas de todos os nós    │
│                     no caminho                      │
│                                                     │
│   Repete milhares de vezes, seleciona a jogada     │
│   com mais visitas                                  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Estrutura de Dados do Nó

### Dados Principais

Cada nó do MCTS precisa armazenar:

```python
class MCTSNode:
    def __init__(self, state, parent=None, prior=0.0):
        # Informações básicas
        self.state = state              # Estado do tabuleiro
        self.parent = parent            # Nó pai
        self.children = {}              # Dicionário de filhos {ação: nó}
        self.action = None              # Ação que levou a este nó

        # Informações estatísticas
        self.visit_count = 0            # N(s): Contagem de visitas
        self.value_sum = 0.0            # W(s): Soma dos valores
        self.prior = prior              # P(s,a): Probabilidade a priori

        # Para busca paralela
        self.virtual_loss = 0           # Perda virtual
        self.is_expanded = False        # Se já foi expandido

    @property
    def value(self):
        """Q(s) = W(s) / N(s)"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
```

### Otimização de Memória

O KataGo usa várias técnicas para reduzir o uso de memória:

```python
# Usando arrays numpy em vez de dict Python
class OptimizedNode:
    __slots__ = ['visit_count', 'value_sum', 'prior', 'children_indices']

    def __init__(self):
        self.visit_count = np.int32(0)
        self.value_sum = np.float32(0.0)
        self.prior = np.float32(0.0)
        self.children_indices = None  # Alocação adiada
```

---

## Selection: Seleção PUCT

### Fórmula PUCT

```
Pontuação de seleção = Q(s,a) + U(s,a)

Onde:
Q(s,a) = W(s,a) / N(s,a)              # Valor médio
U(s,a) = c_puct × P(s,a) × √(N(s)) / (1 + N(s,a))  # Termo de exploração
```

### Descrição dos Parâmetros

| Símbolo | Significado | Valor típico |
|---------|-------------|--------------|
| Q(s,a) | Valor médio da ação a | [-1, +1] |
| P(s,a) | Probabilidade a priori da rede neural | [0, 1] |
| N(s) | Contagem de visitas do nó pai | Inteiro |
| N(s,a) | Contagem de visitas da ação a | Inteiro |
| c_puct | Constante de exploração | 1.0 ~ 2.5 |

### Implementação

```python
def select_child(self, c_puct=1.5):
    """Seleciona o nó filho com maior pontuação PUCT"""
    best_score = -float('inf')
    best_action = None
    best_child = None

    # Raiz quadrada das visitas do nó pai
    sqrt_parent_visits = math.sqrt(self.visit_count)

    for action, child in self.children.items():
        # Valor Q (valor médio)
        if child.visit_count > 0:
            q_value = child.value_sum / child.visit_count
        else:
            q_value = 0.0

        # Valor U (termo de exploração)
        u_value = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)

        # Pontuação total
        score = q_value + u_value

        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    return best_action, best_child
```

### Equilíbrio entre Exploração e Aproveitamento

```
Fase inicial: N(s,a) pequeno
├── U(s,a) grande → Foco na exploração
└── Ações com alta probabilidade a priori são exploradas primeiro

Fase posterior: N(s,a) grande
├── U(s,a) pequeno → Foco no aproveitamento
└── Q(s,a) domina, seleciona ações conhecidamente boas
```

---

## Expansion: Expansão de Nós

### Condições de Expansão

Ao chegar em um nó folha, usa a rede neural para expandir:

```python
def expand(self, policy_probs, legal_moves):
    """Expande o nó, criando nós filhos para todas as jogadas legais"""
    for action in legal_moves:
        if action not in self.children:
            prior = policy_probs[action]  # Probabilidade prevista pela rede neural
            child_state = self.state.play(action)
            self.children[action] = MCTSNode(
                state=child_state,
                parent=self,
                prior=prior
            )

    self.is_expanded = True
```

### Filtragem de Jogadas Legais

```python
def get_legal_moves(state):
    """Obtém todas as jogadas legais"""
    legal = []
    for i in range(361):
        x, y = i // 19, i % 19
        if state.is_legal(x, y):
            legal.append(i)

    # Adiciona pass
    legal.append(361)

    return legal
```

---

## Evaluation: Avaliação pela Rede Neural

### Avaliação Única

```python
def evaluate(self, state):
    """Avalia a posição usando a rede neural"""
    # Codifica recursos de entrada
    features = encode_state(state)  # (22, 19, 19)
    features = torch.tensor(features).unsqueeze(0)  # (1, 22, 19, 19)

    # Inferência da rede neural
    with torch.no_grad():
        output = self.network(features)

    policy = output['policy'][0].numpy()  # (362,)
    value = output['value'][0].item()     # escalar

    return policy, value
```

### Avaliação em Lote (Otimização Crucial)

A GPU é mais eficiente com inferência em lote:

```python
class BatchedEvaluator:
    def __init__(self, network, batch_size=8):
        self.network = network
        self.batch_size = batch_size
        self.pending = []  # Lista de (estado, callback) aguardando avaliação

    def request_evaluation(self, state, callback):
        """Solicita avaliação, executa automaticamente quando o lote está cheio"""
        self.pending.append((state, callback))

        if len(self.pending) >= self.batch_size:
            self.flush()

    def flush(self):
        """Executa avaliação em lote"""
        if not self.pending:
            return

        # Prepara entrada em lote
        states = [s for s, _ in self.pending]
        features = torch.stack([encode_state(s) for s in states])

        # Inferência em lote
        with torch.no_grad():
            outputs = self.network(features)

        # Retorna resultados via callback
        for i, (_, callback) in enumerate(self.pending):
            policy = outputs['policy'][i].numpy()
            value = outputs['value'][i].item()
            callback(policy, value)

        self.pending.clear()
```

---

## Backpropagation: Atualização por Retropropagação

### Retropropagação Básica

```python
def backpropagate(self, value):
    """Retropropaga do nó folha até a raiz, atualizando estatísticas"""
    node = self

    while node is not None:
        node.visit_count += 1
        node.value_sum += value

        # Alternância de perspectiva: o valor para o oponente é oposto
        value = -value

        node = node.parent
```

### Importância da Alternância de Perspectiva

```
Perspectiva do Preto: value = +0.6 (Preto em vantagem)

Caminho de retropropagação:
Nó folha (Preto joga): value_sum += +0.6
    ↑
Nó pai (Branco joga): value_sum += -0.6  ← Desfavorável para Branco
    ↑
Nó avô (Preto joga): value_sum += +0.6
    ↑
...
```

---

## Paralelização: Perda Virtual

### O Problema

Quando múltiplas threads buscam simultaneamente, podem todas selecionar o mesmo nó:

```
Thread 1: Seleciona nó A (Q=0.6, N=100)
Thread 2: Seleciona nó A (Q=0.6, N=100) ← Repetido!
Thread 3: Seleciona nó A (Q=0.6, N=100) ← Repetido!
```

### Solução: Perda Virtual

Ao selecionar um nó, primeiro adiciona "perda virtual" para que outras threads não queiram selecioná-lo:

```python
VIRTUAL_LOSS = 3  # Valor da perda virtual

def select_with_virtual_loss(self):
    """Seleção com perda virtual"""
    action, child = self.select_child()

    # Adiciona perda virtual
    child.visit_count += VIRTUAL_LOSS
    child.value_sum -= VIRTUAL_LOSS  # Finge que perdeu

    return action, child

def backpropagate_with_virtual_loss(self, value):
    """Retropropagação removendo perda virtual"""
    node = self

    while node is not None:
        # Remove perda virtual
        node.visit_count -= VIRTUAL_LOSS
        node.value_sum += VIRTUAL_LOSS

        # Atualização normal
        node.visit_count += 1
        node.value_sum += value

        value = -value
        node = node.parent
```

### Efeito

```
Thread 1: Seleciona nó A, adiciona perda virtual
         Valor Q de A cai temporariamente

Thread 2: Seleciona nó B (porque A parece pior agora)

Thread 3: Seleciona nó C

→ Diferentes threads exploram diferentes ramos, aumentando eficiência
```

---

## Implementação Completa da Busca

```python
class MCTS:
    def __init__(self, network, c_puct=1.5, num_simulations=800):
        self.network = network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.evaluator = BatchedEvaluator(network)

    def search(self, root_state):
        """Executa busca MCTS"""
        root = MCTSNode(root_state)

        # Expande nó raiz
        policy, value = self.evaluate(root_state)
        legal_moves = get_legal_moves(root_state)
        root.expand(policy, legal_moves)

        # Executa simulações
        for _ in range(self.num_simulations):
            node = root
            path = [node]

            # Selection: Desce pela árvore
            while node.is_expanded and node.children:
                action, node = node.select_child(self.c_puct)
                path.append(node)

            # Expansion + Evaluation
            if not node.is_expanded:
                policy, value = self.evaluate(node.state)
                legal_moves = get_legal_moves(node.state)

                if legal_moves:
                    node.expand(policy, legal_moves)

            # Backpropagation
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += value
                value = -value

        # Seleciona a ação com mais visitas
        best_action = max(root.children.items(),
                         key=lambda x: x[1].visit_count)[0]

        return best_action

    def evaluate(self, state):
        features = encode_state(state)
        features = torch.tensor(features).unsqueeze(0)

        with torch.no_grad():
            output = self.network(features)

        return output['policy'][0].numpy(), output['value'][0].item()
```

---

## Técnicas Avançadas

### Ruído de Dirichlet

Adiciona ruído no nó raiz durante o treinamento para aumentar a exploração:

```python
def add_dirichlet_noise(root, alpha=0.03, epsilon=0.25):
    """Adiciona ruído de Dirichlet ao nó raiz"""
    noise = np.random.dirichlet([alpha] * len(root.children))

    for i, child in enumerate(root.children.values()):
        child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]
```

### Parâmetro de Temperatura

Controla a aleatoriedade na seleção de ações:

```python
def select_action_with_temperature(root, temperature=1.0):
    """Seleciona ação baseado na contagem de visitas e temperatura"""
    visits = np.array([c.visit_count for c in root.children.values()])
    actions = list(root.children.keys())

    if temperature == 0:
        # Seleção gulosa
        return actions[np.argmax(visits)]
    else:
        # Seleciona de acordo com distribuição de probabilidade das visitas
        probs = visits ** (1 / temperature)
        probs = probs / probs.sum()
        return np.random.choice(actions, p=probs)
```

### Reutilização de Árvore

A nova jogada pode reutilizar a árvore de busca anterior:

```python
def reuse_tree(root, action):
    """Reutiliza subárvore"""
    if action in root.children:
        new_root = root.children[action]
        new_root.parent = None
        return new_root
    else:
        return None  # Precisa criar nova árvore
```

---

## Resumo de Otimizações de Desempenho

| Técnica | Efeito |
|---------|--------|
| **Avaliação em lote** | Utilização da GPU de 10% → 80%+ |
| **Perda virtual** | Eficiência multi-thread aumenta 3-5x |
| **Reutilização de árvore** | Reduz cold start, economiza 30%+ de computação |
| **Pool de memória** | Reduz overhead de alocação de memória |

---

## Leitura Adicional

- [Arquitetura de Rede Neural Detalhada](../neural-network) — Origem da função de avaliação
- [Backend GPU e Otimização](../gpu-optimization) — Otimização de hardware para inferência em lote
- [Guia de Artigos Importantes](../papers) — Base teórica da fórmula PUCT
