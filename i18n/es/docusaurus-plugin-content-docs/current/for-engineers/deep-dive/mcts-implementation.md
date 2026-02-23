---
sidebar_position: 5
title: Detalles de implementación de MCTS
description: Análisis profundo de la implementación de Monte Carlo Tree Search, selección PUCT y técnicas de paralelización
---

# Detalles de implementación de MCTS

Este artículo analiza en profundidad los detalles de implementación de Monte Carlo Tree Search (MCTS) en KataGo, incluyendo estructuras de datos, estrategias de selección y técnicas de paralelización.

---

## Repaso de los cuatro pasos de MCTS

```
┌─────────────────────────────────────────────────────┐
│                Ciclo de búsqueda MCTS               │
├─────────────────────────────────────────────────────┤
│                                                     │
│   1. Selection    Selección: bajar por el árbol    │
│         │         usando PUCT para elegir nodos    │
│         ▼                                           │
│   2. Expansion    Expansión: al llegar a una hoja, │
│         │         crear nodos hijos                 │
│         ▼                                           │
│   3. Evaluation   Evaluación: evaluar el nodo hoja │
│         │         con la red neuronal               │
│         ▼                                           │
│   4. Backprop     Retropropagación: actualizar     │
│                   estadísticas de todos los nodos   │
│                   en el camino                      │
│                                                     │
│   Repetir miles de veces, elegir la acción con     │
│   más visitas                                       │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Estructura de datos del nodo

### Datos principales

Cada nodo MCTS necesita almacenar:

```python
class MCTSNode:
    def __init__(self, state, parent=None, prior=0.0):
        # Información básica
        self.state = state              # Estado del tablero
        self.parent = parent            # Nodo padre
        self.children = {}              # Diccionario de hijos {acción: nodo}
        self.action = None              # Acción para llegar a este nodo

        # Información estadística
        self.visit_count = 0            # N(s): Número de visitas
        self.value_sum = 0.0            # W(s): Suma de valores
        self.prior = prior              # P(s,a): Probabilidad a priori

        # Para búsqueda paralela
        self.virtual_loss = 0           # Pérdida virtual
        self.is_expanded = False        # Si ya está expandido

    @property
    def value(self):
        """Q(s) = W(s) / N(s)"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
```

### Optimización de memoria

KataGo usa varias técnicas para reducir el uso de memoria:

```python
# Usar arrays numpy en lugar de dict de Python
class OptimizedNode:
    __slots__ = ['visit_count', 'value_sum', 'prior', 'children_indices']

    def __init__(self):
        self.visit_count = np.int32(0)
        self.value_sum = np.float32(0.0)
        self.prior = np.float32(0.0)
        self.children_indices = None  # Asignación diferida
```

---

## Selection: Selección PUCT

### Fórmula PUCT

```
Puntuación de selección = Q(s,a) + U(s,a)

Donde:
Q(s,a) = W(s,a) / N(s,a)              # Valor promedio
U(s,a) = c_puct × P(s,a) × √(N(s)) / (1 + N(s,a))  # Término de exploración
```

### Explicación de parámetros

| Símbolo | Significado | Valor típico |
|---------|-------------|--------------|
| Q(s,a) | Valor promedio de la acción a | [-1, +1] |
| P(s,a) | Probabilidad a priori de la red neuronal | [0, 1] |
| N(s) | Número de visitas del nodo padre | Entero |
| N(s,a) | Número de visitas de la acción a | Entero |
| c_puct | Constante de exploración | 1.0 ~ 2.5 |

### Implementación

```python
def select_child(self, c_puct=1.5):
    """Seleccionar el nodo hijo con mayor puntuación PUCT"""
    best_score = -float('inf')
    best_action = None
    best_child = None

    # Raíz cuadrada del número de visitas del padre
    sqrt_parent_visits = math.sqrt(self.visit_count)

    for action, child in self.children.items():
        # Valor Q (valor promedio)
        if child.visit_count > 0:
            q_value = child.value_sum / child.visit_count
        else:
            q_value = 0.0

        # Valor U (término de exploración)
        u_value = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)

        # Puntuación total
        score = q_value + u_value

        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    return best_action, best_child
```

### Balance entre exploración y explotación

```
Etapa inicial: N(s,a) pequeño
├── U(s,a) grande → Dominado por exploración
└── Acciones con alta probabilidad a priori se exploran primero

Etapa posterior: N(s,a) grande
├── U(s,a) pequeño → Dominado por explotación
└── Q(s,a) domina, se eligen acciones conocidas como buenas
```

---

## Expansion: Expansión de nodos

### Condición de expansión

Al llegar a un nodo hoja, expandir usando la red neuronal:

```python
def expand(self, policy_probs, legal_moves):
    """Expandir el nodo, crear nodos hijos para todas las acciones legales"""
    for action in legal_moves:
        if action not in self.children:
            prior = policy_probs[action]  # Probabilidad predicha por la red
            child_state = self.state.play(action)
            self.children[action] = MCTSNode(
                state=child_state,
                parent=self,
                prior=prior
            )

    self.is_expanded = True
```

### Filtrado de movimientos legales

```python
def get_legal_moves(state):
    """Obtener todos los movimientos legales"""
    legal = []
    for i in range(361):
        x, y = i // 19, i % 19
        if state.is_legal(x, y):
            legal.append(i)

    # Añadir pass
    legal.append(361)

    return legal
```

---

## Evaluation: Evaluación con red neuronal

### Evaluación única

```python
def evaluate(self, state):
    """Usar la red neuronal para evaluar la posición"""
    # Codificar características de entrada
    features = encode_state(state)  # (22, 19, 19)
    features = torch.tensor(features).unsqueeze(0)  # (1, 22, 19, 19)

    # Inferencia de la red neuronal
    with torch.no_grad():
        output = self.network(features)

    policy = output['policy'][0].numpy()  # (362,)
    value = output['value'][0].item()     # escalar

    return policy, value
```

### Evaluación por lotes (optimización clave)

La GPU es más eficiente con inferencia por lotes:

```python
class BatchedEvaluator:
    def __init__(self, network, batch_size=8):
        self.network = network
        self.batch_size = batch_size
        self.pending = []  # Lista de (state, callback) pendientes

    def request_evaluation(self, state, callback):
        """Solicitar evaluación, ejecutar automáticamente cuando el lote esté lleno"""
        self.pending.append((state, callback))

        if len(self.pending) >= self.batch_size:
            self.flush()

    def flush(self):
        """Ejecutar evaluación por lotes"""
        if not self.pending:
            return

        # Preparar entrada por lotes
        states = [s for s, _ in self.pending]
        features = torch.stack([encode_state(s) for s in states])

        # Inferencia por lotes
        with torch.no_grad():
            outputs = self.network(features)

        # Callback de resultados
        for i, (_, callback) in enumerate(self.pending):
            policy = outputs['policy'][i].numpy()
            value = outputs['value'][i].item()
            callback(policy, value)

        self.pending.clear()
```

---

## Backpropagation: Actualización de retropropagación

### Retropropagación básica

```python
def backpropagate(self, value):
    """Retropropagar desde el nodo hoja hasta la raíz, actualizar estadísticas"""
    node = self

    while node is not None:
        node.visit_count += 1
        node.value_sum += value

        # Perspectiva alterna: el valor del oponente es opuesto
        value = -value

        node = node.parent
```

### Importancia de la perspectiva alterna

```
Perspectiva de Negro: value = +0.6 (favorable para Negro)

Camino de retropropagación:
Nodo hoja (turno de Negro): value_sum += +0.6
    ↑
Nodo padre (turno de Blanco): value_sum += -0.6  ← Desfavorable para Blanco
    ↑
Nodo abuelo (turno de Negro): value_sum += +0.6
    ↑
...
```

---

## Paralelización: Pérdida virtual

### Problema

Cuando múltiples hilos buscan simultáneamente, pueden elegir el mismo nodo:

```
Thread 1: Selecciona nodo A (Q=0.6, N=100)
Thread 2: Selecciona nodo A (Q=0.6, N=100) ← ¡Repetido!
Thread 3: Selecciona nodo A (Q=0.6, N=100) ← ¡Repetido!
```

### Solución: Pérdida virtual

Al seleccionar un nodo, agregar primero una "pérdida virtual" para que otros hilos no quieran seleccionarlo:

```python
VIRTUAL_LOSS = 3  # Valor de pérdida virtual

def select_with_virtual_loss(self):
    """Selección con pérdida virtual"""
    action, child = self.select_child()

    # Agregar pérdida virtual
    child.visit_count += VIRTUAL_LOSS
    child.value_sum -= VIRTUAL_LOSS  # Simular pérdida

    return action, child

def backpropagate_with_virtual_loss(self, value):
    """Retropropagar eliminando pérdida virtual"""
    node = self

    while node is not None:
        # Eliminar pérdida virtual
        node.visit_count -= VIRTUAL_LOSS
        node.value_sum += VIRTUAL_LOSS

        # Actualización normal
        node.visit_count += 1
        node.value_sum += value

        value = -value
        node = node.parent
```

### Efecto

```
Thread 1: Selecciona nodo A, agrega pérdida virtual
         El valor Q de A baja temporalmente

Thread 2: Selecciona nodo B (porque A parece peor)

Thread 3: Selecciona nodo C

→ Diferentes hilos exploran diferentes ramas, mejorando la eficiencia
```

---

## Implementación completa de búsqueda

```python
class MCTS:
    def __init__(self, network, c_puct=1.5, num_simulations=800):
        self.network = network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.evaluator = BatchedEvaluator(network)

    def search(self, root_state):
        """Ejecutar búsqueda MCTS"""
        root = MCTSNode(root_state)

        # Expandir nodo raíz
        policy, value = self.evaluate(root_state)
        legal_moves = get_legal_moves(root_state)
        root.expand(policy, legal_moves)

        # Ejecutar simulaciones
        for _ in range(self.num_simulations):
            node = root
            path = [node]

            # Selection: Bajar por el árbol
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

        # Elegir la acción con más visitas
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

## Técnicas avanzadas

### Ruido de Dirichlet

Agregar ruido en el nodo raíz durante el entrenamiento para aumentar la exploración:

```python
def add_dirichlet_noise(root, alpha=0.03, epsilon=0.25):
    """Agregar ruido de Dirichlet en el nodo raíz"""
    noise = np.random.dirichlet([alpha] * len(root.children))

    for i, child in enumerate(root.children.values()):
        child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]
```

### Parámetro de temperatura

Controlar la aleatoriedad en la selección de acciones:

```python
def select_action_with_temperature(root, temperature=1.0):
    """Seleccionar acción según visitas y temperatura"""
    visits = np.array([c.visit_count for c in root.children.values()])
    actions = list(root.children.keys())

    if temperature == 0:
        # Selección voraz
        return actions[np.argmax(visits)]
    else:
        # Seleccionar según distribución de probabilidad basada en visitas
        probs = visits ** (1 / temperature)
        probs = probs / probs.sum()
        return np.random.choice(actions, p=probs)
```

### Reutilización del árbol

Un nuevo movimiento puede reutilizar el árbol de búsqueda anterior:

```python
def reuse_tree(root, action):
    """Reutilizar subárbol"""
    if action in root.children:
        new_root = root.children[action]
        new_root.parent = None
        return new_root
    else:
        return None  # Necesita crear un nuevo árbol
```

---

## Resumen de optimización de rendimiento

| Técnica | Efecto |
|---------|--------|
| **Evaluación por lotes** | Utilización de GPU de 10% → 80%+ |
| **Pérdida virtual** | Eficiencia multi-hilo mejora 3-5x |
| **Reutilización del árbol** | Reduce arranque en frío, ahorra 30%+ de cálculo |
| **Pool de memoria** | Reduce overhead de asignación de memoria |

---

## Lectura adicional

- [Arquitectura de redes neuronales en detalle](../neural-network) — Fuente de la función de evaluación
- [Backend GPU y optimización](../gpu-optimization) — Optimización de hardware para inferencia por lotes
- [Guía de artículos clave](../papers) — Base teórica de la fórmula PUCT
