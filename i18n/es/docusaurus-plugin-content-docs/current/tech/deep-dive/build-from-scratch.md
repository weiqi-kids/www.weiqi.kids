---
sidebar_position: 12
title: Construir una IA de Go desde cero
description: Tutorial paso a paso para implementar una IA de Go estilo AlphaGo Zero simplificada
---

# Construir una IA de Go desde cero

Este articulo te guia paso a paso para implementar una IA de Go simplificada al estilo AlphaGo Zero, cubriendo logica del juego, red neuronal, MCTS y proceso de entrenamiento.

:::info Objetivos de aprendizaje
Al completar este tutorial, tendras una IA de Go capaz de:
- Auto-jugar en tablero 9x9
- Mejorar continuamente a traves de aprendizaje por refuerzo
- Alcanzar nivel amateur principiante
:::

---

## Arquitectura del proyecto

```
mini-alphago/
├── game/
│   ├── __init__.py
│   ├── board.py          # Logica del tablero
│   ├── rules.py          # Implementacion de reglas
│   └── state.py          # Estado del juego
├── model/
│   ├── __init__.py
│   ├── network.py        # Red neuronal
│   └── features.py       # Codificacion de caracteristicas
├── mcts/
│   ├── __init__.py
│   ├── node.py           # Nodo MCTS
│   └── search.py         # Busqueda MCTS
├── training/
│   ├── __init__.py
│   ├── self_play.py      # Auto-juego
│   └── trainer.py        # Entrenador
├── main.py               # Programa principal
└── requirements.txt
```

---

## Paso 1: Tablero y reglas

### Implementacion del tablero

```python
# game/board.py
import numpy as np

class Board:
    """Tablero de Go"""

    EMPTY = 0
    BLACK = 1
    WHITE = 2

    def __init__(self, size=9):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        self.current_player = self.BLACK
        self.ko_point = None
        self.history = []

    def copy(self):
        """Copiar tablero"""
        new_board = Board(self.size)
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.ko_point = self.ko_point
        new_board.history = self.history.copy()
        return new_board

    def get_opponent(self, player):
        """Obtener oponente"""
        return self.WHITE if player == self.BLACK else self.BLACK

    def is_on_board(self, x, y):
        """Verificar si esta en el tablero"""
        return 0 <= x < self.size and 0 <= y < self.size

    def get_neighbors(self, x, y):
        """Obtener puntos adyacentes"""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if self.is_on_board(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def get_group(self, x, y):
        """Obtener cadena (piedras conectadas del mismo color)"""
        color = self.board[x, y]
        if color == self.EMPTY:
            return set(), set()

        group = set()
        liberties = set()
        stack = [(x, y)]

        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in group:
                continue
            group.add((cx, cy))

            for nx, ny in self.get_neighbors(cx, cy):
                if self.board[nx, ny] == self.EMPTY:
                    liberties.add((nx, ny))
                elif self.board[nx, ny] == color and (nx, ny) not in group:
                    stack.append((nx, ny))

        return group, liberties

    def count_liberties(self, x, y):
        """Contar libertades"""
        _, liberties = self.get_group(x, y)
        return len(liberties)

    def remove_group(self, group):
        """Eliminar cadena"""
        for x, y in group:
            self.board[x, y] = self.EMPTY

    def is_legal(self, x, y, player=None):
        """Verificar si es un movimiento legal"""
        if player is None:
            player = self.current_player

        # Verificar si es punto vacio
        if self.board[x, y] != self.EMPTY:
            return False

        # Verificar si es Ko
        if self.ko_point == (x, y):
            return False

        # Simular movimiento
        test_board = self.copy()
        test_board.board[x, y] = player

        # Verificar primero si puede capturar
        opponent = self.get_opponent(player)
        captured = []
        for nx, ny in self.get_neighbors(x, y):
            if test_board.board[nx, ny] == opponent:
                group, liberties = test_board.get_group(nx, ny)
                if len(liberties) == 0:
                    captured.extend(group)

        if captured:
            return True

        # Verificar suicidio
        _, liberties = test_board.get_group(x, y)
        if len(liberties) == 0:
            return False

        return True

    def play(self, x, y):
        """Jugar piedra"""
        if not self.is_legal(x, y):
            return False

        player = self.current_player
        opponent = self.get_opponent(player)

        # Colocar piedra
        self.board[x, y] = player

        # Capturar
        captured = []
        for nx, ny in self.get_neighbors(x, y):
            if self.board[nx, ny] == opponent:
                group, liberties = self.get_group(nx, ny)
                if len(liberties) == 0:
                    captured.extend(group)
                    self.remove_group(group)

        # Establecer Ko
        if len(captured) == 1:
            cx, cy = list(captured)[0]
            _, my_liberties = self.get_group(x, y)
            if len(my_liberties) == 1:
                self.ko_point = (cx, cy)
            else:
                self.ko_point = None
        else:
            self.ko_point = None

        # Registrar historial
        self.history.append((x, y, player))

        # Cambiar jugador
        self.current_player = opponent

        return True

    def pass_move(self):
        """Pasar turno"""
        self.history.append((-1, -1, self.current_player))
        self.current_player = self.get_opponent(self.current_player)
        self.ko_point = None

    def is_game_over(self):
        """Verificar si el juego termino"""
        if len(self.history) < 2:
            return False
        # Ambos jugadores pasan consecutivamente
        return (self.history[-1][0] == -1 and
                self.history[-2][0] == -1)

    def get_legal_moves(self):
        """Obtener todos los movimientos legales"""
        moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.is_legal(x, y):
                    moves.append((x, y))
        moves.append((-1, -1))  # pass
        return moves

    def score(self):
        """Calcular puntuacion (conteo de area simplificado)"""
        black_score = np.sum(self.board == self.BLACK)
        white_score = np.sum(self.board == self.WHITE)

        # Calculo de territorio simplificado
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x, y] == self.EMPTY:
                    neighbors = self.get_neighbors(x, y)
                    colors = set(self.board[nx, ny] for nx, ny in neighbors)
                    colors.discard(self.EMPTY)
                    if len(colors) == 1:
                        if self.BLACK in colors:
                            black_score += 1
                        else:
                            white_score += 1

        komi = 5.5 if self.size == 9 else 7.5
        return black_score - white_score - komi
```

---

## Paso 2: Codificacion de caracteristicas

### Caracteristicas de entrada

```python
# model/features.py
import numpy as np

def encode_board(board):
    """
    Codificar tablero como entrada de red neuronal

    Planos de caracteristicas:
    0: Piedras propias
    1: Piedras del oponente
    2: Puntos vacios
    3: Posicion del ultimo movimiento
    4: Posicion del penultimo movimiento
    5: Posiciones de movimientos legales
    6: Turno de Negro (todos 1 o todos 0)
    """
    size = board.size
    features = np.zeros((7, size, size), dtype=np.float32)

    current = board.current_player
    opponent = board.get_opponent(current)

    # Posiciones basicas de piedras
    features[0] = (board.board == current).astype(np.float32)
    features[1] = (board.board == opponent).astype(np.float32)
    features[2] = (board.board == board.EMPTY).astype(np.float32)

    # Movimientos recientes
    if len(board.history) >= 1:
        x, y, _ = board.history[-1]
        if x >= 0:
            features[3, x, y] = 1.0

    if len(board.history) >= 2:
        x, y, _ = board.history[-2]
        if x >= 0:
            features[4, x, y] = 1.0

    # Movimientos legales
    for x in range(size):
        for y in range(size):
            if board.is_legal(x, y):
                features[5, x, y] = 1.0

    # Turno
    if current == board.BLACK:
        features[6] = np.ones((size, size), dtype=np.float32)

    return features
```

---

## Paso 3: Red neuronal

### Arquitectura de red de doble cabeza

```python
# model/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Bloque residual"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class PolicyValueNetwork(nn.Module):
    """Red de doble cabeza politica-valor"""

    def __init__(self, board_size=9, input_channels=7, num_filters=64, num_blocks=4):
        super().__init__()
        self.board_size = board_size

        # Convolucion inicial
        self.conv_input = nn.Conv2d(input_channels, num_filters, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)

        # Bloques residuales
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_blocks)
        ])

        # Policy Head
        self.policy_conv = nn.Conv2d(num_filters, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size + 1)

        # Value Head
        self.value_conv = nn.Conv2d(num_filters, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Tronco compartido
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.residual_blocks:
            x = block(x)

        # Policy Head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)

        # Value Head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value


def create_network(board_size=9):
    """Crear red"""
    return PolicyValueNetwork(
        board_size=board_size,
        input_channels=7,
        num_filters=64,
        num_blocks=4
    )
```

---

## Paso 4: Implementacion de MCTS

### Clase de nodo

```python
# mcts/node.py
import numpy as np

class MCTSNode:
    """Nodo MCTS"""

    def __init__(self, prior=0.0):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children = {}

    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, policy, legal_moves):
        """Expandir nodo"""
        for move in legal_moves:
            if move not in self.children:
                idx = move[0] * 9 + move[1] if move[0] >= 0 else 81
                self.children[move] = MCTSNode(prior=np.exp(policy[idx]))

    def select_child(self, c_puct=1.5):
        """Seleccionar nodo hijo usando PUCT"""
        best_score = -float('inf')
        best_move = None
        best_child = None

        sqrt_total = np.sqrt(max(1, self.visit_count))

        for move, child in self.children.items():
            if child.visit_count > 0:
                q_value = child.value
            else:
                q_value = 0.0

            u_value = c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child
```

### Implementacion de busqueda

```python
# mcts/search.py
import numpy as np
import torch
from .node import MCTSNode

class MCTS:
    """Monte Carlo Tree Search"""

    def __init__(self, network, board_size=9, num_simulations=100, c_puct=1.5):
        self.network = network
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, board, add_noise=False):
        """Ejecutar busqueda MCTS"""
        root = MCTSNode()

        # Evaluar nodo raiz
        policy, value = self.evaluate(board)
        legal_moves = board.get_legal_moves()
        root.expand(policy, legal_moves)

        # Agregar ruido de Dirichlet (durante entrenamiento)
        if add_noise:
            self.add_dirichlet_noise(root)

        # Ejecutar simulaciones
        for _ in range(self.num_simulations):
            node = root
            scratch_board = board.copy()
            path = [node]

            # Selection
            while node.children and scratch_board.get_legal_moves():
                move, node = node.select_child(self.c_puct)
                if move[0] >= 0:
                    scratch_board.play(move[0], move[1])
                else:
                    scratch_board.pass_move()
                path.append(node)

                if scratch_board.is_game_over():
                    break

            # Expansion + Evaluation
            if not scratch_board.is_game_over():
                policy, value = self.evaluate(scratch_board)
                legal_moves = scratch_board.get_legal_moves()
                if legal_moves:
                    node.expand(policy, legal_moves)

            # Calcular valor desde la perspectiva del punto de inicio de busqueda
            if scratch_board.is_game_over():
                score = scratch_board.score()
                value = 1.0 if score > 0 else (-1.0 if score < 0 else 0.0)
                if board.current_player != scratch_board.BLACK:
                    value = -value

            # Backpropagation
            for node in reversed(path):
                node.visit_count += 1
                node.value_sum += value
                value = -value

        return root

    def evaluate(self, board):
        """Evaluar con red neuronal"""
        from model.features import encode_board

        features = encode_board(board)
        features = torch.tensor(features).unsqueeze(0)

        self.network.eval()
        with torch.no_grad():
            policy, value = self.network(features)

        return policy[0].numpy(), value[0].item()

    def add_dirichlet_noise(self, root, alpha=0.3, epsilon=0.25):
        """Agregar ruido de exploracion"""
        noise = np.random.dirichlet([alpha] * len(root.children))
        for i, child in enumerate(root.children.values()):
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]

    def get_policy(self, root, temperature=1.0):
        """Obtener politica del resultado de busqueda"""
        visits = np.zeros(self.board_size ** 2 + 1)

        for move, child in root.children.items():
            idx = move[0] * self.board_size + move[1] if move[0] >= 0 else self.board_size ** 2
            visits[idx] = child.visit_count

        if temperature == 0:
            policy = np.zeros_like(visits)
            policy[np.argmax(visits)] = 1.0
        else:
            visits = visits ** (1 / temperature)
            policy = visits / visits.sum()

        return policy

    def select_move(self, root, temperature=1.0):
        """Seleccionar movimiento"""
        policy = self.get_policy(root, temperature)
        idx = np.random.choice(len(policy), p=policy)

        if idx == self.board_size ** 2:
            return (-1, -1)
        else:
            return (idx // self.board_size, idx % self.board_size)
```

---

## Paso 5: Auto-juego

```python
# training/self_play.py
import numpy as np
from game.board import Board
from model.features import encode_board

def self_play_game(mcts, temperature=1.0, temp_threshold=30):
    """Ejecutar una partida de auto-juego"""
    board = Board(size=9)
    game_history = []

    move_count = 0
    while not board.is_game_over() and move_count < 200:
        # Busqueda MCTS
        root = mcts.search(board, add_noise=True)

        # Obtener politica
        temp = temperature if move_count < temp_threshold else 0.0
        policy = mcts.get_policy(root, temp)

        # Registrar datos de entrenamiento
        features = encode_board(board)
        game_history.append({
            'features': features,
            'policy': policy,
            'player': board.current_player
        })

        # Seleccionar y ejecutar movimiento
        move = mcts.select_move(root, temp)
        if move[0] >= 0:
            board.play(move[0], move[1])
        else:
            board.pass_move()

        move_count += 1

    # Calcular resultado
    score = board.score()
    winner = Board.BLACK if score > 0 else (Board.WHITE if score < 0 else 0)

    # Marcar valor
    for data in game_history:
        if winner == 0:
            data['value'] = 0.0
        elif data['player'] == winner:
            data['value'] = 1.0
        else:
            data['value'] = -1.0

    return game_history


def generate_training_data(mcts, num_games=100):
    """Generar datos de entrenamiento"""
    all_data = []

    for i in range(num_games):
        print(f"Partida de auto-juego {i+1}/{num_games}")
        game_data = self_play_game(mcts)
        all_data.extend(game_data)

    return all_data
```

---

## Paso 6: Entrenador

```python
# training/trainer.py
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class Trainer:
    """Entrenador"""

    def __init__(self, network, learning_rate=0.001):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    def train_step(self, batch):
        """Paso de entrenamiento"""
        features, target_policy, target_value = batch

        self.network.train()
        self.optimizer.zero_grad()

        # Propagacion hacia adelante
        pred_policy, pred_value = self.network(features)

        # Calcular perdida
        policy_loss = F.kl_div(pred_policy, target_policy, reduction='batchmean')
        value_loss = F.mse_loss(pred_value.squeeze(), target_value)
        total_loss = policy_loss + value_loss

        # Retropropagacion
        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }

    def train_epoch(self, data, batch_size=32):
        """Entrenar una epoca"""
        # Preparar datos
        features = np.array([d['features'] for d in data])
        policies = np.array([d['policy'] for d in data])
        values = np.array([d['value'] for d in data])

        features = torch.tensor(features, dtype=torch.float32)
        policies = torch.tensor(policies, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)

        dataset = TensorDataset(features, policies, values)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_losses = []
        for batch in loader:
            losses = self.train_step(batch)
            total_losses.append(losses['total_loss'])

        return np.mean(total_losses)

    def save(self, path):
        """Guardar modelo"""
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        """Cargar modelo"""
        self.network.load_state_dict(torch.load(path))
```

---

## Paso 7: Programa principal

```python
# main.py
from model.network import create_network
from mcts.search import MCTS
from training.self_play import generate_training_data
from training.trainer import Trainer

def main():
    # Crear red
    network = create_network(board_size=9)
    mcts = MCTS(network, board_size=9, num_simulations=100)
    trainer = Trainer(network)

    # Ciclo de entrenamiento
    num_iterations = 100
    games_per_iteration = 50
    epochs_per_iteration = 10

    for iteration in range(num_iterations):
        print(f"\n=== Iteracion {iteration + 1}/{num_iterations} ===")

        # Auto-juego
        print("Generando partidas de auto-juego...")
        training_data = generate_training_data(mcts, num_games=games_per_iteration)

        # Entrenamiento
        print("Entrenando...")
        for epoch in range(epochs_per_iteration):
            loss = trainer.train_epoch(training_data)
            print(f"  Epoca {epoch + 1}: loss = {loss:.4f}")

        # Guardar
        trainer.save(f"model_iter_{iteration + 1}.pt")

    print("\nEntrenamiento completado!")


if __name__ == "__main__":
    main()
```

---

## Ejecucion y pruebas

### Instalar dependencias

```bash
pip install torch numpy
```

### Ejecutar entrenamiento

```bash
python main.py
```

### Salida esperada

```
=== Iteracion 1/100 ===
Generando partidas de auto-juego...
Partida de auto-juego 1/50
Partida de auto-juego 2/50
...
Entrenando...
  Epoca 1: loss = 2.3456
  Epoca 2: loss = 1.8765
  ...
```

---

## Sugerencias de mejora

### Mejoras a corto plazo

| Elemento de mejora | Descripcion |
|-------------------|-------------|
| Aumentar bloques residuales | 4 → 8 → 16 bloques |
| Aumentar numero de canales | 64 → 128 → 256 |
| Aumentar simulaciones | 100 → 400 → 800 |
| Dataset mas grande | 50 → 200 → 1000 partidas/iteracion |

### Mejoras a largo plazo

- Soportar tablero 19x19
- Agregar objetivos de entrenamiento auxiliares (prediccion de territorio)
- Implementar auto-juego paralelo
- Agregar aceleracion GPU

---

## Lectura adicional

- [Arquitectura de redes neuronales en detalle](../neural-network) — Diseno de red mas profundo
- [Detalles de implementacion de MCTS](../mcts-implementation) — Tecnicas de busqueda avanzadas
- [Analisis del mecanismo de entrenamiento de KataGo](../training) — Sistema de entrenamiento de produccion
- [Guia de articulos clave](../papers) — Fundamentos teoricos
