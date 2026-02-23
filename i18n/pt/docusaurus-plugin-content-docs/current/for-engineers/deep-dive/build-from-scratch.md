---
sidebar_position: 12
title: Construindo uma IA de Go do Zero
description: Tutorial passo a passo para implementar uma IA de Go simplificada no estilo AlphaGo Zero
---

# Construindo uma IA de Go do Zero

Este artigo guia voce passo a passo na implementacao de uma IA de Go simplificada no estilo AlphaGo Zero, abrangendo logica do jogo, rede neural, MCTS e fluxo de treinamento.

:::info Objetivo de Aprendizado
Apos completar este tutorial, voce tera uma IA de Go capaz de:
- Jogar contra si mesma em tabuleiro 9x9
- Melhorar continuamente atraves de aprendizado por reforco
- Atingir nivel de jogo amador iniciante
:::

---

## Arquitetura do Projeto

```
mini-alphago/
├── game/
│   ├── __init__.py
│   ├── board.py          # Logica do tabuleiro
│   ├── rules.py          # Implementacao de regras
│   └── state.py          # Estado do jogo
├── model/
│   ├── __init__.py
│   ├── network.py        # Rede neural
│   └── features.py       # Codificacao de recursos
├── mcts/
│   ├── __init__.py
│   ├── node.py           # No MCTS
│   └── search.py         # Busca MCTS
├── training/
│   ├── __init__.py
│   ├── self_play.py      # Auto-jogo
│   └── trainer.py        # Treinador
├── main.py               # Programa principal
└── requirements.txt
```

---

## Passo 1: Tabuleiro e Regras

### Implementacao do Tabuleiro

```python
# game/board.py
import numpy as np

class Board:
    """Tabuleiro de Go"""

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
        """Copia o tabuleiro"""
        new_board = Board(self.size)
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.ko_point = self.ko_point
        new_board.history = self.history.copy()
        return new_board

    def get_opponent(self, player):
        """Obtem o oponente"""
        return self.WHITE if player == self.BLACK else self.BLACK

    def is_on_board(self, x, y):
        """Verifica se esta no tabuleiro"""
        return 0 <= x < self.size and 0 <= y < self.size

    def get_neighbors(self, x, y):
        """Obtem pontos adjacentes"""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if self.is_on_board(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def get_group(self, x, y):
        """Obtem grupo (pedras conectadas da mesma cor)"""
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
        """Conta liberdades"""
        _, liberties = self.get_group(x, y)
        return len(liberties)

    def remove_group(self, group):
        """Remove grupo"""
        for x, y in group:
            self.board[x, y] = self.EMPTY

    def is_legal(self, x, y, player=None):
        """Verifica se e jogada legal"""
        if player is None:
            player = self.current_player

        # Verifica se e ponto vazio
        if self.board[x, y] != self.EMPTY:
            return False

        # Verifica se e Ko
        if self.ko_point == (x, y):
            return False

        # Simula jogada
        test_board = self.copy()
        test_board.board[x, y] = player

        # Primeiro verifica se pode capturar
        opponent = self.get_opponent(player)
        captured = []
        for nx, ny in self.get_neighbors(x, y):
            if test_board.board[nx, ny] == opponent:
                group, liberties = test_board.get_group(nx, ny)
                if len(liberties) == 0:
                    captured.extend(group)

        if captured:
            return True

        # Verifica suicidio
        _, liberties = test_board.get_group(x, y)
        if len(liberties) == 0:
            return False

        return True

    def play(self, x, y):
        """Joga uma pedra"""
        if not self.is_legal(x, y):
            return False

        player = self.current_player
        opponent = self.get_opponent(player)

        # Coloca pedra
        self.board[x, y] = player

        # Captura
        captured = []
        for nx, ny in self.get_neighbors(x, y):
            if self.board[nx, ny] == opponent:
                group, liberties = self.get_group(nx, ny)
                if len(liberties) == 0:
                    captured.extend(group)
                    self.remove_group(group)

        # Define Ko
        if len(captured) == 1:
            cx, cy = list(captured)[0]
            _, my_liberties = self.get_group(x, y)
            if len(my_liberties) == 1:
                self.ko_point = (cx, cy)
            else:
                self.ko_point = None
        else:
            self.ko_point = None

        # Registra historico
        self.history.append((x, y, player))

        # Troca jogador
        self.current_player = opponent

        return True

    def pass_move(self):
        """Pass (jogada virtual)"""
        self.history.append((-1, -1, self.current_player))
        self.current_player = self.get_opponent(self.current_player)
        self.ko_point = None

    def is_game_over(self):
        """Verifica se o jogo acabou"""
        if len(self.history) < 2:
            return False
        # Ambos passaram consecutivamente
        return (self.history[-1][0] == -1 and
                self.history[-2][0] == -1)

    def get_legal_moves(self):
        """Obtem todas as jogadas legais"""
        moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.is_legal(x, y):
                    moves.append((x, y))
        moves.append((-1, -1))  # pass
        return moves

    def score(self):
        """Calcula pontuacao (metodo de contagem de area simplificado)"""
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

## Passo 2: Codificacao de Recursos

### Recursos de Entrada

```python
# model/features.py
import numpy as np

def encode_board(board):
    """
    Codifica o tabuleiro como entrada para a rede neural

    Planos de recursos:
    0: Pedras proprias
    1: Pedras do oponente
    2: Pontos vazios
    3: Posicao da ultima jogada
    4: Posicao da penultima jogada
    5: Posicoes de jogadas legais
    6: Preto a jogar (todo 1 ou todo 0)
    """
    size = board.size
    features = np.zeros((7, size, size), dtype=np.float32)

    current = board.current_player
    opponent = board.get_opponent(current)

    # Posicoes basicas das pedras
    features[0] = (board.board == current).astype(np.float32)
    features[1] = (board.board == opponent).astype(np.float32)
    features[2] = (board.board == board.EMPTY).astype(np.float32)

    # Jogadas recentes
    if len(board.history) >= 1:
        x, y, _ = board.history[-1]
        if x >= 0:
            features[3, x, y] = 1.0

    if len(board.history) >= 2:
        x, y, _ = board.history[-2]
        if x >= 0:
            features[4, x, y] = 1.0

    # Jogadas legais
    for x in range(size):
        for y in range(size):
            if board.is_legal(x, y):
                features[5, x, y] = 1.0

    # De quem e a vez
    if current == board.BLACK:
        features[6] = np.ones((size, size), dtype=np.float32)

    return features
```

---

## Passo 3: Rede Neural

### Arquitetura de Rede com Duas Cabecas

```python
# model/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Bloco residual"""

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
    """Rede de duas cabecas Policy-Value"""

    def __init__(self, board_size=9, input_channels=7, num_filters=64, num_blocks=4):
        super().__init__()
        self.board_size = board_size

        # Convolucao inicial
        self.conv_input = nn.Conv2d(input_channels, num_filters, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)

        # Blocos residuais
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
        # Trunk compartilhado
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
    """Cria a rede"""
    return PolicyValueNetwork(
        board_size=board_size,
        input_channels=7,
        num_filters=64,
        num_blocks=4
    )
```

---

## Passo 4: Implementacao do MCTS

### Classe de No

```python
# mcts/node.py
import numpy as np

class MCTSNode:
    """No MCTS"""

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
        """Expande o no"""
        for move in legal_moves:
            if move not in self.children:
                idx = move[0] * 9 + move[1] if move[0] >= 0 else 81
                self.children[move] = MCTSNode(prior=np.exp(policy[idx]))

    def select_child(self, c_puct=1.5):
        """Seleciona no filho usando PUCT"""
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

### Implementacao da Busca

```python
# mcts/search.py
import numpy as np
import torch
from .node import MCTSNode

class MCTS:
    """Busca em Arvore de Monte Carlo"""

    def __init__(self, network, board_size=9, num_simulations=100, c_puct=1.5):
        self.network = network
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, board, add_noise=False):
        """Executa busca MCTS"""
        root = MCTSNode()

        # Avalia no raiz
        policy, value = self.evaluate(board)
        legal_moves = board.get_legal_moves()
        root.expand(policy, legal_moves)

        # Adiciona ruido de Dirichlet (durante treinamento)
        if add_noise:
            self.add_dirichlet_noise(root)

        # Executa simulacoes
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

            # Calcula valor da perspectiva do ponto de partida da busca
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
        """Avalia usando rede neural"""
        from model.features import encode_board

        features = encode_board(board)
        features = torch.tensor(features).unsqueeze(0)

        self.network.eval()
        with torch.no_grad():
            policy, value = self.network(features)

        return policy[0].numpy(), value[0].item()

    def add_dirichlet_noise(self, root, alpha=0.3, epsilon=0.25):
        """Adiciona ruido de exploracao"""
        noise = np.random.dirichlet([alpha] * len(root.children))
        for i, child in enumerate(root.children.values()):
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]

    def get_policy(self, root, temperature=1.0):
        """Obtem politica do resultado da busca"""
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
        """Seleciona jogada"""
        policy = self.get_policy(root, temperature)
        idx = np.random.choice(len(policy), p=policy)

        if idx == self.board_size ** 2:
            return (-1, -1)
        else:
            return (idx // self.board_size, idx % self.board_size)
```

---

## Passo 5: Auto-jogo

```python
# training/self_play.py
import numpy as np
from game.board import Board
from model.features import encode_board

def self_play_game(mcts, temperature=1.0, temp_threshold=30):
    """Executa uma partida de auto-jogo"""
    board = Board(size=9)
    game_history = []

    move_count = 0
    while not board.is_game_over() and move_count < 200:
        # Busca MCTS
        root = mcts.search(board, add_noise=True)

        # Obtem politica
        temp = temperature if move_count < temp_threshold else 0.0
        policy = mcts.get_policy(root, temp)

        # Registra dados de treinamento
        features = encode_board(board)
        game_history.append({
            'features': features,
            'policy': policy,
            'player': board.current_player
        })

        # Seleciona e executa jogada
        move = mcts.select_move(root, temp)
        if move[0] >= 0:
            board.play(move[0], move[1])
        else:
            board.pass_move()

        move_count += 1

    # Calcula resultado
    score = board.score()
    winner = Board.BLACK if score > 0 else (Board.WHITE if score < 0 else 0)

    # Marca valores
    for data in game_history:
        if winner == 0:
            data['value'] = 0.0
        elif data['player'] == winner:
            data['value'] = 1.0
        else:
            data['value'] = -1.0

    return game_history


def generate_training_data(mcts, num_games=100):
    """Gera dados de treinamento"""
    all_data = []

    for i in range(num_games):
        print(f"Partida de auto-jogo {i+1}/{num_games}")
        game_data = self_play_game(mcts)
        all_data.extend(game_data)

    return all_data
```

---

## Passo 6: Treinador

```python
# training/trainer.py
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class Trainer:
    """Treinador"""

    def __init__(self, network, learning_rate=0.001):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    def train_step(self, batch):
        """Passo de treinamento"""
        features, target_policy, target_value = batch

        self.network.train()
        self.optimizer.zero_grad()

        # Forward pass
        pred_policy, pred_value = self.network(features)

        # Calcula perda
        policy_loss = F.kl_div(pred_policy, target_policy, reduction='batchmean')
        value_loss = F.mse_loss(pred_value.squeeze(), target_value)
        total_loss = policy_loss + value_loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }

    def train_epoch(self, data, batch_size=32):
        """Treina uma epoca"""
        # Prepara dados
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
        """Salva modelo"""
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        """Carrega modelo"""
        self.network.load_state_dict(torch.load(path))
```

---

## Passo 7: Programa Principal

```python
# main.py
from model.network import create_network
from mcts.search import MCTS
from training.self_play import generate_training_data
from training.trainer import Trainer

def main():
    # Cria rede
    network = create_network(board_size=9)
    mcts = MCTS(network, board_size=9, num_simulations=100)
    trainer = Trainer(network)

    # Loop de treinamento
    num_iterations = 100
    games_per_iteration = 50
    epochs_per_iteration = 10

    for iteration in range(num_iterations):
        print(f"\n=== Iteracao {iteration + 1}/{num_iterations} ===")

        # Auto-jogo
        print("Gerando partidas de auto-jogo...")
        training_data = generate_training_data(mcts, num_games=games_per_iteration)

        # Treinamento
        print("Treinando...")
        for epoch in range(epochs_per_iteration):
            loss = trainer.train_epoch(training_data)
            print(f"  Epoca {epoch + 1}: loss = {loss:.4f}")

        # Salva
        trainer.save(f"model_iter_{iteration + 1}.pt")

    print("\nTreinamento completo!")


if __name__ == "__main__":
    main()
```

---

## Execucao e Teste

### Instalar Dependencias

```bash
pip install torch numpy
```

### Executar Treinamento

```bash
python main.py
```

### Saida Esperada

```
=== Iteracao 1/100 ===
Gerando partidas de auto-jogo...
Partida de auto-jogo 1/50
Partida de auto-jogo 2/50
...
Treinando...
  Epoca 1: loss = 2.3456
  Epoca 2: loss = 1.8765
  ...
```

---

## Sugestoes de Melhoria

### Melhorias de Curto Prazo

| Item de Melhoria | Descricao |
|------------------|-----------|
| Aumentar blocos residuais | 4 → 8 → 16 blocos |
| Aumentar numero de canais | 64 → 128 → 256 |
| Aumentar numero de simulacoes | 100 → 400 → 800 |
| Conjunto de treinamento maior | 50 → 200 → 1000 jogos/iteracao |

### Melhorias de Longo Prazo

- Suportar tabuleiro 19x19
- Adicionar objetivos de treinamento auxiliares (previsao de territorio)
- Implementar auto-jogo paralelo
- Adicionar aceleracao por GPU

---

## Leitura Adicional

- [Arquitetura de Rede Neural Detalhada](../neural-network) — Design de rede mais aprofundado
- [Detalhes de Implementacao do MCTS](../mcts-implementation) — Tecnicas de busca avancadas
- [Analise do Mecanismo de Treinamento do KataGo](../training) — Sistema de treinamento de producao
- [Guia de Artigos Importantes](../papers) — Fundamentos teoricos
