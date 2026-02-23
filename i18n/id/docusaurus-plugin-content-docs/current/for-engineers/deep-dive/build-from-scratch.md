---
sidebar_position: 12
title: Membangun AI Go dari Nol
description: Tutorial langkah demi langkah untuk mengimplementasikan AI Go bergaya AlphaGo Zero versi sederhana
---

# Membangun AI Go dari Nol

Artikel ini memandu Anda langkah demi langkah untuk mengimplementasikan AI Go bergaya AlphaGo Zero yang disederhanakan, mencakup logika game, neural network, MCTS, dan alur pelatihan.

:::info Tujuan Pembelajaran
Setelah menyelesaikan tutorial ini, Anda akan memiliki AI Go yang dapat:
- Self-play di papan 9×9
- Terus berkembang melalui reinforcement learning
- Mencapai level amatir pemula
:::

---

## Arsitektur Proyek

```
mini-alphago/
├── game/
│   ├── __init__.py
│   ├── board.py          # Logika papan
│   ├── rules.py          # Implementasi aturan
│   └── state.py          # Status game
├── model/
│   ├── __init__.py
│   ├── network.py        # Neural network
│   └── features.py       # Encoding fitur
├── mcts/
│   ├── __init__.py
│   ├── node.py           # Node MCTS
│   └── search.py         # Pencarian MCTS
├── training/
│   ├── __init__.py
│   ├── self_play.py      # Self-play
│   └── trainer.py        # Trainer
├── main.py               # Program utama
└── requirements.txt
```

---

## Langkah 1: Papan dan Aturan

### Implementasi Papan

```python
# game/board.py
import numpy as np

class Board:
    """Papan Go"""

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
        """Salin papan"""
        new_board = Board(self.size)
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.ko_point = self.ko_point
        new_board.history = self.history.copy()
        return new_board

    def get_opponent(self, player):
        """Dapatkan lawan"""
        return self.WHITE if player == self.BLACK else self.BLACK

    def is_on_board(self, x, y):
        """Periksa apakah di papan"""
        return 0 <= x < self.size and 0 <= y < self.size

    def get_neighbors(self, x, y):
        """Dapatkan titik yang berdekatan"""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if self.is_on_board(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def get_group(self, x, y):
        """Dapatkan grup batu (batu dengan warna sama yang terhubung)"""
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
        """Hitung jumlah liberty"""
        _, liberties = self.get_group(x, y)
        return len(liberties)

    def remove_group(self, group):
        """Hapus grup batu"""
        for x, y in group:
            self.board[x, y] = self.EMPTY

    def is_legal(self, x, y, player=None):
        """Periksa apakah langkah legal"""
        if player is None:
            player = self.current_player

        # Periksa apakah titik kosong
        if self.board[x, y] != self.EMPTY:
            return False

        # Periksa apakah Ko
        if self.ko_point == (x, y):
            return False

        # Simulasi penempatan
        test_board = self.copy()
        test_board.board[x, y] = player

        # Periksa dulu apakah bisa menangkap
        opponent = self.get_opponent(player)
        captured = []
        for nx, ny in self.get_neighbors(x, y):
            if test_board.board[nx, ny] == opponent:
                group, liberties = test_board.get_group(nx, ny)
                if len(liberties) == 0:
                    captured.extend(group)

        if captured:
            return True

        # Periksa bunuh diri
        _, liberties = test_board.get_group(x, y)
        if len(liberties) == 0:
            return False

        return True

    def play(self, x, y):
        """Letakkan batu"""
        if not self.is_legal(x, y):
            return False

        player = self.current_player
        opponent = self.get_opponent(player)

        # Letakkan batu
        self.board[x, y] = player

        # Tangkap
        captured = []
        for nx, ny in self.get_neighbors(x, y):
            if self.board[nx, ny] == opponent:
                group, liberties = self.get_group(nx, ny)
                if len(liberties) == 0:
                    captured.extend(group)
                    self.remove_group(group)

        # Set Ko
        if len(captured) == 1:
            cx, cy = list(captured)[0]
            _, my_liberties = self.get_group(x, y)
            if len(my_liberties) == 1:
                self.ko_point = (cx, cy)
            else:
                self.ko_point = None
        else:
            self.ko_point = None

        # Catat history
        self.history.append((x, y, player))

        # Ganti pemain
        self.current_player = opponent

        return True

    def pass_move(self):
        """Pass"""
        self.history.append((-1, -1, self.current_player))
        self.current_player = self.get_opponent(self.current_player)
        self.ko_point = None

    def is_game_over(self):
        """Periksa apakah selesai"""
        if len(self.history) < 2:
            return False
        # Kedua pemain pass berturut-turut
        return (self.history[-1][0] == -1 and
                self.history[-2][0] == -1)

    def get_legal_moves(self):
        """Dapatkan semua langkah legal"""
        moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.is_legal(x, y):
                    moves.append((x, y))
        moves.append((-1, -1))  # pass
        return moves

    def score(self):
        """Hitung skor (metode penghitungan area sederhana)"""
        black_score = np.sum(self.board == self.BLACK)
        white_score = np.sum(self.board == self.WHITE)

        # Penghitungan teritori sederhana
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

## Langkah 2: Encoding Fitur

### Fitur Input

```python
# model/features.py
import numpy as np

def encode_board(board):
    """
    Encode papan sebagai input neural network

    Feature plane:
    0: Batu sendiri
    1: Batu lawan
    2: Titik kosong
    3: Posisi langkah terakhir
    4: Posisi langkah kedua terakhir
    5: Posisi langkah legal
    6: Giliran Hitam (semua 1 atau semua 0)
    """
    size = board.size
    features = np.zeros((7, size, size), dtype=np.float32)

    current = board.current_player
    opponent = board.get_opponent(current)

    # Posisi batu dasar
    features[0] = (board.board == current).astype(np.float32)
    features[1] = (board.board == opponent).astype(np.float32)
    features[2] = (board.board == board.EMPTY).astype(np.float32)

    # Penempatan terbaru
    if len(board.history) >= 1:
        x, y, _ = board.history[-1]
        if x >= 0:
            features[3, x, y] = 1.0

    if len(board.history) >= 2:
        x, y, _ = board.history[-2]
        if x >= 0:
            features[4, x, y] = 1.0

    # Langkah legal
    for x in range(size):
        for y in range(size):
            if board.is_legal(x, y):
                features[5, x, y] = 1.0

    # Giliran siapa
    if current == board.BLACK:
        features[6] = np.ones((size, size), dtype=np.float32)

    return features
```

---

## Langkah 3: Neural Network

### Arsitektur Dual-Head Network

```python
# model/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Blok residual"""

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
    """Network dual-head policy-value"""

    def __init__(self, board_size=9, input_channels=7, num_filters=64, num_blocks=4):
        super().__init__()
        self.board_size = board_size

        # Konvolusi awal
        self.conv_input = nn.Conv2d(input_channels, num_filters, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)

        # Blok residual
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
        # Backbone bersama
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
    """Buat network"""
    return PolicyValueNetwork(
        board_size=board_size,
        input_channels=7,
        num_filters=64,
        num_blocks=4
    )
```

---

## Langkah 4: Implementasi MCTS

### Kelas Node

```python
# mcts/node.py
import numpy as np

class MCTSNode:
    """Node MCTS"""

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
        """Ekspansi node"""
        for move in legal_moves:
            if move not in self.children:
                idx = move[0] * 9 + move[1] if move[0] >= 0 else 81
                self.children[move] = MCTSNode(prior=np.exp(policy[idx]))

    def select_child(self, c_puct=1.5):
        """Pilih child node menggunakan PUCT"""
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

### Implementasi Pencarian

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
        """Jalankan pencarian MCTS"""
        root = MCTSNode()

        # Evaluasi root node
        policy, value = self.evaluate(board)
        legal_moves = board.get_legal_moves()
        root.expand(policy, legal_moves)

        # Tambahkan noise Dirichlet (saat pelatihan)
        if add_noise:
            self.add_dirichlet_noise(root)

        # Jalankan simulasi
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

            # Hitung nilai dari perspektif titik awal pencarian
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
        """Evaluasi menggunakan neural network"""
        from model.features import encode_board

        features = encode_board(board)
        features = torch.tensor(features).unsqueeze(0)

        self.network.eval()
        with torch.no_grad():
            policy, value = self.network(features)

        return policy[0].numpy(), value[0].item()

    def add_dirichlet_noise(self, root, alpha=0.3, epsilon=0.25):
        """Tambahkan noise eksplorasi"""
        noise = np.random.dirichlet([alpha] * len(root.children))
        for i, child in enumerate(root.children.values()):
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]

    def get_policy(self, root, temperature=1.0):
        """Dapatkan kebijakan dari hasil pencarian"""
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
        """Pilih langkah"""
        policy = self.get_policy(root, temperature)
        idx = np.random.choice(len(policy), p=policy)

        if idx == self.board_size ** 2:
            return (-1, -1)
        else:
            return (idx // self.board_size, idx % self.board_size)
```

---

## Langkah 5: Self-play

```python
# training/self_play.py
import numpy as np
from game.board import Board
from model.features import encode_board

def self_play_game(mcts, temperature=1.0, temp_threshold=30):
    """Jalankan satu permainan self-play"""
    board = Board(size=9)
    game_history = []

    move_count = 0
    while not board.is_game_over() and move_count < 200:
        # Pencarian MCTS
        root = mcts.search(board, add_noise=True)

        # Dapatkan kebijakan
        temp = temperature if move_count < temp_threshold else 0.0
        policy = mcts.get_policy(root, temp)

        # Catat data pelatihan
        features = encode_board(board)
        game_history.append({
            'features': features,
            'policy': policy,
            'player': board.current_player
        })

        # Pilih dan jalankan langkah
        move = mcts.select_move(root, temp)
        if move[0] >= 0:
            board.play(move[0], move[1])
        else:
            board.pass_move()

        move_count += 1

    # Hitung menang/kalah
    score = board.score()
    winner = Board.BLACK if score > 0 else (Board.WHITE if score < 0 else 0)

    # Tandai nilai
    for data in game_history:
        if winner == 0:
            data['value'] = 0.0
        elif data['player'] == winner:
            data['value'] = 1.0
        else:
            data['value'] = -1.0

    return game_history


def generate_training_data(mcts, num_games=100):
    """Hasilkan data pelatihan"""
    all_data = []

    for i in range(num_games):
        print(f"Self-play game {i+1}/{num_games}")
        game_data = self_play_game(mcts)
        all_data.extend(game_data)

    return all_data
```

---

## Langkah 6: Trainer

```python
# training/trainer.py
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class Trainer:
    """Trainer"""

    def __init__(self, network, learning_rate=0.001):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    def train_step(self, batch):
        """Langkah pelatihan tunggal"""
        features, target_policy, target_value = batch

        self.network.train()
        self.optimizer.zero_grad()

        # Forward pass
        pred_policy, pred_value = self.network(features)

        # Hitung loss
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
        """Latih satu epoch"""
        # Siapkan data
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
        """Simpan model"""
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        """Muat model"""
        self.network.load_state_dict(torch.load(path))
```

---

## Langkah 7: Program Utama

```python
# main.py
from model.network import create_network
from mcts.search import MCTS
from training.self_play import generate_training_data
from training.trainer import Trainer

def main():
    # Buat network
    network = create_network(board_size=9)
    mcts = MCTS(network, board_size=9, num_simulations=100)
    trainer = Trainer(network)

    # Loop pelatihan
    num_iterations = 100
    games_per_iteration = 50
    epochs_per_iteration = 10

    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")

        # Self-play
        print("Generating self-play games...")
        training_data = generate_training_data(mcts, num_games=games_per_iteration)

        # Pelatihan
        print("Training...")
        for epoch in range(epochs_per_iteration):
            loss = trainer.train_epoch(training_data)
            print(f"  Epoch {epoch + 1}: loss = {loss:.4f}")

        # Simpan
        trainer.save(f"model_iter_{iteration + 1}.pt")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
```

---

## Menjalankan dan Pengujian

### Instalasi Dependensi

```bash
pip install torch numpy
```

### Jalankan Pelatihan

```bash
python main.py
```

### Output yang Diharapkan

```
=== Iteration 1/100 ===
Generating self-play games...
Self-play game 1/50
Self-play game 2/50
...
Training...
  Epoch 1: loss = 2.3456
  Epoch 2: loss = 1.8765
  ...
```

---

## Saran Perbaikan

### Perbaikan Jangka Pendek

| Item Perbaikan | Deskripsi |
|----------------|-----------|
| Tambah blok residual | 4 → 8 → 16 blok |
| Tambah channel | 64 → 128 → 256 |
| Tambah simulasi | 100 → 400 → 800 |
| Dataset lebih besar | 50 → 200 → 1000 game/iterasi |

### Perbaikan Jangka Panjang

- Dukung papan 19×19
- Tambahkan target pelatihan tambahan (prediksi teritori)
- Implementasikan self-play paralel
- Tambahkan akselerasi GPU

---

## Bacaan Lanjutan

- [Detail Arsitektur Neural Network](../neural-network) — Desain network yang lebih mendalam
- [Detail Implementasi MCTS](../mcts-implementation) — Teknik pencarian lanjutan
- [Analisis Mekanisme Pelatihan KataGo](../training) — Sistem pelatihan tingkat produksi
- [Panduan Paper Kunci](../papers) — Dasar teori
