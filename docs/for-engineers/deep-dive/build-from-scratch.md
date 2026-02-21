---
sidebar_position: 12
title: 從零打造圍棋 AI
description: 一步步教你實作一個簡易版的 AlphaGo Zero 風格圍棋 AI
---

# 從零打造圍棋 AI

本文帶你一步步實作一個簡化版的 AlphaGo Zero 風格圍棋 AI，涵蓋遊戲邏輯、神經網路、MCTS 與訓練流程。

:::info 學習目標
完成本教學後，你將擁有一個能夠：
- 在 9×9 棋盤上自我對弈
- 透過強化學習持續進步
- 達到業餘初級棋力的圍棋 AI
:::

---

## 專案架構

```
mini-alphago/
├── game/
│   ├── __init__.py
│   ├── board.py          # 棋盤邏輯
│   ├── rules.py          # 規則實作
│   └── state.py          # 遊戲狀態
├── model/
│   ├── __init__.py
│   ├── network.py        # 神經網路
│   └── features.py       # 特徵編碼
├── mcts/
│   ├── __init__.py
│   ├── node.py           # MCTS 節點
│   └── search.py         # MCTS 搜索
├── training/
│   ├── __init__.py
│   ├── self_play.py      # 自我對弈
│   └── trainer.py        # 訓練器
├── main.py               # 主程式
└── requirements.txt
```

---

## 第一步：棋盤與規則

### 棋盤實作

```python
# game/board.py
import numpy as np

class Board:
    """圍棋棋盤"""

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
        """複製棋盤"""
        new_board = Board(self.size)
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.ko_point = self.ko_point
        new_board.history = self.history.copy()
        return new_board

    def get_opponent(self, player):
        """取得對手"""
        return self.WHITE if player == self.BLACK else self.BLACK

    def is_on_board(self, x, y):
        """檢查是否在棋盤上"""
        return 0 <= x < self.size and 0 <= y < self.size

    def get_neighbors(self, x, y):
        """取得相鄰點"""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if self.is_on_board(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def get_group(self, x, y):
        """取得棋串（連通的同色棋子）"""
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
        """計算氣數"""
        _, liberties = self.get_group(x, y)
        return len(liberties)

    def remove_group(self, group):
        """移除棋串"""
        for x, y in group:
            self.board[x, y] = self.EMPTY

    def is_legal(self, x, y, player=None):
        """檢查是否為合法手"""
        if player is None:
            player = self.current_player

        # 檢查是否為空點
        if self.board[x, y] != self.EMPTY:
            return False

        # 檢查是否為 Ko
        if self.ko_point == (x, y):
            return False

        # 模擬落子
        test_board = self.copy()
        test_board.board[x, y] = player

        # 先檢查是否能提子
        opponent = self.get_opponent(player)
        captured = []
        for nx, ny in self.get_neighbors(x, y):
            if test_board.board[nx, ny] == opponent:
                group, liberties = test_board.get_group(nx, ny)
                if len(liberties) == 0:
                    captured.extend(group)

        if captured:
            return True

        # 檢查自殺
        _, liberties = test_board.get_group(x, y)
        if len(liberties) == 0:
            return False

        return True

    def play(self, x, y):
        """落子"""
        if not self.is_legal(x, y):
            return False

        player = self.current_player
        opponent = self.get_opponent(player)

        # 落子
        self.board[x, y] = player

        # 提子
        captured = []
        for nx, ny in self.get_neighbors(x, y):
            if self.board[nx, ny] == opponent:
                group, liberties = self.get_group(nx, ny)
                if len(liberties) == 0:
                    captured.extend(group)
                    self.remove_group(group)

        # 設定 Ko
        if len(captured) == 1:
            cx, cy = list(captured)[0]
            _, my_liberties = self.get_group(x, y)
            if len(my_liberties) == 1:
                self.ko_point = (cx, cy)
            else:
                self.ko_point = None
        else:
            self.ko_point = None

        # 記錄歷史
        self.history.append((x, y, player))

        # 切換玩家
        self.current_player = opponent

        return True

    def pass_move(self):
        """虛手（pass）"""
        self.history.append((-1, -1, self.current_player))
        self.current_player = self.get_opponent(self.current_player)
        self.ko_point = None

    def is_game_over(self):
        """檢查是否結束"""
        if len(self.history) < 2:
            return False
        # 雙方連續 pass
        return (self.history[-1][0] == -1 and
                self.history[-2][0] == -1)

    def get_legal_moves(self):
        """取得所有合法手"""
        moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.is_legal(x, y):
                    moves.append((x, y))
        moves.append((-1, -1))  # pass
        return moves

    def score(self):
        """計算勝負（簡易數子法）"""
        black_score = np.sum(self.board == self.BLACK)
        white_score = np.sum(self.board == self.WHITE)

        # 簡易領地計算
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

## 第二步：特徵編碼

### 輸入特徵

```python
# model/features.py
import numpy as np

def encode_board(board):
    """
    將棋盤編碼為神經網路輸入

    特徵平面：
    0: 己方棋子
    1: 對方棋子
    2: 空點
    3: 上一手位置
    4: 倒數第二手位置
    5: 合法手位置
    6: 輪到黑棋（全 1 或全 0）
    """
    size = board.size
    features = np.zeros((7, size, size), dtype=np.float32)

    current = board.current_player
    opponent = board.get_opponent(current)

    # 基本棋子位置
    features[0] = (board.board == current).astype(np.float32)
    features[1] = (board.board == opponent).astype(np.float32)
    features[2] = (board.board == board.EMPTY).astype(np.float32)

    # 最近的落子
    if len(board.history) >= 1:
        x, y, _ = board.history[-1]
        if x >= 0:
            features[3, x, y] = 1.0

    if len(board.history) >= 2:
        x, y, _ = board.history[-2]
        if x >= 0:
            features[4, x, y] = 1.0

    # 合法手
    for x in range(size):
        for y in range(size):
            if board.is_legal(x, y):
                features[5, x, y] = 1.0

    # 輪到誰
    if current == board.BLACK:
        features[6] = np.ones((size, size), dtype=np.float32)

    return features
```

---

## 第三步：神經網路

### 雙頭網路架構

```python
# model/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """殘差塊"""

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
    """策略-價值雙頭網路"""

    def __init__(self, board_size=9, input_channels=7, num_filters=64, num_blocks=4):
        super().__init__()
        self.board_size = board_size

        # 初始卷積
        self.conv_input = nn.Conv2d(input_channels, num_filters, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)

        # 殘差塊
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
        # 共享主幹
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
    """建立網路"""
    return PolicyValueNetwork(
        board_size=board_size,
        input_channels=7,
        num_filters=64,
        num_blocks=4
    )
```

---

## 第四步：MCTS 實作

### 節點類別

```python
# mcts/node.py
import numpy as np

class MCTSNode:
    """MCTS 節點"""

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
        """展開節點"""
        for move in legal_moves:
            if move not in self.children:
                idx = move[0] * 9 + move[1] if move[0] >= 0 else 81
                self.children[move] = MCTSNode(prior=np.exp(policy[idx]))

    def select_child(self, c_puct=1.5):
        """用 PUCT 選擇子節點"""
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

### 搜索實作

```python
# mcts/search.py
import numpy as np
import torch
from .node import MCTSNode

class MCTS:
    """蒙地卡羅樹搜索"""

    def __init__(self, network, board_size=9, num_simulations=100, c_puct=1.5):
        self.network = network
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, board, add_noise=False):
        """執行 MCTS 搜索"""
        root = MCTSNode()

        # 評估根節點
        policy, value = self.evaluate(board)
        legal_moves = board.get_legal_moves()
        root.expand(policy, legal_moves)

        # 加入 Dirichlet 噪聲（訓練時）
        if add_noise:
            self.add_dirichlet_noise(root)

        # 執行模擬
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

            # 從搜索起點的視角計算價值
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
        """用神經網路評估"""
        from model.features import encode_board

        features = encode_board(board)
        features = torch.tensor(features).unsqueeze(0)

        self.network.eval()
        with torch.no_grad():
            policy, value = self.network(features)

        return policy[0].numpy(), value[0].item()

    def add_dirichlet_noise(self, root, alpha=0.3, epsilon=0.25):
        """加入探索噪聲"""
        noise = np.random.dirichlet([alpha] * len(root.children))
        for i, child in enumerate(root.children.values()):
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]

    def get_policy(self, root, temperature=1.0):
        """從搜索結果取得策略"""
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
        """選擇落子"""
        policy = self.get_policy(root, temperature)
        idx = np.random.choice(len(policy), p=policy)

        if idx == self.board_size ** 2:
            return (-1, -1)
        else:
            return (idx // self.board_size, idx % self.board_size)
```

---

## 第五步：自我對弈

```python
# training/self_play.py
import numpy as np
from game.board import Board
from model.features import encode_board

def self_play_game(mcts, temperature=1.0, temp_threshold=30):
    """執行一局自我對弈"""
    board = Board(size=9)
    game_history = []

    move_count = 0
    while not board.is_game_over() and move_count < 200:
        # MCTS 搜索
        root = mcts.search(board, add_noise=True)

        # 取得策略
        temp = temperature if move_count < temp_threshold else 0.0
        policy = mcts.get_policy(root, temp)

        # 記錄訓練資料
        features = encode_board(board)
        game_history.append({
            'features': features,
            'policy': policy,
            'player': board.current_player
        })

        # 選擇並執行落子
        move = mcts.select_move(root, temp)
        if move[0] >= 0:
            board.play(move[0], move[1])
        else:
            board.pass_move()

        move_count += 1

    # 計算勝負
    score = board.score()
    winner = Board.BLACK if score > 0 else (Board.WHITE if score < 0 else 0)

    # 標記價值
    for data in game_history:
        if winner == 0:
            data['value'] = 0.0
        elif data['player'] == winner:
            data['value'] = 1.0
        else:
            data['value'] = -1.0

    return game_history


def generate_training_data(mcts, num_games=100):
    """產生訓練資料"""
    all_data = []

    for i in range(num_games):
        print(f"Self-play game {i+1}/{num_games}")
        game_data = self_play_game(mcts)
        all_data.extend(game_data)

    return all_data
```

---

## 第六步：訓練器

```python
# training/trainer.py
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class Trainer:
    """訓練器"""

    def __init__(self, network, learning_rate=0.001):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    def train_step(self, batch):
        """單步訓練"""
        features, target_policy, target_value = batch

        self.network.train()
        self.optimizer.zero_grad()

        # 前向傳播
        pred_policy, pred_value = self.network(features)

        # 計算損失
        policy_loss = F.kl_div(pred_policy, target_policy, reduction='batchmean')
        value_loss = F.mse_loss(pred_value.squeeze(), target_value)
        total_loss = policy_loss + value_loss

        # 反向傳播
        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }

    def train_epoch(self, data, batch_size=32):
        """訓練一個 epoch"""
        # 準備資料
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
        """儲存模型"""
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        """載入模型"""
        self.network.load_state_dict(torch.load(path))
```

---

## 第七步：主程式

```python
# main.py
from model.network import create_network
from mcts.search import MCTS
from training.self_play import generate_training_data
from training.trainer import Trainer

def main():
    # 建立網路
    network = create_network(board_size=9)
    mcts = MCTS(network, board_size=9, num_simulations=100)
    trainer = Trainer(network)

    # 訓練循環
    num_iterations = 100
    games_per_iteration = 50
    epochs_per_iteration = 10

    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")

        # 自我對弈
        print("Generating self-play games...")
        training_data = generate_training_data(mcts, num_games=games_per_iteration)

        # 訓練
        print("Training...")
        for epoch in range(epochs_per_iteration):
            loss = trainer.train_epoch(training_data)
            print(f"  Epoch {epoch + 1}: loss = {loss:.4f}")

        # 儲存
        trainer.save(f"model_iter_{iteration + 1}.pt")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
```

---

## 執行與測試

### 安裝依賴

```bash
pip install torch numpy
```

### 執行訓練

```bash
python main.py
```

### 預期輸出

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

## 改進建議

### 短期改進

| 改進項目 | 說明 |
|---------|------|
| 增加殘差塊 | 4 → 8 → 16 塊 |
| 增加通道數 | 64 → 128 → 256 |
| 增加模擬次數 | 100 → 400 → 800 |
| 更大的訓練集 | 50 → 200 → 1000 局/迭代 |

### 長期改進

- 支援 19×19 棋盤
- 加入輔助訓練目標（領地預測）
- 實作並行自我對弈
- 加入 GPU 加速

---

## 延伸閱讀

- [神經網路架構詳解](../neural-network) — 更深入的網路設計
- [MCTS 實作細節](../mcts-implementation) — 進階搜索技術
- [KataGo 訓練機制解析](../training) — 生產級訓練系統
- [關鍵論文導讀](../papers) — 理論基礎
