---
sidebar_position: 12
title: 从零打造围棋 AI
description: 一步步教你实现一个简易版的 AlphaGo Zero 风格围棋 AI
---

# 从零打造围棋 AI

本文带你一步步实现一个简化版的 AlphaGo Zero 风格围棋 AI，涵盖游戏逻辑、神经网络、MCTS 与训练流程。

:::info 学习目标
完成本教程后，你将拥有一个能够：
- 在 9×9 棋盘上自我对弈
- 通过强化学习持续进步
- 达到业余初级棋力的围棋 AI
:::

---

## 项目架构

```
mini-alphago/
├── game/
│   ├── __init__.py
│   ├── board.py          # 棋盘逻辑
│   ├── rules.py          # 规则实现
│   └── state.py          # 游戏状态
├── model/
│   ├── __init__.py
│   ├── network.py        # 神经网络
│   └── features.py       # 特征编码
├── mcts/
│   ├── __init__.py
│   ├── node.py           # MCTS 节点
│   └── search.py         # MCTS 搜索
├── training/
│   ├── __init__.py
│   ├── self_play.py      # 自我对弈
│   └── trainer.py        # 训练器
├── main.py               # 主程序
└── requirements.txt
```

---

## 第一步：棋盘与规则

### 棋盘实现

```python
# game/board.py
import numpy as np

class Board:
    """围棋棋盘"""

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
        """复制棋盘"""
        new_board = Board(self.size)
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.ko_point = self.ko_point
        new_board.history = self.history.copy()
        return new_board

    def get_opponent(self, player):
        """获取对手"""
        return self.WHITE if player == self.BLACK else self.BLACK

    def is_on_board(self, x, y):
        """检查是否在棋盘上"""
        return 0 <= x < self.size and 0 <= y < self.size

    def get_neighbors(self, x, y):
        """获取相邻点"""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if self.is_on_board(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def get_group(self, x, y):
        """获取棋串（连通的同色棋子）"""
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
        """计算气数"""
        _, liberties = self.get_group(x, y)
        return len(liberties)

    def remove_group(self, group):
        """移除棋串"""
        for x, y in group:
            self.board[x, y] = self.EMPTY

    def is_legal(self, x, y, player=None):
        """检查是否为合法手"""
        if player is None:
            player = self.current_player

        # 检查是否为空点
        if self.board[x, y] != self.EMPTY:
            return False

        # 检查是否为 Ko
        if self.ko_point == (x, y):
            return False

        # 模拟落子
        test_board = self.copy()
        test_board.board[x, y] = player

        # 先检查是否能提子
        opponent = self.get_opponent(player)
        captured = []
        for nx, ny in self.get_neighbors(x, y):
            if test_board.board[nx, ny] == opponent:
                group, liberties = test_board.get_group(nx, ny)
                if len(liberties) == 0:
                    captured.extend(group)

        if captured:
            return True

        # 检查自杀
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

        # 设置 Ko
        if len(captured) == 1:
            cx, cy = list(captured)[0]
            _, my_liberties = self.get_group(x, y)
            if len(my_liberties) == 1:
                self.ko_point = (cx, cy)
            else:
                self.ko_point = None
        else:
            self.ko_point = None

        # 记录历史
        self.history.append((x, y, player))

        # 切换玩家
        self.current_player = opponent

        return True

    def pass_move(self):
        """虚手（pass）"""
        self.history.append((-1, -1, self.current_player))
        self.current_player = self.get_opponent(self.current_player)
        self.ko_point = None

    def is_game_over(self):
        """检查是否结束"""
        if len(self.history) < 2:
            return False
        # 双方连续 pass
        return (self.history[-1][0] == -1 and
                self.history[-2][0] == -1)

    def get_legal_moves(self):
        """获取所有合法手"""
        moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.is_legal(x, y):
                    moves.append((x, y))
        moves.append((-1, -1))  # pass
        return moves

    def score(self):
        """计算胜负（简易数子法）"""
        black_score = np.sum(self.board == self.BLACK)
        white_score = np.sum(self.board == self.WHITE)

        # 简易领地计算
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

## 第二步：特征编码

### 输入特征

```python
# model/features.py
import numpy as np

def encode_board(board):
    """
    将棋盘编码为神经网络输入

    特征平面：
    0: 己方棋子
    1: 对方棋子
    2: 空点
    3: 上一手位置
    4: 倒数第二手位置
    5: 合法手位置
    6: 轮到黑棋（全 1 或全 0）
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

    # 轮到谁
    if current == board.BLACK:
        features[6] = np.ones((size, size), dtype=np.float32)

    return features
```

---

## 第三步：神经网络

### 双头网络架构

```python
# model/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """残差块"""

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
    """策略-价值双头网络"""

    def __init__(self, board_size=9, input_channels=7, num_filters=64, num_blocks=4):
        super().__init__()
        self.board_size = board_size

        # 初始卷积
        self.conv_input = nn.Conv2d(input_channels, num_filters, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)

        # 残差块
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
        # 共享主干
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
    """创建网络"""
    return PolicyValueNetwork(
        board_size=board_size,
        input_channels=7,
        num_filters=64,
        num_blocks=4
    )
```

---

## 第四步：MCTS 实现

### 节点类

```python
# mcts/node.py
import numpy as np

class MCTSNode:
    """MCTS 节点"""

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
        """展开节点"""
        for move in legal_moves:
            if move not in self.children:
                idx = move[0] * 9 + move[1] if move[0] >= 0 else 81
                self.children[move] = MCTSNode(prior=np.exp(policy[idx]))

    def select_child(self, c_puct=1.5):
        """用 PUCT 选择子节点"""
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

### 搜索实现

```python
# mcts/search.py
import numpy as np
import torch
from .node import MCTSNode

class MCTS:
    """蒙特卡洛树搜索"""

    def __init__(self, network, board_size=9, num_simulations=100, c_puct=1.5):
        self.network = network
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, board, add_noise=False):
        """执行 MCTS 搜索"""
        root = MCTSNode()

        # 评估根节点
        policy, value = self.evaluate(board)
        legal_moves = board.get_legal_moves()
        root.expand(policy, legal_moves)

        # 加入 Dirichlet 噪声（训练时）
        if add_noise:
            self.add_dirichlet_noise(root)

        # 执行模拟
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

            # 从搜索起点的视角计算价值
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
        """用神经网络评估"""
        from model.features import encode_board

        features = encode_board(board)
        features = torch.tensor(features).unsqueeze(0)

        self.network.eval()
        with torch.no_grad():
            policy, value = self.network(features)

        return policy[0].numpy(), value[0].item()

    def add_dirichlet_noise(self, root, alpha=0.3, epsilon=0.25):
        """加入探索噪声"""
        noise = np.random.dirichlet([alpha] * len(root.children))
        for i, child in enumerate(root.children.values()):
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]

    def get_policy(self, root, temperature=1.0):
        """从搜索结果获取策略"""
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
        """选择落子"""
        policy = self.get_policy(root, temperature)
        idx = np.random.choice(len(policy), p=policy)

        if idx == self.board_size ** 2:
            return (-1, -1)
        else:
            return (idx // self.board_size, idx % self.board_size)
```

---

## 第五步：自我对弈

```python
# training/self_play.py
import numpy as np
from game.board import Board
from model.features import encode_board

def self_play_game(mcts, temperature=1.0, temp_threshold=30):
    """执行一局自我对弈"""
    board = Board(size=9)
    game_history = []

    move_count = 0
    while not board.is_game_over() and move_count < 200:
        # MCTS 搜索
        root = mcts.search(board, add_noise=True)

        # 获取策略
        temp = temperature if move_count < temp_threshold else 0.0
        policy = mcts.get_policy(root, temp)

        # 记录训练数据
        features = encode_board(board)
        game_history.append({
            'features': features,
            'policy': policy,
            'player': board.current_player
        })

        # 选择并执行落子
        move = mcts.select_move(root, temp)
        if move[0] >= 0:
            board.play(move[0], move[1])
        else:
            board.pass_move()

        move_count += 1

    # 计算胜负
    score = board.score()
    winner = Board.BLACK if score > 0 else (Board.WHITE if score < 0 else 0)

    # 标记价值
    for data in game_history:
        if winner == 0:
            data['value'] = 0.0
        elif data['player'] == winner:
            data['value'] = 1.0
        else:
            data['value'] = -1.0

    return game_history


def generate_training_data(mcts, num_games=100):
    """生成训练数据"""
    all_data = []

    for i in range(num_games):
        print(f"Self-play game {i+1}/{num_games}")
        game_data = self_play_game(mcts)
        all_data.extend(game_data)

    return all_data
```

---

## 第六步：训练器

```python
# training/trainer.py
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class Trainer:
    """训练器"""

    def __init__(self, network, learning_rate=0.001):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    def train_step(self, batch):
        """单步训练"""
        features, target_policy, target_value = batch

        self.network.train()
        self.optimizer.zero_grad()

        # 前向传播
        pred_policy, pred_value = self.network(features)

        # 计算损失
        policy_loss = F.kl_div(pred_policy, target_policy, reduction='batchmean')
        value_loss = F.mse_loss(pred_value.squeeze(), target_value)
        total_loss = policy_loss + value_loss

        # 反向传播
        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }

    def train_epoch(self, data, batch_size=32):
        """训练一个 epoch"""
        # 准备数据
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
        """保存模型"""
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        """加载模型"""
        self.network.load_state_dict(torch.load(path))
```

---

## 第七步：主程序

```python
# main.py
from model.network import create_network
from mcts.search import MCTS
from training.self_play import generate_training_data
from training.trainer import Trainer

def main():
    # 创建网络
    network = create_network(board_size=9)
    mcts = MCTS(network, board_size=9, num_simulations=100)
    trainer = Trainer(network)

    # 训练循环
    num_iterations = 100
    games_per_iteration = 50
    epochs_per_iteration = 10

    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")

        # 自我对弈
        print("Generating self-play games...")
        training_data = generate_training_data(mcts, num_games=games_per_iteration)

        # 训练
        print("Training...")
        for epoch in range(epochs_per_iteration):
            loss = trainer.train_epoch(training_data)
            print(f"  Epoch {epoch + 1}: loss = {loss:.4f}")

        # 保存
        trainer.save(f"model_iter_{iteration + 1}.pt")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
```

---

## 执行与测试

### 安装依赖

```bash
pip install torch numpy
```

### 执行训练

```bash
python main.py
```

### 预期输出

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

## 改进建议

### 短期改进

| 改进项目 | 说明 |
|---------|------|
| 增加残差块 | 4 → 8 → 16 块 |
| 增加通道数 | 64 → 128 → 256 |
| 增加模拟次数 | 100 → 400 → 800 |
| 更大的训练集 | 50 → 200 → 1000 局/迭代 |

### 长期改进

- 支持 19×19 棋盘
- 加入辅助训练目标（领地预测）
- 实现并行自我对弈
- 加入 GPU 加速

---

## 延伸阅读

- [神经网络架构详解](../neural-network) — 更深入的网络设计
- [MCTS 实现细节](../mcts-implementation) — 进阶搜索技术
- [KataGo 训练机制解析](../training) — 生产级训练系统
- [关键论文导读](../papers) — 理论基础
