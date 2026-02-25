---
sidebar_position: 12
title: शून्य से गो AI बनाना
description: चरण-दर-चरण सरलीकृत AlphaGo Zero शैली गो AI का कार्यान्वयन सिखाएं
---

# शून्य से गो AI बनाना

यह लेख आपको चरण-दर-चरण सरलीकृत AlphaGo Zero शैली गो AI बनाना सिखाता है, जिसमें गेम लॉजिक, न्यूरल नेटवर्क, MCTS और प्रशिक्षण प्रक्रिया शामिल है।

:::info सीखने के लक्ष्य
इस ट्यूटोरियल को पूरा करने के बाद, आपके पास एक गो AI होगा जो:
- 9×9 बोर्ड पर सेल्फ-प्ले कर सकता है
- रिइन्फोर्समेंट लर्निंग से निरंतर सुधार कर सकता है
- एमेच्योर शुरुआती स्तर की खेल शक्ति प्राप्त कर सकता है
:::

---

## प्रोजेक्ट आर्किटेक्चर

```
mini-alphago/
├── game/
│   ├── __init__.py
│   ├── board.py          # बोर्ड लॉजिक
│   ├── rules.py          # नियम कार्यान्वयन
│   └── state.py          # गेम स्थिति
├── model/
│   ├── __init__.py
│   ├── network.py        # न्यूरल नेटवर्क
│   └── features.py       # फीचर एनकोडिंग
├── mcts/
│   ├── __init__.py
│   ├── node.py           # MCTS नोड
│   └── search.py         # MCTS सर्च
├── training/
│   ├── __init__.py
│   ├── self_play.py      # सेल्फ-प्ले
│   └── trainer.py        # ट्रेनर
├── main.py               # मुख्य प्रोग्राम
└── requirements.txt
```

---

## पहला चरण: बोर्ड और नियम

### बोर्ड कार्यान्वयन

```python
# game/board.py
import numpy as np

class Board:
    """गो बोर्ड"""

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
        """बोर्ड कॉपी करें"""
        new_board = Board(self.size)
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.ko_point = self.ko_point
        new_board.history = self.history.copy()
        return new_board

    def get_opponent(self, player):
        """प्रतिद्वंद्वी प्राप्त करें"""
        return self.WHITE if player == self.BLACK else self.BLACK

    def is_on_board(self, x, y):
        """बोर्ड पर है या नहीं जांचें"""
        return 0 <= x < self.size and 0 <= y < self.size

    def get_neighbors(self, x, y):
        """पड़ोसी बिंदु प्राप्त करें"""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if self.is_on_board(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def get_group(self, x, y):
        """पत्थर समूह प्राप्त करें (जुड़े समान रंग के पत्थर)"""
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
        """लिबर्टी गिनें"""
        _, liberties = self.get_group(x, y)
        return len(liberties)

    def remove_group(self, group):
        """पत्थर समूह हटाएं"""
        for x, y in group:
            self.board[x, y] = self.EMPTY

    def is_legal(self, x, y, player=None):
        """वैध चाल है या नहीं जांचें"""
        if player is None:
            player = self.current_player

        # खाली बिंदु है या नहीं जांचें
        if self.board[x, y] != self.EMPTY:
            return False

        # Ko है या नहीं जांचें
        if self.ko_point == (x, y):
            return False

        # चाल सिमुलेट करें
        test_board = self.copy()
        test_board.board[x, y] = player

        # पहले कैप्चर कर सकते हैं या नहीं जांचें
        opponent = self.get_opponent(player)
        captured = []
        for nx, ny in self.get_neighbors(x, y):
            if test_board.board[nx, ny] == opponent:
                group, liberties = test_board.get_group(nx, ny)
                if len(liberties) == 0:
                    captured.extend(group)

        if captured:
            return True

        # आत्महत्या जांचें
        _, liberties = test_board.get_group(x, y)
        if len(liberties) == 0:
            return False

        return True

    def play(self, x, y):
        """चाल खेलें"""
        if not self.is_legal(x, y):
            return False

        player = self.current_player
        opponent = self.get_opponent(player)

        # चाल खेलें
        self.board[x, y] = player

        # कैप्चर करें
        captured = []
        for nx, ny in self.get_neighbors(x, y):
            if self.board[nx, ny] == opponent:
                group, liberties = self.get_group(nx, ny)
                if len(liberties) == 0:
                    captured.extend(group)
                    self.remove_group(group)

        # Ko सेट करें
        if len(captured) == 1:
            cx, cy = list(captured)[0]
            _, my_liberties = self.get_group(x, y)
            if len(my_liberties) == 1:
                self.ko_point = (cx, cy)
            else:
                self.ko_point = None
        else:
            self.ko_point = None

        # इतिहास रिकॉर्ड करें
        self.history.append((x, y, player))

        # खिलाड़ी बदलें
        self.current_player = opponent

        return True

    def pass_move(self):
        """पास (खाली चाल)"""
        self.history.append((-1, -1, self.current_player))
        self.current_player = self.get_opponent(self.current_player)
        self.ko_point = None

    def is_game_over(self):
        """गेम समाप्त है या नहीं जांचें"""
        if len(self.history) < 2:
            return False
        # दोनों पक्ष लगातार पास
        return (self.history[-1][0] == -1 and
                self.history[-2][0] == -1)

    def get_legal_moves(self):
        """सभी वैध चालें प्राप्त करें"""
        moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.is_legal(x, y):
                    moves.append((x, y))
        moves.append((-1, -1))  # pass
        return moves

    def score(self):
        """जीत/हार गणना करें (सरल एरिया स्कोरिंग)"""
        black_score = np.sum(self.board == self.BLACK)
        white_score = np.sum(self.board == self.WHITE)

        # सरल क्षेत्र गणना
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

## दूसरा चरण: फीचर एनकोडिंग

### इनपुट फीचर्स

```python
# model/features.py
import numpy as np

def encode_board(board):
    """
    बोर्ड को न्यूरल नेटवर्क इनपुट में एनकोड करें

    फीचर प्लेन:
    0: अपने पत्थर
    1: प्रतिद्वंद्वी के पत्थर
    2: खाली बिंदु
    3: पिछली चाल स्थिति
    4: दूसरी अंतिम चाल स्थिति
    5: वैध चाल स्थान
    6: काले की बारी (सभी 1 या सभी 0)
    """
    size = board.size
    features = np.zeros((7, size, size), dtype=np.float32)

    current = board.current_player
    opponent = board.get_opponent(current)

    # मूल पत्थर स्थिति
    features[0] = (board.board == current).astype(np.float32)
    features[1] = (board.board == opponent).astype(np.float32)
    features[2] = (board.board == board.EMPTY).astype(np.float32)

    # हाल की चालें
    if len(board.history) >= 1:
        x, y, _ = board.history[-1]
        if x >= 0:
            features[3, x, y] = 1.0

    if len(board.history) >= 2:
        x, y, _ = board.history[-2]
        if x >= 0:
            features[4, x, y] = 1.0

    # वैध चालें
    for x in range(size):
        for y in range(size):
            if board.is_legal(x, y):
                features[5, x, y] = 1.0

    # किसकी बारी
    if current == board.BLACK:
        features[6] = np.ones((size, size), dtype=np.float32)

    return features
```

---

## तीसरा चरण: न्यूरल नेटवर्क

### ड्यूअल-हेड नेटवर्क आर्किटेक्चर

```python
# model/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """रेसिड्यूअल ब्लॉक"""

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
    """पॉलिसी-वैल्यू ड्यूअल-हेड नेटवर्क"""

    def __init__(self, board_size=9, input_channels=7, num_filters=64, num_blocks=4):
        super().__init__()
        self.board_size = board_size

        # प्रारंभिक कन्वोल्यूशन
        self.conv_input = nn.Conv2d(input_channels, num_filters, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)

        # रेसिड्यूअल ब्लॉक
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
        # साझा ट्रंक
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
    """नेटवर्क बनाएं"""
    return PolicyValueNetwork(
        board_size=board_size,
        input_channels=7,
        num_filters=64,
        num_blocks=4
    )
```

---

## चौथा चरण: MCTS कार्यान्वयन

### नोड क्लास

```python
# mcts/node.py
import numpy as np

class MCTSNode:
    """MCTS नोड"""

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
        """नोड विस्तार करें"""
        for move in legal_moves:
            if move not in self.children:
                idx = move[0] * 9 + move[1] if move[0] >= 0 else 81
                self.children[move] = MCTSNode(prior=np.exp(policy[idx]))

    def select_child(self, c_puct=1.5):
        """PUCT से चाइल्ड नोड चुनें"""
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

### सर्च कार्यान्वयन

```python
# mcts/search.py
import numpy as np
import torch
from .node import MCTSNode

class MCTS:
    """मोंटे कार्लो ट्री सर्च"""

    def __init__(self, network, board_size=9, num_simulations=100, c_puct=1.5):
        self.network = network
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, board, add_noise=False):
        """MCTS सर्च निष्पादित करें"""
        root = MCTSNode()

        # रूट नोड मूल्यांकन
        policy, value = self.evaluate(board)
        legal_moves = board.get_legal_moves()
        root.expand(policy, legal_moves)

        # Dirichlet नॉइज़ जोड़ें (प्रशिक्षण के दौरान)
        if add_noise:
            self.add_dirichlet_noise(root)

        # सिमुलेशन निष्पादित करें
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

            # सर्च प्रारंभ बिंदु से मूल्य गणना
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
        """न्यूरल नेटवर्क से मूल्यांकन"""
        from model.features import encode_board

        features = encode_board(board)
        features = torch.tensor(features).unsqueeze(0)

        self.network.eval()
        with torch.no_grad():
            policy, value = self.network(features)

        return policy[0].numpy(), value[0].item()

    def add_dirichlet_noise(self, root, alpha=0.3, epsilon=0.25):
        """अन्वेषण नॉइज़ जोड़ें"""
        noise = np.random.dirichlet([alpha] * len(root.children))
        for i, child in enumerate(root.children.values()):
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]

    def get_policy(self, root, temperature=1.0):
        """सर्च परिणाम से पॉलिसी प्राप्त करें"""
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
        """चाल चुनें"""
        policy = self.get_policy(root, temperature)
        idx = np.random.choice(len(policy), p=policy)

        if idx == self.board_size ** 2:
            return (-1, -1)
        else:
            return (idx // self.board_size, idx % self.board_size)
```

---

## पांचवां चरण: सेल्फ-प्ले

```python
# training/self_play.py
import numpy as np
from game.board import Board
from model.features import encode_board

def self_play_game(mcts, temperature=1.0, temp_threshold=30):
    """एक सेल्फ-प्ले गेम खेलें"""
    board = Board(size=9)
    game_history = []

    move_count = 0
    while not board.is_game_over() and move_count < 200:
        # MCTS सर्च
        root = mcts.search(board, add_noise=True)

        # पॉलिसी प्राप्त करें
        temp = temperature if move_count < temp_threshold else 0.0
        policy = mcts.get_policy(root, temp)

        # प्रशिक्षण डेटा रिकॉर्ड करें
        features = encode_board(board)
        game_history.append({
            'features': features,
            'policy': policy,
            'player': board.current_player
        })

        # चाल चुनें और खेलें
        move = mcts.select_move(root, temp)
        if move[0] >= 0:
            board.play(move[0], move[1])
        else:
            board.pass_move()

        move_count += 1

    # जीत/हार गणना
    score = board.score()
    winner = Board.BLACK if score > 0 else (Board.WHITE if score < 0 else 0)

    # मूल्य चिह्नित करें
    for data in game_history:
        if winner == 0:
            data['value'] = 0.0
        elif data['player'] == winner:
            data['value'] = 1.0
        else:
            data['value'] = -1.0

    return game_history


def generate_training_data(mcts, num_games=100):
    """प्रशिक्षण डेटा उत्पन्न करें"""
    all_data = []

    for i in range(num_games):
        print(f"Self-play game {i+1}/{num_games}")
        game_data = self_play_game(mcts)
        all_data.extend(game_data)

    return all_data
```

---

## छठा चरण: ट्रेनर

```python
# training/trainer.py
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class Trainer:
    """ट्रेनर"""

    def __init__(self, network, learning_rate=0.001):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    def train_step(self, batch):
        """एक प्रशिक्षण चरण"""
        features, target_policy, target_value = batch

        self.network.train()
        self.optimizer.zero_grad()

        # फॉरवर्ड प्रोपेगेशन
        pred_policy, pred_value = self.network(features)

        # लॉस गणना
        policy_loss = F.kl_div(pred_policy, target_policy, reduction='batchmean')
        value_loss = F.mse_loss(pred_value.squeeze(), target_value)
        total_loss = policy_loss + value_loss

        # बैकप्रोपेगेशन
        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }

    def train_epoch(self, data, batch_size=32):
        """एक epoch प्रशिक्षित करें"""
        # डेटा तैयार करें
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
        """मॉडल सेव करें"""
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        """मॉडल लोड करें"""
        self.network.load_state_dict(torch.load(path))
```

---

## सातवां चरण: मुख्य प्रोग्राम

```python
# main.py
from model.network import create_network
from mcts.search import MCTS
from training.self_play import generate_training_data
from training.trainer import Trainer

def main():
    # नेटवर्क बनाएं
    network = create_network(board_size=9)
    mcts = MCTS(network, board_size=9, num_simulations=100)
    trainer = Trainer(network)

    # प्रशिक्षण लूप
    num_iterations = 100
    games_per_iteration = 50
    epochs_per_iteration = 10

    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")

        # सेल्फ-प्ले
        print("Generating self-play games...")
        training_data = generate_training_data(mcts, num_games=games_per_iteration)

        # प्रशिक्षण
        print("Training...")
        for epoch in range(epochs_per_iteration):
            loss = trainer.train_epoch(training_data)
            print(f"  Epoch {epoch + 1}: loss = {loss:.4f}")

        # सेव करें
        trainer.save(f"model_iter_{iteration + 1}.pt")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
```

---

## चलाएं और परीक्षण करें

### डिपेंडेंसी इंस्टॉल करें

```bash
pip install torch numpy
```

### प्रशिक्षण चलाएं

```bash
python main.py
```

### अपेक्षित आउटपुट

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

## सुधार सुझाव

### अल्पकालिक सुधार

| सुधार आइटम | विवरण |
|---------|------|
| रेसिड्यूअल ब्लॉक बढ़ाएं | 4 → 8 → 16 ब्लॉक |
| चैनल बढ़ाएं | 64 → 128 → 256 |
| सिमुलेशन बढ़ाएं | 100 → 400 → 800 |
| बड़ा प्रशिक्षण सेट | 50 → 200 → 1000 गेम/इटरेशन |

### दीर्घकालिक सुधार

- 19×19 बोर्ड समर्थन
- सहायक प्रशिक्षण लक्ष्य जोड़ें (क्षेत्र पूर्वानुमान)
- पैरेलल सेल्फ-प्ले कार्यान्वित करें
- GPU त्वरण जोड़ें

---

## आगे पढ़ें

- [न्यूरल नेटवर्क आर्किटेक्चर विस्तृत व्याख्या](../neural-network) — गहरा नेटवर्क डिज़ाइन
- [MCTS कार्यान्वयन विवरण](../mcts-implementation) — उन्नत सर्च तकनीकें
- [KataGo प्रशिक्षण तंत्र विश्लेषण](../training) — प्रोडक्शन-ग्रेड प्रशिक्षण प्रणाली
- [मुख्य पेपर गाइड](../papers) — सैद्धांतिक आधार
