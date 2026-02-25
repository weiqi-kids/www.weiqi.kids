---
sidebar_position: 12
title: بناء ذكاء اصطناعي للغو من الصفر
description: دليل خطوة بخطوة لبناء ذكاء اصطناعي للغو على طراز AlphaGo Zero المبسط
---

# بناء ذكاء اصطناعي للغو من الصفر

يرشدك هذا المقال خطوة بخطوة لبناء ذكاء اصطناعي للغو على طراز AlphaGo Zero المبسط، يشمل منطق اللعبة والشبكة العصبية وMCTS وعملية التدريب.

:::info أهداف التعلم
بعد إكمال هذا الدليل، سيكون لديك ذكاء اصطناعي للغو قادر على:
- اللعب الذاتي على لوحة 9×9
- التحسن المستمر من خلال التعلم المعزز
- الوصول لمستوى هاوٍ مبتدئ
:::

---

## هيكل المشروع

```
mini-alphago/
├── game/
│   ├── __init__.py
│   ├── board.py          # منطق اللوحة
│   ├── rules.py          # تنفيذ القواعد
│   └── state.py          # حالة اللعبة
├── model/
│   ├── __init__.py
│   ├── network.py        # الشبكة العصبية
│   └── features.py       # ترميز الميزات
├── mcts/
│   ├── __init__.py
│   ├── node.py           # عقدة MCTS
│   └── search.py         # بحث MCTS
├── training/
│   ├── __init__.py
│   ├── self_play.py      # اللعب الذاتي
│   └── trainer.py        # المدرب
├── main.py               # البرنامج الرئيسي
└── requirements.txt
```

---

## الخطوة الأولى: اللوحة والقواعد

### تنفيذ اللوحة

```python
# game/board.py
import numpy as np

class Board:
    """لوحة الغو"""

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
        """نسخ اللوحة"""
        new_board = Board(self.size)
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.ko_point = self.ko_point
        new_board.history = self.history.copy()
        return new_board

    def get_opponent(self, player):
        """الحصول على الخصم"""
        return self.WHITE if player == self.BLACK else self.BLACK

    def is_on_board(self, x, y):
        """التحقق من أنه على اللوحة"""
        return 0 <= x < self.size and 0 <= y < self.size

    def get_neighbors(self, x, y):
        """الحصول على النقاط المجاورة"""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if self.is_on_board(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def get_group(self, x, y):
        """الحصول على السلسلة (الأحجار المتصلة من نفس اللون)"""
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
        """حساب عدد الحريات"""
        _, liberties = self.get_group(x, y)
        return len(liberties)

    def remove_group(self, group):
        """إزالة السلسلة"""
        for x, y in group:
            self.board[x, y] = self.EMPTY

    def is_legal(self, x, y, player=None):
        """التحقق من أنها حركة قانونية"""
        if player is None:
            player = self.current_player

        # التحقق من أنها نقطة فارغة
        if self.board[x, y] != self.EMPTY:
            return False

        # التحقق من الكو
        if self.ko_point == (x, y):
            return False

        # محاكاة الحركة
        test_board = self.copy()
        test_board.board[x, y] = player

        # التحقق أولاً من إمكانية الأسر
        opponent = self.get_opponent(player)
        captured = []
        for nx, ny in self.get_neighbors(x, y):
            if test_board.board[nx, ny] == opponent:
                group, liberties = test_board.get_group(nx, ny)
                if len(liberties) == 0:
                    captured.extend(group)

        if captured:
            return True

        # التحقق من الانتحار
        _, liberties = test_board.get_group(x, y)
        if len(liberties) == 0:
            return False

        return True

    def play(self, x, y):
        """وضع حجر"""
        if not self.is_legal(x, y):
            return False

        player = self.current_player
        opponent = self.get_opponent(player)

        # وضع الحجر
        self.board[x, y] = player

        # الأسر
        captured = []
        for nx, ny in self.get_neighbors(x, y):
            if self.board[nx, ny] == opponent:
                group, liberties = self.get_group(nx, ny)
                if len(liberties) == 0:
                    captured.extend(group)
                    self.remove_group(group)

        # إعداد الكو
        if len(captured) == 1:
            cx, cy = list(captured)[0]
            _, my_liberties = self.get_group(x, y)
            if len(my_liberties) == 1:
                self.ko_point = (cx, cy)
            else:
                self.ko_point = None
        else:
            self.ko_point = None

        # تسجيل التاريخ
        self.history.append((x, y, player))

        # تبديل اللاعب
        self.current_player = opponent

        return True

    def pass_move(self):
        """التمرير (pass)"""
        self.history.append((-1, -1, self.current_player))
        self.current_player = self.get_opponent(self.current_player)
        self.ko_point = None

    def is_game_over(self):
        """التحقق من انتهاء اللعبة"""
        if len(self.history) < 2:
            return False
        # كلا الطرفين يمرران متتاليين
        return (self.history[-1][0] == -1 and
                self.history[-2][0] == -1)

    def get_legal_moves(self):
        """الحصول على جميع الحركات القانونية"""
        moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.is_legal(x, y):
                    moves.append((x, y))
        moves.append((-1, -1))  # pass
        return moves

    def score(self):
        """حساب النتيجة (عد الأحجار المبسط)"""
        black_score = np.sum(self.board == self.BLACK)
        white_score = np.sum(self.board == self.WHITE)

        # حساب المنطقة المبسط
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

## الخطوة الثانية: ترميز الميزات

### ميزات الإدخال

```python
# model/features.py
import numpy as np

def encode_board(board):
    """
    ترميز اللوحة كمدخل للشبكة العصبية

    مستويات الميزات:
    0: أحجارنا
    1: أحجار الخصم
    2: النقاط الفارغة
    3: موقع الحركة الأخيرة
    4: موقع الحركة قبل الأخيرة
    5: مواقع الحركات القانونية
    6: دور الأسود (كلها 1 أو كلها 0)
    """
    size = board.size
    features = np.zeros((7, size, size), dtype=np.float32)

    current = board.current_player
    opponent = board.get_opponent(current)

    # مواقع الأحجار الأساسية
    features[0] = (board.board == current).astype(np.float32)
    features[1] = (board.board == opponent).astype(np.float32)
    features[2] = (board.board == board.EMPTY).astype(np.float32)

    # أحدث الحركات
    if len(board.history) >= 1:
        x, y, _ = board.history[-1]
        if x >= 0:
            features[3, x, y] = 1.0

    if len(board.history) >= 2:
        x, y, _ = board.history[-2]
        if x >= 0:
            features[4, x, y] = 1.0

    # الحركات القانونية
    for x in range(size):
        for y in range(size):
            if board.is_legal(x, y):
                features[5, x, y] = 1.0

    # من يلعب
    if current == board.BLACK:
        features[6] = np.ones((size, size), dtype=np.float32)

    return features
```

---

## الخطوة الثالثة: الشبكة العصبية

### بنية الشبكة برأسين

```python
# model/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """كتلة متبقية"""

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
    """شبكة السياسة-القيمة برأسين"""

    def __init__(self, board_size=9, input_channels=7, num_filters=64, num_blocks=4):
        super().__init__()
        self.board_size = board_size

        # الالتفاف الأولي
        self.conv_input = nn.Conv2d(input_channels, num_filters, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)

        # الكتل المتبقية
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_blocks)
        ])

        # رأس السياسة
        self.policy_conv = nn.Conv2d(num_filters, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size + 1)

        # رأس القيمة
        self.value_conv = nn.Conv2d(num_filters, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # الجذع المشترك
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.residual_blocks:
            x = block(x)

        # رأس السياسة
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)

        # رأس القيمة
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value


def create_network(board_size=9):
    """إنشاء الشبكة"""
    return PolicyValueNetwork(
        board_size=board_size,
        input_channels=7,
        num_filters=64,
        num_blocks=4
    )
```

---

## الخطوة الرابعة: تنفيذ MCTS

### فئة العقدة

```python
# mcts/node.py
import numpy as np

class MCTSNode:
    """عقدة MCTS"""

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
        """توسيع العقدة"""
        for move in legal_moves:
            if move not in self.children:
                idx = move[0] * 9 + move[1] if move[0] >= 0 else 81
                self.children[move] = MCTSNode(prior=np.exp(policy[idx]))

    def select_child(self, c_puct=1.5):
        """اختيار العقدة الفرعية باستخدام PUCT"""
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

### تنفيذ البحث

```python
# mcts/search.py
import numpy as np
import torch
from .node import MCTSNode

class MCTS:
    """بحث شجرة مونت كارلو"""

    def __init__(self, network, board_size=9, num_simulations=100, c_puct=1.5):
        self.network = network
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, board, add_noise=False):
        """تنفيذ بحث MCTS"""
        root = MCTSNode()

        # تقييم عقدة الجذر
        policy, value = self.evaluate(board)
        legal_moves = board.get_legal_moves()
        root.expand(policy, legal_moves)

        # إضافة ضوضاء Dirichlet (أثناء التدريب)
        if add_noise:
            self.add_dirichlet_noise(root)

        # تنفيذ المحاكاة
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

            # حساب القيمة من منظور نقطة بداية البحث
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
        """التقييم بالشبكة العصبية"""
        from model.features import encode_board

        features = encode_board(board)
        features = torch.tensor(features).unsqueeze(0)

        self.network.eval()
        with torch.no_grad():
            policy, value = self.network(features)

        return policy[0].numpy(), value[0].item()

    def add_dirichlet_noise(self, root, alpha=0.3, epsilon=0.25):
        """إضافة ضوضاء استكشاف"""
        noise = np.random.dirichlet([alpha] * len(root.children))
        for i, child in enumerate(root.children.values()):
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]

    def get_policy(self, root, temperature=1.0):
        """الحصول على السياسة من نتائج البحث"""
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
        """اختيار الحركة"""
        policy = self.get_policy(root, temperature)
        idx = np.random.choice(len(policy), p=policy)

        if idx == self.board_size ** 2:
            return (-1, -1)
        else:
            return (idx // self.board_size, idx % self.board_size)
```

---

## الخطوة الخامسة: اللعب الذاتي

```python
# training/self_play.py
import numpy as np
from game.board import Board
from model.features import encode_board

def self_play_game(mcts, temperature=1.0, temp_threshold=30):
    """تنفيذ مباراة لعب ذاتي"""
    board = Board(size=9)
    game_history = []

    move_count = 0
    while not board.is_game_over() and move_count < 200:
        # بحث MCTS
        root = mcts.search(board, add_noise=True)

        # الحصول على السياسة
        temp = temperature if move_count < temp_threshold else 0.0
        policy = mcts.get_policy(root, temp)

        # تسجيل بيانات التدريب
        features = encode_board(board)
        game_history.append({
            'features': features,
            'policy': policy,
            'player': board.current_player
        })

        # اختيار وتنفيذ الحركة
        move = mcts.select_move(root, temp)
        if move[0] >= 0:
            board.play(move[0], move[1])
        else:
            board.pass_move()

        move_count += 1

    # حساب الفائز
    score = board.score()
    winner = Board.BLACK if score > 0 else (Board.WHITE if score < 0 else 0)

    # تحديد القيم
    for data in game_history:
        if winner == 0:
            data['value'] = 0.0
        elif data['player'] == winner:
            data['value'] = 1.0
        else:
            data['value'] = -1.0

    return game_history


def generate_training_data(mcts, num_games=100):
    """توليد بيانات التدريب"""
    all_data = []

    for i in range(num_games):
        print(f"مباراة لعب ذاتي {i+1}/{num_games}")
        game_data = self_play_game(mcts)
        all_data.extend(game_data)

    return all_data
```

---

## الخطوة السادسة: المدرب

```python
# training/trainer.py
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class Trainer:
    """المدرب"""

    def __init__(self, network, learning_rate=0.001):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    def train_step(self, batch):
        """خطوة تدريب واحدة"""
        features, target_policy, target_value = batch

        self.network.train()
        self.optimizer.zero_grad()

        # التمرير الأمامي
        pred_policy, pred_value = self.network(features)

        # حساب الخسارة
        policy_loss = F.kl_div(pred_policy, target_policy, reduction='batchmean')
        value_loss = F.mse_loss(pred_value.squeeze(), target_value)
        total_loss = policy_loss + value_loss

        # التمرير العكسي
        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }

    def train_epoch(self, data, batch_size=32):
        """تدريب epoch واحد"""
        # إعداد البيانات
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
        """حفظ النموذج"""
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        """تحميل النموذج"""
        self.network.load_state_dict(torch.load(path))
```

---

## الخطوة السابعة: البرنامج الرئيسي

```python
# main.py
from model.network import create_network
from mcts.search import MCTS
from training.self_play import generate_training_data
from training.trainer import Trainer

def main():
    # إنشاء الشبكة
    network = create_network(board_size=9)
    mcts = MCTS(network, board_size=9, num_simulations=100)
    trainer = Trainer(network)

    # دورة التدريب
    num_iterations = 100
    games_per_iteration = 50
    epochs_per_iteration = 10

    for iteration in range(num_iterations):
        print(f"\n=== التكرار {iteration + 1}/{num_iterations} ===")

        # اللعب الذاتي
        print("توليد مباريات اللعب الذاتي...")
        training_data = generate_training_data(mcts, num_games=games_per_iteration)

        # التدريب
        print("التدريب...")
        for epoch in range(epochs_per_iteration):
            loss = trainer.train_epoch(training_data)
            print(f"  Epoch {epoch + 1}: loss = {loss:.4f}")

        # الحفظ
        trainer.save(f"model_iter_{iteration + 1}.pt")

    print("\nاكتمل التدريب!")


if __name__ == "__main__":
    main()
```

---

## التشغيل والاختبار

### تثبيت التبعيات

```bash
pip install torch numpy
```

### تشغيل التدريب

```bash
python main.py
```

### المخرجات المتوقعة

```
=== التكرار 1/100 ===
توليد مباريات اللعب الذاتي...
مباراة لعب ذاتي 1/50
مباراة لعب ذاتي 2/50
...
التدريب...
  Epoch 1: loss = 2.3456
  Epoch 2: loss = 1.8765
  ...
```

---

## اقتراحات التحسين

### تحسينات قصيرة المدى

| عنصر التحسين | الوصف |
|--------------|-------|
| زيادة الكتل المتبقية | 4 → 8 → 16 كتلة |
| زيادة عدد القنوات | 64 → 128 → 256 |
| زيادة عدد المحاكاة | 100 → 400 → 800 |
| مجموعة تدريب أكبر | 50 → 200 → 1000 مباراة/تكرار |

### تحسينات طويلة المدى

- دعم لوحة 19×19
- إضافة أهداف تدريب مساعدة (توقع المنطقة)
- تنفيذ لعب ذاتي متوازي
- إضافة تسريع GPU

---

## قراءات إضافية

- [شرح بنية الشبكة العصبية](../neural-network) — تصميم شبكة أعمق
- [تفاصيل تنفيذ MCTS](../mcts-implementation) — تقنيات بحث متقدمة
- [تحليل آلية تدريب KataGo](../training) — نظام تدريب بمستوى إنتاجي
- [دليل قراءة الأوراق الرئيسية](../papers) — الأساس النظري
