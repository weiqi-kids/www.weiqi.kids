---
sidebar_position: 5
title: MCTS 实现细节
description: 深入解析蒙特卡洛树搜索的实现、PUCT 选择与并行化技术
---

# MCTS 实现细节

本文深入解析 KataGo 中蒙特卡洛树搜索（MCTS）的实现细节，包括数据结构、选择策略与并行化技术。

---

## MCTS 四步骤回顾

```
┌─────────────────────────────────────────────────────┐
│                    MCTS 搜索循环                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│   1. Selection      选择：沿树向下，用 PUCT 选节点  │
│         │                                           │
│         ▼                                           │
│   2. Expansion      展开：到达叶节点，创建子节点    │
│         │                                           │
│         ▼                                           │
│   3. Evaluation     评估：用神经网络评估叶节点      │
│         │                                           │
│         ▼                                           │
│   4. Backprop       回传：更新路径上所有节点统计    │
│                                                     │
│   重复数千次，选择访问次数最多的动作                │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 节点数据结构

### 核心数据

每个 MCTS 节点需要存储：

```python
class MCTSNode:
    def __init__(self, state, parent=None, prior=0.0):
        # 基本信息
        self.state = state              # 棋盘状态
        self.parent = parent            # 父节点
        self.children = {}              # 子节点字典 {action: node}
        self.action = None              # 到达此节点的动作

        # 统计信息
        self.visit_count = 0            # N(s)：访问次数
        self.value_sum = 0.0            # W(s)：价值总和
        self.prior = prior              # P(s,a)：先验概率

        # 并行搜索用
        self.virtual_loss = 0           # 虚拟损失
        self.is_expanded = False        # 是否已展开

    @property
    def value(self):
        """Q(s) = W(s) / N(s)"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
```

### 内存优化

KataGo 使用多种技术减少内存用量：

```python
# 使用 numpy 数组而非 Python dict
class OptimizedNode:
    __slots__ = ['visit_count', 'value_sum', 'prior', 'children_indices']

    def __init__(self):
        self.visit_count = np.int32(0)
        self.value_sum = np.float32(0.0)
        self.prior = np.float32(0.0)
        self.children_indices = None  # 延迟分配
```

---

## Selection：PUCT 选择

### PUCT 公式

```
选择分数 = Q(s,a) + U(s,a)

其中：
Q(s,a) = W(s,a) / N(s,a)              # 平均价值
U(s,a) = c_puct × P(s,a) × √(N(s)) / (1 + N(s,a))  # 探索项
```

### 参数说明

| 符号 | 意义 | 典型值 |
|------|------|--------|
| Q(s,a) | 动作 a 的平均价值 | [-1, +1] |
| P(s,a) | 神经网络的先验概率 | [0, 1] |
| N(s) | 父节点访问次数 | 整数 |
| N(s,a) | 动作 a 的访问次数 | 整数 |
| c_puct | 探索常数 | 1.0 ~ 2.5 |

### 实现

```python
def select_child(self, c_puct=1.5):
    """选择 PUCT 分数最高的子节点"""
    best_score = -float('inf')
    best_action = None
    best_child = None

    # 父节点访问次数的平方根
    sqrt_parent_visits = math.sqrt(self.visit_count)

    for action, child in self.children.items():
        # Q 值（平均价值）
        if child.visit_count > 0:
            q_value = child.value_sum / child.visit_count
        else:
            q_value = 0.0

        # U 值（探索项）
        u_value = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)

        # 总分数
        score = q_value + u_value

        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    return best_action, best_child
```

### 探索 vs 利用的平衡

```
初期：N(s,a) 小
├── U(s,a) 大 → 探索为主
└── 高先验概率的动作优先被探索

后期：N(s,a) 大
├── U(s,a) 小 → 利用为主
└── Q(s,a) 主导，选择已知好的动作
```

---

## Expansion：节点展开

### 展开条件

到达叶节点时，使用神经网络展开：

```python
def expand(self, policy_probs, legal_moves):
    """展开节点，创建所有合法动作的子节点"""
    for action in legal_moves:
        if action not in self.children:
            prior = policy_probs[action]  # 神经网络预测的概率
            child_state = self.state.play(action)
            self.children[action] = MCTSNode(
                state=child_state,
                parent=self,
                prior=prior
            )

    self.is_expanded = True
```

### 合法动作过滤

```python
def get_legal_moves(state):
    """获取所有合法动作"""
    legal = []
    for i in range(361):
        x, y = i // 19, i % 19
        if state.is_legal(x, y):
            legal.append(i)

    # 加入 pass
    legal.append(361)

    return legal
```

---

## Evaluation：神经网络评估

### 单次评估

```python
def evaluate(self, state):
    """使用神经网络评估局面"""
    # 编码输入特征
    features = encode_state(state)  # (22, 19, 19)
    features = torch.tensor(features).unsqueeze(0)  # (1, 22, 19, 19)

    # 神经网络推理
    with torch.no_grad():
        output = self.network(features)

    policy = output['policy'][0].numpy()  # (362,)
    value = output['value'][0].item()     # scalar

    return policy, value
```

### 批量评估（关键优化）

GPU 在批量推理时效率最高：

```python
class BatchedEvaluator:
    def __init__(self, network, batch_size=8):
        self.network = network
        self.batch_size = batch_size
        self.pending = []  # 待评估的 (state, callback) 列表

    def request_evaluation(self, state, callback):
        """请求评估，当批量满时自动执行"""
        self.pending.append((state, callback))

        if len(self.pending) >= self.batch_size:
            self.flush()

    def flush(self):
        """执行批量评估"""
        if not self.pending:
            return

        # 准备批量输入
        states = [s for s, _ in self.pending]
        features = torch.stack([encode_state(s) for s in states])

        # 批量推理
        with torch.no_grad():
            outputs = self.network(features)

        # 回调结果
        for i, (_, callback) in enumerate(self.pending):
            policy = outputs['policy'][i].numpy()
            value = outputs['value'][i].item()
            callback(policy, value)

        self.pending.clear()
```

---

## Backpropagation：回传更新

### 基本回传

```python
def backpropagate(self, value):
    """从叶节点回传到根节点，更新统计信息"""
    node = self

    while node is not None:
        node.visit_count += 1
        node.value_sum += value

        # 交替视角：对手的价值是相反的
        value = -value

        node = node.parent
```

### 视角交替的重要性

```
黑方视角：value = +0.6（黑方有利）

回传路径：
叶节点（黑方走）: value_sum += +0.6
    ↑
父节点（白方走）: value_sum += -0.6  ← 对白方来说是不利的
    ↑
祖父节点（黑方走）: value_sum += +0.6
    ↑
...
```

---

## 并行化：虚拟损失

### 问题

多线程同时搜索时，可能都选到同一个节点：

```
Thread 1: 选择节点 A（Q=0.6, N=100）
Thread 2: 选择节点 A（Q=0.6, N=100）← 重复！
Thread 3: 选择节点 A（Q=0.6, N=100）← 重复！
```

### 解决方案：虚拟损失

选择节点时，先加上"虚拟损失"，让其他线程不想选它：

```python
VIRTUAL_LOSS = 3  # 虚拟损失值

def select_with_virtual_loss(self):
    """带虚拟损失的选择"""
    action, child = self.select_child()

    # 加上虚拟损失
    child.visit_count += VIRTUAL_LOSS
    child.value_sum -= VIRTUAL_LOSS  # 假装输了

    return action, child

def backpropagate_with_virtual_loss(self, value):
    """回传时移除虚拟损失"""
    node = self

    while node is not None:
        # 移除虚拟损失
        node.visit_count -= VIRTUAL_LOSS
        node.value_sum += VIRTUAL_LOSS

        # 正常更新
        node.visit_count += 1
        node.value_sum += value

        value = -value
        node = node.parent
```

### 效果

```
Thread 1: 选择节点 A，加虚拟损失
         A 的 Q 值暂时下降

Thread 2: 选择节点 B（因为 A 看起来变差了）

Thread 3: 选择节点 C

→ 不同线程探索不同分支，提高效率
```

---

## 完整搜索实现

```python
class MCTS:
    def __init__(self, network, c_puct=1.5, num_simulations=800):
        self.network = network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.evaluator = BatchedEvaluator(network)

    def search(self, root_state):
        """执行 MCTS 搜索"""
        root = MCTSNode(root_state)

        # 展开根节点
        policy, value = self.evaluate(root_state)
        legal_moves = get_legal_moves(root_state)
        root.expand(policy, legal_moves)

        # 执行模拟
        for _ in range(self.num_simulations):
            node = root
            path = [node]

            # Selection：沿树向下
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

        # 选择访问次数最多的动作
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

## 进阶技术

### Dirichlet 噪声

训练时在根节点加入噪声，增加探索：

```python
def add_dirichlet_noise(root, alpha=0.03, epsilon=0.25):
    """在根节点加入 Dirichlet 噪声"""
    noise = np.random.dirichlet([alpha] * len(root.children))

    for i, child in enumerate(root.children.values()):
        child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]
```

### 温度参数

控制动作选择的随机性：

```python
def select_action_with_temperature(root, temperature=1.0):
    """根据访问次数和温度选择动作"""
    visits = np.array([c.visit_count for c in root.children.values()])
    actions = list(root.children.keys())

    if temperature == 0:
        # 贪婪选择
        return actions[np.argmax(visits)]
    else:
        # 根据访问次数的概率分布选择
        probs = visits ** (1 / temperature)
        probs = probs / probs.sum()
        return np.random.choice(actions, p=probs)
```

### 树重用

新的一步可以重用之前的搜索树：

```python
def reuse_tree(root, action):
    """重用子树"""
    if action in root.children:
        new_root = root.children[action]
        new_root.parent = None
        return new_root
    else:
        return None  # 需要创建新树
```

---

## 性能优化总结

| 技术 | 效果 |
|------|------|
| **批量评估** | GPU 利用率从 10% → 80%+ |
| **虚拟损失** | 多线程效率提升 3-5x |
| **树重用** | 减少冷启动，节省 30%+ 计算 |
| **内存池** | 减少内存分配开销 |

---

## 延伸阅读

- [神经网络架构详解](../neural-network) — 评估函数的来源
- [GPU 后端与优化](../gpu-optimization) — 批量推理的硬件优化
- [关键论文导读](../papers) — PUCT 公式的理论基础
