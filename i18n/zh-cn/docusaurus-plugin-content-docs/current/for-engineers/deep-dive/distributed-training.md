---
sidebar_position: 7
title: 分布式训练架构
description: KataGo 分布式训练系统的架构、Self-play Worker 与模型发布流程
---

# 分布式训练架构

本文介绍 KataGo 的分布式训练系统架构，说明如何通过全球社区的算力持续改进模型。

---

## 系统架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                    KataGo 分布式训练系统                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│   │  Worker 1   │    │  Worker 2   │    │  Worker N   │   │
│   │  (Self-play)│    │  (Self-play)│    │  (Self-play)│   │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘   │
│          │                  │                  │           │
│          └────────────┬─────┴──────────────────┘           │
│                       │                                     │
│                       ▼                                     │
│            ┌─────────────────────┐                         │
│            │    训练服务器       │                         │
│            │  (Data Collection)  │                         │
│            └──────────┬──────────┘                         │
│                       │                                     │
│                       ▼                                     │
│            ┌─────────────────────┐                         │
│            │    训练流程         │                         │
│            │  (Model Training)   │                         │
│            └──────────┬──────────┘                         │
│                       │                                     │
│                       ▼                                     │
│            ┌─────────────────────┐                         │
│            │    新模型发布       │                         │
│            │  (Model Release)    │                         │
│            └─────────────────────┘                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Self-play Worker

### 工作流程

每个 Worker 执行以下循环：

```python
def self_play_worker():
    while True:
        # 1. 下载最新模型
        model = download_latest_model()

        # 2. 执行自我对弈
        games = []
        for _ in range(batch_size):
            game = play_game(model)
            games.append(game)

        # 3. 上传对局数据
        upload_games(games)

        # 4. 检查新模型
        if new_model_available():
            model = download_latest_model()
```

### 对局生成

```python
def play_game(model):
    """执行一局自我对弈"""
    game = Game()
    positions = []

    while not game.is_terminal():
        # MCTS 搜索
        mcts = MCTS(model, num_simulations=800)
        policy = mcts.get_policy(game.state)

        # 加入 Dirichlet 噪声（增加探索）
        if game.move_count < 30:
            policy = add_dirichlet_noise(policy)

        # 根据 policy 选择动作
        if game.move_count < 30:
            # 前 30 手用温度采样
            action = sample_with_temperature(policy, temp=1.0)
        else:
            # 之后贪婪选择
            action = np.argmax(policy)

        # 记录训练数据
        positions.append({
            'state': game.state.copy(),
            'policy': policy,
            'player': game.current_player
        })

        game.play(action)

    # 标记胜负
    winner = game.get_winner()
    for pos in positions:
        pos['value'] = 1.0 if pos['player'] == winner else -1.0

    return positions
```

### 数据格式

```json
{
  "version": 1,
  "rules": "chinese",
  "komi": 7.5,
  "board_size": 19,
  "positions": [
    {
      "move_number": 0,
      "board": "...",
      "policy": [0.01, 0.02, ...],
      "value": 1.0,
      "score": 2.5
    }
  ]
}
```

---

## 数据收集服务器

### 功能

1. **接收对局数据**：从 Workers 收集对局
2. **数据验证**：检查格式、过滤异常
3. **数据存储**：写入训练数据集
4. **统计监控**：追踪对局数量、Worker 状态

### 数据验证

```python
def validate_game(game_data):
    """验证对局数据"""
    checks = [
        len(game_data['positions']) > 10,  # 最少手数
        len(game_data['positions']) < 500,  # 最多手数
        all(is_valid_policy(p['policy']) for p in game_data['positions']),
        game_data['rules'] in SUPPORTED_RULES,
    ]
    return all(checks)
```

### 数据存储结构

```
training_data/
├── run_001/
│   ├── games_00001.npz
│   ├── games_00002.npz
│   └── ...
├── run_002/
│   └── ...
└── current/
    └── latest_games.npz
```

---

## 训练流程

### 训练循环

```python
def training_loop():
    model = load_model()
    optimizer = Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        # 加载最新的对局数据
        dataset = load_recent_games(num_games=100000)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        for batch in dataloader:
            states = batch['states']
            target_policies = batch['policies']
            target_values = batch['values']

            # 前向传播
            pred_policies, pred_values = model(states)

            # 计算损失
            policy_loss = cross_entropy(pred_policies, target_policies)
            value_loss = mse_loss(pred_values, target_values)
            loss = policy_loss + value_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 定期评估
        if epoch % 100 == 0:
            evaluate_model(model)
```

### 损失函数

KataGo 使用多个损失项：

```python
def compute_loss(predictions, targets):
    # Policy 损失（交叉熵）
    policy_loss = F.cross_entropy(
        predictions['policy'],
        targets['policy']
    )

    # Value 损失（MSE）
    value_loss = F.mse_loss(
        predictions['value'],
        targets['value']
    )

    # Score 损失（MSE）
    score_loss = F.mse_loss(
        predictions['score'],
        targets['score']
    )

    # Ownership 损失（MSE）
    ownership_loss = F.mse_loss(
        predictions['ownership'],
        targets['ownership']
    )

    # 加权总和
    total_loss = (
        1.0 * policy_loss +
        1.0 * value_loss +
        0.5 * score_loss +
        0.5 * ownership_loss
    )

    return total_loss
```

---

## 模型评估与发布

### Elo 评估

新模型需要与旧模型对战来评估棋力：

```python
def evaluate_new_model(new_model, baseline_model, num_games=400):
    """评估新模型的 Elo"""
    wins = 0
    losses = 0
    draws = 0

    for _ in range(num_games // 2):
        # 新模型执黑
        result = play_game(new_model, baseline_model)
        if result == 'black_wins':
            wins += 1
        elif result == 'white_wins':
            losses += 1
        else:
            draws += 1

        # 新模型执白
        result = play_game(baseline_model, new_model)
        if result == 'white_wins':
            wins += 1
        elif result == 'black_wins':
            losses += 1
        else:
            draws += 1

    # 计算 Elo 差距
    win_rate = (wins + 0.5 * draws) / num_games
    elo_diff = 400 * math.log10(win_rate / (1 - win_rate))

    return elo_diff
```

### 发布条件

```python
def should_release_model(new_model, current_best):
    """决定是否发布新模型"""
    elo_diff = evaluate_new_model(new_model, current_best)

    # 条件：Elo 提升超过阈值
    if elo_diff > 20:
        return True

    # 或：达到一定的训练步数
    if training_steps % 10000 == 0:
        return True

    return False
```

### 模型版本命名

```
kata1-b18c384nbt-s{steps}-d{data}.bin.gz

示例：
kata1-b18c384nbt-s9996604416-d4316597426.bin.gz
├── kata1: 训练系列
├── b18c384nbt: 架构（18 残差块、384 通道）
├── s9996604416: 训练步数
└── d4316597426: 训练数据量
```

---

## KataGo Training 参与指南

### 系统需求

| 项目 | 最低需求 | 建议需求 |
|------|---------|---------|
| GPU | GTX 1060 | RTX 3060+ |
| 显存 | 4 GB | 8 GB+ |
| 网络 | 10 Mbps | 50 Mbps+ |
| 运行时间 | 持续运行 | 24/7 |

### 安装 Worker

```bash
# 下载 Worker
wget https://katagotraining.org/download/worker

# 配置
./katago contribute -config contribute.cfg

# 开始贡献
./katago contribute
```

### 配置文件

```ini
# contribute.cfg

# 服务器配置
serverUrl = https://katagotraining.org/

# 用户名（用于统计）
username = your_username

# GPU 配置
numNNServerThreadsPerModel = 1
nnMaxBatchSize = 16

# 对局配置
gamesPerBatch = 25
```

### 监控贡献

```bash
# 查看统计
https://katagotraining.org/contributions/

# 本地日志
tail -f katago_contribute.log
```

---

## 训练统计

### KataGo 训练里程碑

| 时间 | 对局数 | Elo |
|------|--------|-----|
| 2019.06 | 10M | 初始 |
| 2020.01 | 100M | +500 |
| 2021.01 | 500M | +800 |
| 2022.01 | 1B | +1000 |
| 2024.01 | 5B+ | +1200 |

### 社区贡献者

- 数百位全球贡献者
- 累计数千 GPU 年算力
- 持续 24/7 运行

---

## 进阶主题

### 课程学习（Curriculum Learning）

逐步增加训练难度：

```python
def get_training_config(training_step):
    if training_step < 100000:
        return {'board_size': 9, 'visits': 200}
    elif training_step < 500000:
        return {'board_size': 13, 'visits': 400}
    else:
        return {'board_size': 19, 'visits': 800}
```

### 数据增强

利用棋盘对称性增加数据量：

```python
def augment_position(state, policy):
    """8 种对称变换"""
    augmented = []

    for rotation in [0, 90, 180, 270]:
        for flip in [False, True]:
            aug_state = transform(state, rotation, flip)
            aug_policy = transform_policy(policy, rotation, flip)
            augmented.append((aug_state, aug_policy))

    return augmented
```

---

## 延伸阅读

- [KataGo 训练机制解析](../training) — 训练流程详解
- [参与开源社区](../contributing) — 如何贡献代码
- [评估与基准测试](../evaluation) — 模型评估方法
