---
sidebar_position: 3
title: KataGo 训练机制解析
description: 深入理解 KataGo 的自我对弈训练流程与核心技术
---

# KataGo 训练机制解析

本文深入解析 KataGo 的训练机制，帮助你理解自我对弈训练的运作原理。

---

## 训练概述

### 训练循环

```
初始模型 → 自我对弈 → 收集数据 → 训练更新 → 更强模型 → 重复
```

**动画对应**：
- E5 自我对弈 ↔ 不动点收敛
- E6 棋力曲线 ↔ S 曲线成长
- H1 MDP ↔ 马尔可夫链

### 硬件需求

| 模型规模 | GPU 显存 | 训练时间 |
|---------|-----------|---------|
| b6c96 | 4 GB | 数小时 |
| b10c128 | 8 GB | 1-2 天 |
| b18c384 | 16 GB | 1-2 周 |
| b40c256 | 24 GB+ | 数周 |

---

## 环境设置

### 安装依赖

```bash
# Python 环境
conda create -n katago python=3.10
conda activate katago

# PyTorch（CUDA 版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 其他依赖
pip install numpy h5py tqdm tensorboard
```

### 获取训练代码

```bash
git clone https://github.com/lightvector/KataGo.git
cd KataGo/python
```

---

## 训练配置

### 配置文件结构

```yaml
# configs/train_config.yaml

# 模型架构
model:
  num_blocks: 10          # 残差块数量
  trunk_channels: 128     # 主干通道数
  policy_channels: 32     # Policy 头通道数
  value_channels: 32      # Value 头通道数

# 训练参数
training:
  batch_size: 256
  learning_rate: 0.001
  lr_schedule: "cosine"
  weight_decay: 0.0001
  epochs: 100

# 自我对弈参数
selfplay:
  num_games_per_iteration: 1000
  max_visits: 600
  temperature: 1.0
  temperature_drop_move: 20

# 数据配置
data:
  max_history_games: 500000
  shuffle_buffer_size: 100000
```

### 模型规模对照

| 名称 | num_blocks | trunk_channels | 参数量 |
|------|-----------|----------------|--------|
| b6c96 | 6 | 96 | ~1M |
| b10c128 | 10 | 128 | ~3M |
| b18c384 | 18 | 384 | ~20M |
| b40c256 | 40 | 256 | ~45M |

**动画对应**：
- F2 网络大小 vs 棋力：容量缩放
- F6 神经缩放律：双对数关系

---

## 训练流程

### 步骤 1：初始化模型

```python
# init_model.py
import torch
from model import KataGoModel

config = {
    'num_blocks': 10,
    'trunk_channels': 128,
    'input_features': 22,
    'policy_size': 362,  # 361 + pass
}

model = KataGoModel(config)
torch.save(model.state_dict(), 'model_init.pt')
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
```

### 步骤 2：自我对弈产生数据

```bash
# 编译 C++ 引擎
cd ../cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=CUDA
make -j$(nproc)

# 执行自我对弈
./katago selfplay \
  -model ../python/model_init.pt \
  -output-dir ../python/selfplay_data \
  -config selfplay.cfg \
  -num-games 1000
```

自我对弈配置（selfplay.cfg）：

```ini
maxVisits = 600
numSearchThreads = 4

# 温度配置（增加探索）
chosenMoveTemperature = 1.0
chosenMoveTemperatureEarly = 1.0
chosenMoveTemperatureHalflife = 20

# Dirichlet 噪声（增加多样性）
rootNoiseEnabled = true
rootDirichletNoiseTotalConcentration = 10.83
rootDirichletNoiseWeight = 0.25
```

**动画对应**：
- C3 探索 vs 利用：温度参数
- E10 Dirichlet 噪声：根节点探索

### 步骤 3：训练神经网络

```python
# train.py
import torch
from torch.utils.data import DataLoader
from model import KataGoModel
from dataset import SelfPlayDataset

# 加载数据
dataset = SelfPlayDataset('selfplay_data/')
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# 加载模型
model = KataGoModel(config)
model.load_state_dict(torch.load('model_init.pt'))
model = model.cuda()

# 优化器
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

# 学习率调度
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=0.00001
)

# 训练循环
for epoch in range(100):
    model.train()
    total_loss = 0

    for batch in dataloader:
        inputs = batch['inputs'].cuda()
        policy_target = batch['policy'].cuda()
        value_target = batch['value'].cuda()
        ownership_target = batch['ownership'].cuda()

        # 前向传播
        policy_pred, value_pred, ownership_pred = model(inputs)

        # 计算损失
        policy_loss = torch.nn.functional.cross_entropy(
            policy_pred, policy_target
        )
        value_loss = torch.nn.functional.mse_loss(
            value_pred, value_target
        )
        ownership_loss = torch.nn.functional.mse_loss(
            ownership_pred, ownership_target
        )

        loss = policy_loss + value_loss + 0.5 * ownership_loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")

    # 保存检查点
    torch.save(model.state_dict(), f'model_epoch{epoch}.pt')
```

**动画对应**：
- D5 梯度下降：optimizer.step()
- K2 动量：Adam 优化器
- K4 学习率衰减：CosineAnnealingLR
- K5 梯度裁剪：clip_grad_norm_

### 步骤 4：评估与迭代

```bash
# 评估新模型 vs 旧模型
./katago match \
  -model1 model_epoch99.pt \
  -model2 model_init.pt \
  -num-games 100 \
  -output match_results.txt
```

如果新模型胜率 > 55%，则替换旧模型，进入下一轮迭代。

---

## 损失函数详解

### Policy Loss

```python
# 交叉熵损失
policy_loss = -sum(target * log(pred))
```

目标：让预测的概率分布接近 MCTS 搜索结果。

**动画对应**：
- J1 策略熵：交叉熵
- J2 KL 散度：分布距离

### Value Loss

```python
# 均方误差
value_loss = (pred - actual_result)^2
```

目标：预测对局最终结果（胜/负/和）。

### Ownership Loss

```python
# 每点归属预测
ownership_loss = mean((pred - actual_ownership)^2)
```

目标：预测每个位置最终归属。

---

## 进阶技巧

### 1. 数据增强

利用棋盘的对称性：

```python
def augment_data(board, policy, ownership):
    """对 D4 群的 8 种变换进行数据增强"""
    augmented = []

    for rotation in range(4):
        for flip in [False, True]:
            # 旋转与翻转
            aug_board = transform(board, rotation, flip)
            aug_policy = transform(policy, rotation, flip)
            aug_ownership = transform(ownership, rotation, flip)
            augmented.append((aug_board, aug_policy, aug_ownership))

    return augmented
```

**动画对应**：
- A9 棋盘对称性：D4 群
- L4 数据增强：对称性利用

### 2. 课程学习

从简单到复杂：

```python
# 先用较少搜索次数训练
schedule = [
    (100, 10000),   # 100 visits, 10000 games
    (200, 20000),   # 200 visits, 20000 games
    (400, 50000),   # 400 visits, 50000 games
    (600, 100000),  # 600 visits, 100000 games
]
```

**动画对应**：
- E12 训练课程：课程学习

### 3. 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    policy_pred, value_pred, ownership_pred = model(inputs)
    loss = compute_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 4. 多 GPU 训练

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# 初始化分布式
dist.init_process_group(backend='nccl')

# 包装模型
model = DistributedDataParallel(model)
```

---

## 监控与调试

### TensorBoard 监控

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/training')

# 记录损失
writer.add_scalar('Loss/policy', policy_loss, step)
writer.add_scalar('Loss/value', value_loss, step)
writer.add_scalar('Loss/total', total_loss, step)

# 记录学习率
writer.add_scalar('LR', scheduler.get_last_lr()[0], step)
```

```bash
tensorboard --logdir runs
```

### 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 损失不下降 | 学习率太低/太高 | 调整学习率 |
| 损失震荡 | 批量大小太小 | 增加批量大小 |
| 过拟合 | 数据不足 | 产生更多自我对弈数据 |
| 棋力不增长 | 搜索次数太少 | 增加 maxVisits |

**动画对应**：
- L1 过拟合：过度适应
- L2 正则化：weight_decay
- D6 学习率效应：调参

---

## 小规模实验建议

如果你只是想实验，建议：

1. **使用 9×9 棋盘**：大幅减少计算量
2. **使用小型模型**：b6c96 足够实验
3. **减少搜索次数**：100-200 visits
4. **使用预训练模型微调**：比从零开始快

```bash
# 9×9 棋盘配置
boardSize = 9
maxVisits = 100
```

---

## 延伸阅读

- [源代码导读](../source-code) — 理解代码结构
- [参与开源社区](../contributing) — 加入分布式训练
- [KataGo 的关键创新](../../how-it-works/katago-innovations) — 50 倍效率的秘密
