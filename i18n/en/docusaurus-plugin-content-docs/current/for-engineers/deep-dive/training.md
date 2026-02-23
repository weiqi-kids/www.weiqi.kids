---
sidebar_position: 3
title: KataGo Training Mechanism
description: Deep understanding of KataGo's self-play training process and core techniques
---

# KataGo Training Mechanism

This article provides an in-depth analysis of KataGo's training mechanism, helping you understand how self-play training works.

---

## Training Overview

### Training Loop

```
Initial model → Self-play → Collect data → Train update → Stronger model → Repeat
```

**Animation correspondence**:
- E5 Self-play ↔ Fixed point convergence
- E6 Strength curve ↔ S-curve growth
- H1 MDP ↔ Markov chain

### Hardware Requirements

| Model Scale | GPU Memory | Training Time |
|-------------|-----------|---------------|
| b6c96 | 4 GB | Several hours |
| b10c128 | 8 GB | 1-2 days |
| b18c384 | 16 GB | 1-2 weeks |
| b40c256 | 24 GB+ | Several weeks |

---

## Environment Setup

### Install Dependencies

```bash
# Python environment
conda create -n katago python=3.10
conda activate katago

# PyTorch (CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Other dependencies
pip install numpy h5py tqdm tensorboard
```

### Get Training Code

```bash
git clone https://github.com/lightvector/KataGo.git
cd KataGo/python
```

---

## Training Configuration

### Config File Structure

```yaml
# configs/train_config.yaml

# Model architecture
model:
  num_blocks: 10          # Number of residual blocks
  trunk_channels: 128     # Trunk channel count
  policy_channels: 32     # Policy head channels
  value_channels: 32      # Value head channels

# Training parameters
training:
  batch_size: 256
  learning_rate: 0.001
  lr_schedule: "cosine"
  weight_decay: 0.0001
  epochs: 100

# Self-play parameters
selfplay:
  num_games_per_iteration: 1000
  max_visits: 600
  temperature: 1.0
  temperature_drop_move: 20

# Data settings
data:
  max_history_games: 500000
  shuffle_buffer_size: 100000
```

### Model Scale Reference

| Name | num_blocks | trunk_channels | Parameters |
|------|-----------|----------------|------------|
| b6c96 | 6 | 96 | ~1M |
| b10c128 | 10 | 128 | ~3M |
| b18c384 | 18 | 384 | ~20M |
| b40c256 | 40 | 256 | ~45M |

**Animation correspondence**:
- F2 Network size vs strength: Capacity scaling
- F6 Neural scaling laws: Log-log relationship

---

## Training Process

### Step 1: Initialize Model

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
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Step 2: Self-Play Data Generation

```bash
# Compile C++ engine
cd ../cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=CUDA
make -j$(nproc)

# Run self-play
./katago selfplay \
  -model ../python/model_init.pt \
  -output-dir ../python/selfplay_data \
  -config selfplay.cfg \
  -num-games 1000
```

Self-play configuration (selfplay.cfg):

```ini
maxVisits = 600
numSearchThreads = 4

# Temperature settings (increase exploration)
chosenMoveTemperature = 1.0
chosenMoveTemperatureEarly = 1.0
chosenMoveTemperatureHalflife = 20

# Dirichlet noise (increase diversity)
rootNoiseEnabled = true
rootDirichletNoiseTotalConcentration = 10.83
rootDirichletNoiseWeight = 0.25
```

**Animation correspondence**:
- C3 Exploration vs exploitation: Temperature parameter
- E10 Dirichlet noise: Root exploration

### Step 3: Train Neural Network

```python
# train.py
import torch
from torch.utils.data import DataLoader
from model import KataGoModel
from dataset import SelfPlayDataset

# Load data
dataset = SelfPlayDataset('selfplay_data/')
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# Load model
model = KataGoModel(config)
model.load_state_dict(torch.load('model_init.pt'))
model = model.cuda()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=0.00001
)

# Training loop
for epoch in range(100):
    model.train()
    total_loss = 0

    for batch in dataloader:
        inputs = batch['inputs'].cuda()
        policy_target = batch['policy'].cuda()
        value_target = batch['value'].cuda()
        ownership_target = batch['ownership'].cuda()

        # Forward pass
        policy_pred, value_pred, ownership_pred = model(inputs)

        # Compute loss
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

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), f'model_epoch{epoch}.pt')
```

**Animation correspondence**:
- D5 Gradient descent: optimizer.step()
- K2 Momentum: Adam optimizer
- K4 Learning rate decay: CosineAnnealingLR
- K5 Gradient clipping: clip_grad_norm_

### Step 4: Evaluation & Iteration

```bash
# Evaluate new model vs old model
./katago match \
  -model1 model_epoch99.pt \
  -model2 model_init.pt \
  -num-games 100 \
  -output match_results.txt
```

If new model win rate > 55%, replace old model and proceed to next iteration.

---

## Loss Functions Explained

### Policy Loss

```python
# Cross-entropy loss
policy_loss = -sum(target * log(pred))
```

Goal: Make predicted probability distribution close to MCTS search results.

**Animation correspondence**:
- J1 Policy entropy: Cross-entropy
- J2 KL divergence: Distribution distance

### Value Loss

```python
# Mean squared error
value_loss = (pred - actual_result)^2
```

Goal: Predict final game result (win/loss/draw).

### Ownership Loss

```python
# Per-point ownership prediction
ownership_loss = mean((pred - actual_ownership)^2)
```

Goal: Predict final ownership of each position.

---

## Advanced Techniques

### 1. Data Augmentation

Leverage board symmetry:

```python
def augment_data(board, policy, ownership):
    """Data augmentation using D4 group's 8 transformations"""
    augmented = []

    for rotation in range(4):
        for flip in [False, True]:
            # Rotation and flip
            aug_board = transform(board, rotation, flip)
            aug_policy = transform(policy, rotation, flip)
            aug_ownership = transform(ownership, rotation, flip)
            augmented.append((aug_board, aug_policy, aug_ownership))

    return augmented
```

**Animation correspondence**:
- A9 Board symmetry: D4 group
- L4 Data augmentation: Symmetry exploitation

### 2. Curriculum Learning

From simple to complex:

```python
# Train with fewer search visits first
schedule = [
    (100, 10000),   # 100 visits, 10000 games
    (200, 20000),   # 200 visits, 20000 games
    (400, 50000),   # 400 visits, 50000 games
    (600, 100000),  # 600 visits, 100000 games
]
```

**Animation correspondence**:
- E12 Training curriculum: Curriculum learning

### 3. Mixed Precision Training

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

### 4. Multi-GPU Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Initialize distributed
dist.init_process_group(backend='nccl')

# Wrap model
model = DistributedDataParallel(model)
```

---

## Monitoring & Debugging

### TensorBoard Monitoring

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/training')

# Log losses
writer.add_scalar('Loss/policy', policy_loss, step)
writer.add_scalar('Loss/value', value_loss, step)
writer.add_scalar('Loss/total', total_loss, step)

# Log learning rate
writer.add_scalar('LR', scheduler.get_last_lr()[0], step)
```

```bash
tensorboard --logdir runs
```

### Common Issues

| Issue | Possible Cause | Solution |
|-------|----------------|----------|
| Loss not decreasing | Learning rate too low/high | Adjust learning rate |
| Loss oscillating | Batch size too small | Increase batch size |
| Overfitting | Insufficient data | Generate more self-play data |
| Strength not improving | Too few search visits | Increase maxVisits |

**Animation correspondence**:
- L1 Overfitting: Over-adaptation
- L2 Regularization: weight_decay
- D6 Learning rate effects: Tuning

---

## Small-Scale Experiment Suggestions

If you just want to experiment, we recommend:

1. **Use 9×9 board**: Dramatically reduces computation
2. **Use small model**: b6c96 is enough for experiments
3. **Reduce search visits**: 100-200 visits
4. **Fine-tune pretrained model**: Faster than training from scratch

```bash
# 9×9 board settings
boardSize = 9
maxVisits = 100
```

---

## Further Reading

- [Source Code Guide](../source-code) — Understanding code structure
- [Contributing to Open Source](../contributing) — Join distributed training
- [KataGo's Key Innovations](../../how-it-works/katago-innovations) — The secret to 50x efficiency
