---
sidebar_position: 3
title: KataGo 訓練機制解析
description: 深入理解 KataGo 的自我對弈訓練流程與核心技術
---

# KataGo 訓練機制解析

本文深入解析 KataGo 的訓練機制，幫助你理解自我對弈訓練的運作原理。

---

## 訓練概述

### 訓練循環

```
初始模型 → 自我對弈 → 收集資料 → 訓練更新 → 更強模型 → 重複
```

**動畫對應**：
- 🎬 E5 自我對弈 ↔ 不動點收斂
- 🎬 E6 棋力曲線 ↔ S 曲線成長
- 🎬 H1 MDP ↔ 馬可夫鏈

### 硬體需求

| 模型規模 | GPU 記憶體 | 訓練時間 |
|---------|-----------|---------|
| b6c96 | 4 GB | 數小時 |
| b10c128 | 8 GB | 1-2 天 |
| b18c384 | 16 GB | 1-2 週 |
| b40c256 | 24 GB+ | 數週 |

---

## 環境設定

### 安裝依賴

```bash
# Python 環境
conda create -n katago python=3.10
conda activate katago

# PyTorch（CUDA 版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 其他依賴
pip install numpy h5py tqdm tensorboard
```

### 取得訓練程式碼

```bash
git clone https://github.com/lightvector/KataGo.git
cd KataGo/python
```

---

## 訓練設定

### 設定檔結構

```yaml
# configs/train_config.yaml

# 模型架構
model:
  num_blocks: 10          # 殘差塊數量
  trunk_channels: 128     # 主幹通道數
  policy_channels: 32     # Policy 頭通道數
  value_channels: 32      # Value 頭通道數

# 訓練參數
training:
  batch_size: 256
  learning_rate: 0.001
  lr_schedule: "cosine"
  weight_decay: 0.0001
  epochs: 100

# 自我對弈參數
selfplay:
  num_games_per_iteration: 1000
  max_visits: 600
  temperature: 1.0
  temperature_drop_move: 20

# 資料設定
data:
  max_history_games: 500000
  shuffle_buffer_size: 100000
```

### 模型規模對照

| 名稱 | num_blocks | trunk_channels | 參數量 |
|------|-----------|----------------|--------|
| b6c96 | 6 | 96 | ~1M |
| b10c128 | 10 | 128 | ~3M |
| b18c384 | 18 | 384 | ~20M |
| b40c256 | 40 | 256 | ~45M |

**動畫對應**：
- 🎬 F2 網路大小 vs 棋力：容量縮放
- 🎬 F6 神經縮放律：雙對數關係

---

## 訓練流程

### 步驟 1：初始化模型

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
print(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}")
```

### 步驟 2：自我對弈產生資料

```bash
# 編譯 C++ 引擎
cd ../cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=CUDA
make -j$(nproc)

# 執行自我對弈
./katago selfplay \
  -model ../python/model_init.pt \
  -output-dir ../python/selfplay_data \
  -config selfplay.cfg \
  -num-games 1000
```

自我對弈設定（selfplay.cfg）：

```ini
maxVisits = 600
numSearchThreads = 4

# 溫度設定（增加探索）
chosenMoveTemperature = 1.0
chosenMoveTemperatureEarly = 1.0
chosenMoveTemperatureHalflife = 20

# Dirichlet 噪聲（增加多樣性）
rootNoiseEnabled = true
rootDirichletNoiseTotalConcentration = 10.83
rootDirichletNoiseWeight = 0.25
```

**動畫對應**：
- 🎬 C3 探索 vs 利用：溫度參數
- 🎬 E10 Dirichlet 噪聲：根節點探索

### 步驟 3：訓練神經網路

```python
# train.py
import torch
from torch.utils.data import DataLoader
from model import KataGoModel
from dataset import SelfPlayDataset

# 載入資料
dataset = SelfPlayDataset('selfplay_data/')
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# 載入模型
model = KataGoModel(config)
model.load_state_dict(torch.load('model_init.pt'))
model = model.cuda()

# 優化器
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

# 學習率排程
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=0.00001
)

# 訓練循環
for epoch in range(100):
    model.train()
    total_loss = 0

    for batch in dataloader:
        inputs = batch['inputs'].cuda()
        policy_target = batch['policy'].cuda()
        value_target = batch['value'].cuda()
        ownership_target = batch['ownership'].cuda()

        # 前向傳播
        policy_pred, value_pred, ownership_pred = model(inputs)

        # 計算損失
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

        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")

    # 儲存檢查點
    torch.save(model.state_dict(), f'model_epoch{epoch}.pt')
```

**動畫對應**：
- 🎬 D5 梯度下降：optimizer.step()
- 🎬 K2 動量：Adam 優化器
- 🎬 K4 學習率衰減：CosineAnnealingLR
- 🎬 K5 梯度裁剪：clip_grad_norm_

### 步驟 4：評估與迭代

```bash
# 評估新模型 vs 舊模型
./katago match \
  -model1 model_epoch99.pt \
  -model2 model_init.pt \
  -num-games 100 \
  -output match_results.txt
```

如果新模型勝率 > 55%，則取代舊模型，進入下一輪迭代。

---

## 損失函數詳解

### Policy Loss

```python
# 交叉熵損失
policy_loss = -sum(target * log(pred))
```

目標：讓預測的機率分布接近 MCTS 搜索結果。

**動畫對應**：
- 🎬 J1 策略熵：交叉熵
- 🎬 J2 KL 散度：分布距離

### Value Loss

```python
# 均方誤差
value_loss = (pred - actual_result)^2
```

目標：預測對局最終結果（勝/負/和）。

### Ownership Loss

```python
# 每點歸屬預測
ownership_loss = mean((pred - actual_ownership)^2)
```

目標：預測每個位置最終歸屬。

---

## 進階技巧

### 1. 資料增強

利用棋盤的對稱性：

```python
def augment_data(board, policy, ownership):
    """對 D4 群的 8 種變換進行資料增強"""
    augmented = []

    for rotation in range(4):
        for flip in [False, True]:
            # 旋轉與翻轉
            aug_board = transform(board, rotation, flip)
            aug_policy = transform(policy, rotation, flip)
            aug_ownership = transform(ownership, rotation, flip)
            augmented.append((aug_board, aug_policy, aug_ownership))

    return augmented
```

**動畫對應**：
- 🎬 A9 棋盤對稱性：D4 群
- 🎬 L4 資料增強：對稱性利用

### 2. 課程學習

從簡單到複雜：

```python
# 先用較少搜索次數訓練
schedule = [
    (100, 10000),   # 100 visits, 10000 games
    (200, 20000),   # 200 visits, 20000 games
    (400, 50000),   # 400 visits, 50000 games
    (600, 100000),  # 600 visits, 100000 games
]
```

**動畫對應**：
- 🎬 E12 訓練課程：課程學習

### 3. 混合精度訓練

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

### 4. 多 GPU 訓練

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# 初始化分散式
dist.init_process_group(backend='nccl')

# 包裝模型
model = DistributedDataParallel(model)
```

---

## 監控與除錯

### TensorBoard 監控

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/training')

# 記錄損失
writer.add_scalar('Loss/policy', policy_loss, step)
writer.add_scalar('Loss/value', value_loss, step)
writer.add_scalar('Loss/total', total_loss, step)

# 記錄學習率
writer.add_scalar('LR', scheduler.get_last_lr()[0], step)
```

```bash
tensorboard --logdir runs
```

### 常見問題

| 問題 | 可能原因 | 解決方案 |
|------|---------|---------|
| 損失不下降 | 學習率太低/太高 | 調整學習率 |
| 損失震盪 | 批次大小太小 | 增加批次大小 |
| 過擬合 | 資料不足 | 產生更多自我對弈資料 |
| 棋力不增長 | 搜索次數太少 | 增加 maxVisits |

**動畫對應**：
- 🎬 L1 過擬合：過度適應
- 🎬 L2 正則化：weight_decay
- 🎬 D6 學習率效應：調參

---

## 小規模實驗建議

如果你只是想實驗，建議：

1. **使用 9×9 棋盤**：大幅減少計算量
2. **使用小型模型**：b6c96 足夠實驗
3. **減少搜索次數**：100-200 visits
4. **使用預訓練模型微調**：比從零開始快

```bash
# 9×9 棋盤設定
boardSize = 9
maxVisits = 100
```

---

## 延伸閱讀

- [原始碼導讀](../source-code) — 理解程式碼結構
- [參與開源社群](../contributing) — 加入分散式訓練
- [KataGo 的關鍵創新](../../how-it-works/katago-innovations) — 50 倍效率的秘密
