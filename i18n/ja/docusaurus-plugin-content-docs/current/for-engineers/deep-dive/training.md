---
sidebar_position: 3
title: KataGo訓練メカニズム解析
description: KataGoの自己対局訓練フローと核心技術の詳細解説
---

# KataGo訓練メカニズム解析

本記事では、KataGoの訓練メカニズムを詳しく解説し、自己対局訓練の動作原理を理解できるようにします。

---

## 訓練概要

### 訓練ループ

```
初期モデル → 自己対局 → データ収集 → 訓練更新 → より強いモデル → 繰り返し
```

**アニメーション対応**：
- E5 自己対局 ↔ 不動点収束
- E6 棋力曲線 ↔ S字カーブ成長
- H1 MDP ↔ マルコフ連鎖

### ハードウェア要件

| モデル規模 | GPUメモリ | 訓練時間 |
|---------|-----------|---------|
| b6c96 | 4 GB | 数時間 |
| b10c128 | 8 GB | 1-2日 |
| b18c384 | 16 GB | 1-2週間 |
| b40c256 | 24 GB以上 | 数週間 |

---

## 環境設定

### 依存関係のインストール

```bash
# Python環境
conda create -n katago python=3.10
conda activate katago

# PyTorch（CUDA版）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# その他の依存関係
pip install numpy h5py tqdm tensorboard
```

### 訓練コードの取得

```bash
git clone https://github.com/lightvector/KataGo.git
cd KataGo/python
```

---

## 訓練設定

### 設定ファイル構造

```yaml
# configs/train_config.yaml

# モデルアーキテクチャ
model:
  num_blocks: 10          # 残差ブロック数
  trunk_channels: 128     # トランクチャネル数
  policy_channels: 32     # Policyヘッドチャネル数
  value_channels: 32      # Valueヘッドチャネル数

# 訓練パラメータ
training:
  batch_size: 256
  learning_rate: 0.001
  lr_schedule: "cosine"
  weight_decay: 0.0001
  epochs: 100

# 自己対局パラメータ
selfplay:
  num_games_per_iteration: 1000
  max_visits: 600
  temperature: 1.0
  temperature_drop_move: 20

# データ設定
data:
  max_history_games: 500000
  shuffle_buffer_size: 100000
```

### モデル規模対照表

| 名前 | num_blocks | trunk_channels | パラメータ数 |
|------|-----------|----------------|--------|
| b6c96 | 6 | 96 | ~1M |
| b10c128 | 10 | 128 | ~3M |
| b18c384 | 18 | 384 | ~20M |
| b40c256 | 40 | 256 | ~45M |

**アニメーション対応**：
- F2 ネットワークサイズ vs 棋力：容量スケーリング
- F6 ニューラルスケーリング則：両対数関係

---

## 訓練フロー

### ステップ1：モデルの初期化

```python
# init_model.py
import torch
from model import KataGoModel

config = {
    'num_blocks': 10,
    'trunk_channels': 128,
    'input_features': 22,
    'policy_size': 362,  # 361 + パス
}

model = KataGoModel(config)
torch.save(model.state_dict(), 'model_init.pt')
print(f"モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")
```

### ステップ2：自己対局でデータを生成

```bash
# C++エンジンをコンパイル
cd ../cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=CUDA
make -j$(nproc)

# 自己対局を実行
./katago selfplay \
  -model ../python/model_init.pt \
  -output-dir ../python/selfplay_data \
  -config selfplay.cfg \
  -num-games 1000
```

自己対局設定（selfplay.cfg）：

```ini
maxVisits = 600
numSearchThreads = 4

# 温度設定（探索を増やす）
chosenMoveTemperature = 1.0
chosenMoveTemperatureEarly = 1.0
chosenMoveTemperatureHalflife = 20

# ディリクレノイズ（多様性を増やす）
rootNoiseEnabled = true
rootDirichletNoiseTotalConcentration = 10.83
rootDirichletNoiseWeight = 0.25
```

**アニメーション対応**：
- C3 探索 vs 活用：温度パラメータ
- E10 ディリクレノイズ：ルートノード探索

### ステップ3：ニューラルネットワークの訓練

```python
# train.py
import torch
from torch.utils.data import DataLoader
from model import KataGoModel
from dataset import SelfPlayDataset

# データをロード
dataset = SelfPlayDataset('selfplay_data/')
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# モデルをロード
model = KataGoModel(config)
model.load_state_dict(torch.load('model_init.pt'))
model = model.cuda()

# オプティマイザ
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

# 学習率スケジューラ
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=0.00001
)

# 訓練ループ
for epoch in range(100):
    model.train()
    total_loss = 0

    for batch in dataloader:
        inputs = batch['inputs'].cuda()
        policy_target = batch['policy'].cuda()
        value_target = batch['value'].cuda()
        ownership_target = batch['ownership'].cuda()

        # 順伝播
        policy_pred, value_pred, ownership_pred = model(inputs)

        # 損失を計算
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

        # 逆伝播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")

    # チェックポイントを保存
    torch.save(model.state_dict(), f'model_epoch{epoch}.pt')
```

**アニメーション対応**：
- D5 勾配降下：optimizer.step()
- K2 モメンタム：Adamオプティマイザ
- K4 学習率減衰：CosineAnnealingLR
- K5 勾配クリッピング：clip_grad_norm_

### ステップ4：評価と反復

```bash
# 新モデル vs 旧モデルを評価
./katago match \
  -model1 model_epoch99.pt \
  -model2 model_init.pt \
  -num-games 100 \
  -output match_results.txt
```

新モデルの勝率が55%以上であれば、旧モデルを置き換えて次のイテレーションに進みます。

---

## 損失関数の詳細

### Policy Loss

```python
# 交差エントロピー損失
policy_loss = -sum(target * log(pred))
```

目標：予測確率分布をMCTS探索結果に近づける。

**アニメーション対応**：
- J1 方策エントロピー：交差エントロピー
- J2 KLダイバージェンス：分布距離

### Value Loss

```python
# 平均二乗誤差
value_loss = (pred - actual_result)^2
```

目標：対局の最終結果（勝/負/引き分け）を予測。

### Ownership Loss

```python
# 各点の帰属予測
ownership_loss = mean((pred - actual_ownership)^2)
```

目標：各位置の最終帰属を予測。

---

## 高度なテクニック

### 1. データ拡張

盤面の対称性を利用：

```python
def augment_data(board, policy, ownership):
    """D4群の8種類の変換でデータ拡張"""
    augmented = []

    for rotation in range(4):
        for flip in [False, True]:
            # 回転と反転
            aug_board = transform(board, rotation, flip)
            aug_policy = transform(policy, rotation, flip)
            aug_ownership = transform(ownership, rotation, flip)
            augmented.append((aug_board, aug_policy, aug_ownership))

    return augmented
```

**アニメーション対応**：
- A9 盤面の対称性：D4群
- L4 データ拡張：対称性の活用

### 2. カリキュラム学習

簡単なものから複雑なものへ：

```python
# 最初は少ない探索回数で訓練
schedule = [
    (100, 10000),   # 100 visits, 10000 games
    (200, 20000),   # 200 visits, 20000 games
    (400, 50000),   # 400 visits, 50000 games
    (600, 100000),  # 600 visits, 100000 games
]
```

**アニメーション対応**：
- E12 訓練カリキュラム：カリキュラム学習

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

### 4. マルチGPU訓練

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# 分散初期化
dist.init_process_group(backend='nccl')

# モデルをラップ
model = DistributedDataParallel(model)
```

---

## 監視とデバッグ

### TensorBoardによる監視

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/training')

# 損失を記録
writer.add_scalar('Loss/policy', policy_loss, step)
writer.add_scalar('Loss/value', value_loss, step)
writer.add_scalar('Loss/total', total_loss, step)

# 学習率を記録
writer.add_scalar('LR', scheduler.get_last_lr()[0], step)
```

```bash
tensorboard --logdir runs
```

### よくある問題

| 問題 | 考えられる原因 | 解決策 |
|------|---------|---------|
| 損失が下がらない | 学習率が低すぎる/高すぎる | 学習率を調整 |
| 損失が振動する | バッチサイズが小さすぎる | バッチサイズを増やす |
| 過学習 | データ不足 | より多くの自己対局データを生成 |
| 棋力が向上しない | 探索回数が少なすぎる | maxVisitsを増やす |

**アニメーション対応**：
- L1 過学習：過適合
- L2 正則化：weight_decay
- D6 学習率の効果：パラメータ調整

---

## 小規模実験の提案

実験だけなら以下をお勧めします：

1. **9×9盤面を使用**：計算量を大幅に削減
2. **小型モデルを使用**：b6c96で実験には十分
3. **探索回数を減らす**：100-200 visits
4. **事前訓練モデルをファインチューニング**：ゼロから始めるより速い

```bash
# 9×9盤面設定
boardSize = 9
maxVisits = 100
```

---

## 関連記事

- [ソースコード解説](../source-code) — コード構造を理解
- [オープンソースコミュニティへの参加](../contributing) — 分散訓練に参加
- [KataGoの重要なイノベーション](../../how-it-works/katago-innovations) — 50倍の効率の秘密
