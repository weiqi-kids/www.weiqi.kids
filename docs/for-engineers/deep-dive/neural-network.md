---
sidebar_position: 4
title: 神經網路架構詳解
description: 深入解析 KataGo 的神經網路設計、輸入特徵與多頭輸出架構
---

# 神經網路架構詳解

本文深入解析 KataGo 神經網路的完整架構，從輸入特徵編碼到多頭輸出設計。

---

## 架構總覽

KataGo 使用**單一神經網路、多頭輸出**的設計：

```
輸入特徵（19×19×22）
        │
        ▼
┌───────────────────┐
│     初始卷積層     │
│   256 filters     │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│     殘差塔        │
│  20-60 個殘差塊   │
│  + 全局池化層     │
└─────────┬─────────┘
          │
    ┌─────┴─────┬─────────┬─────────┐
    │           │         │         │
    ▼           ▼         ▼         ▼
 Policy      Value     Score   Ownership
  Head       Head      Head      Head
    │           │         │         │
    ▼           ▼         ▼         ▼
362 機率    勝率      目數差    361 歸屬
(含 pass)  (-1~+1)    (目)     (-1~+1)
```

---

## 輸入特徵編碼

### 特徵平面總覽

KataGo 使用 **22 個特徵平面**（19×19×22），每個平面是一個 19×19 的矩陣：

| 平面 | 內容 | 說明 |
|------|------|------|
| 0 | 己方棋子 | 1 = 有己方棋子，0 = 無 |
| 1 | 對方棋子 | 1 = 有對方棋子，0 = 無 |
| 2 | 空點 | 1 = 空點，0 = 有棋子 |
| 3-10 | 歷史狀態 | 過去 8 步的棋盤變化 |
| 11 | 劫禁點 | 1 = 此處為劫禁，0 = 可下 |
| 12-17 | 氣數編碼 | 1氣、2氣、3氣...的棋串 |
| 18-21 | 規則編碼 | 中國/日本規則、komi 等 |

### 歷史狀態堆疊

為了讓神經網路理解局面的**動態變化**，KataGo 會堆疊過去 8 步的棋盤狀態：

```python
# 歷史狀態編碼（概念）
def encode_history(game_history, current_player):
    features = []

    for t in range(8):  # 過去 8 步
        if t < len(game_history):
            board = game_history[-(t+1)]
            # 編碼該時間點的己方/對方棋子
            features.append(encode_board(board, current_player))
        else:
            # 歷史不足，填零
            features.append(np.zeros((19, 19)))

    return np.stack(features, axis=0)
```

### 規則編碼

KataGo 支援多種規則，透過特徵平面告知神經網路：

```python
# 規則編碼（概念）
def encode_rules(rules, komi):
    rule_features = np.zeros((4, 19, 19))

    # 規則類型（one-hot）
    if rules == "chinese":
        rule_features[0] = 1.0
    elif rules == "japanese":
        rule_features[1] = 1.0

    # Komi 正規化
    normalized_komi = komi / 15.0  # 正規化到 [-1, 1]
    rule_features[2] = normalized_komi

    # 當前玩家
    rule_features[3] = 1.0 if current_player == BLACK else 0.0

    return rule_features
```

---

## 主幹網路：殘差塔

### 殘差塊結構

KataGo 使用 **Pre-activation ResNet** 結構：

```
輸入 x
    │
    ├────────────────────┐
    │                    │
    ▼                    │
BatchNorm                │
    │                    │
    ▼                    │
ReLU                     │
    │                    │
    ▼                    │
Conv 3×3                 │
    │                    │
    ▼                    │
BatchNorm                │
    │                    │
    ▼                    │
ReLU                     │
    │                    │
    ▼                    │
Conv 3×3                 │
    │                    │
    ▼                    │
    +  ←─────────────────┘ (殘差連接)
    │
    ▼
輸出
```

### 程式碼示例

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        return out + residual  # 殘差連接
```

### 全局池化層

KataGo 的關鍵創新之一：在殘差塊中加入**全局池化**，讓網路能看到全局資訊：

```python
class GlobalPoolingBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.fc = nn.Linear(channels, channels)

    def forward(self, x):
        # 局部路徑
        local = self.conv(x)

        # 全局路徑
        global_pool = x.mean(dim=[2, 3])  # 全局平均池化
        global_fc = self.fc(global_pool)
        global_broadcast = global_fc.unsqueeze(2).unsqueeze(3)
        global_broadcast = global_broadcast.expand(-1, -1, 19, 19)

        # 融合
        return local + global_broadcast
```

**為什麼需要全局池化？**

傳統卷積只看局部（3×3 感受野），即使堆疊很多層，對全局資訊的感知仍有限。全局池化讓網路能直接「看到」：
- 整盤棋的子數差異
- 全局的勢力分佈
- 整體的形勢判斷

---

## 輸出頭設計

### Policy Head（策略頭）

輸出每個位置的落子機率：

```python
class PolicyHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2, 1)  # 1×1 卷積
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2 * 19 * 19, 362)  # 361 + pass

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.softmax(out, dim=1)  # 機率分佈
```

**輸出格式**：362 維向量
- 索引 0-360：棋盤上 361 個位置的落子機率
- 索引 361：pass 的機率

### Value Head（價值頭）

輸出當前局面的勝率：

```python
class ValueHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(19 * 19, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.tanh(self.fc2(out))  # 輸出 -1 到 +1
        return out
```

**輸出格式**：單一數值 [-1, +1]
- +1：己方必勝
- -1：對方必勝
- 0：均勢

### Score Head（目數頭）

KataGo 獨有，預測最終目數差：

```python
class ScoreHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(19 * 19, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)  # 無限制輸出
        return out
```

**輸出格式**：單一數值（目數）
- 正數：己方領先
- 負數：對方領先

### Ownership Head（領地頭）

預測每個點最終歸屬：

```python
class OwnershipHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 1)
        self.bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        out = F.relu(self.bn(self.conv1(x)))
        out = torch.tanh(self.conv2(out))  # 每點 -1 到 +1
        return out.view(out.size(0), -1)  # 展平為 361
```

**輸出格式**：361 維向量，每個值在 [-1, +1]
- +1：該點屬於己方領地
- -1：該點屬於對方領地
- 0：中立或爭議區域

---

## 與 AlphaZero 的差異

| 面向 | AlphaZero | KataGo |
|------|-----------|--------|
| **輸出頭** | 2 個（Policy + Value） | **4 個**（+ Score + Ownership） |
| **全局池化** | 無 | **有** |
| **輸入特徵** | 17 平面 | **22 平面**（含規則編碼） |
| **殘差塊** | 標準 ResNet | **Pre-activation + 全局池化** |
| **多規則支援** | 無 | **有**（透過特徵編碼） |

---

## 模型規模

KataGo 提供不同規模的模型：

| 模型 | 殘差塊數 | 通道數 | 參數量 | 適用場景 |
|------|---------|--------|--------|---------|
| b10c128 | 10 | 128 | ~5M | CPU、快速測試 |
| b18c384 | 18 | 384 | ~75M | 一般 GPU |
| b40c256 | 40 | 256 | ~95M | 高階 GPU |
| b60c320 | 60 | 320 | ~200M | 頂級 GPU |

**命名規則**：`b{殘差塊數}c{通道數}`

---

## 完整網路實作

```python
class KataGoNetwork(nn.Module):
    def __init__(self, num_blocks=18, channels=384):
        super().__init__()

        # 初始卷積
        self.initial_conv = nn.Conv2d(22, channels, 3, padding=1)
        self.initial_bn = nn.BatchNorm2d(channels)

        # 殘差塔
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_blocks)
        ])

        # 全局池化塊（每隔幾個殘差塊插入一個）
        self.global_pooling_blocks = nn.ModuleList([
            GlobalPoolingBlock(channels) for _ in range(num_blocks // 6)
        ])

        # 輸出頭
        self.policy_head = PolicyHead(channels)
        self.value_head = ValueHead(channels)
        self.score_head = ScoreHead(channels)
        self.ownership_head = OwnershipHead(channels)

    def forward(self, x):
        # 初始卷積
        out = F.relu(self.initial_bn(self.initial_conv(x)))

        # 殘差塔
        gp_idx = 0
        for i, block in enumerate(self.residual_blocks):
            out = block(out)

            # 每 6 個殘差塊後插入全局池化
            if (i + 1) % 6 == 0 and gp_idx < len(self.global_pooling_blocks):
                out = self.global_pooling_blocks[gp_idx](out)
                gp_idx += 1

        # 輸出頭
        policy = self.policy_head(out)
        value = self.value_head(out)
        score = self.score_head(out)
        ownership = self.ownership_head(out)

        return {
            'policy': policy,
            'value': value,
            'score': score,
            'ownership': ownership
        }
```

---

## 延伸閱讀

- [MCTS 實作細節](../mcts-implementation) — 搜索與神經網路的結合
- [KataGo 訓練機制解析](../training) — 網路如何訓練
- [關鍵論文導讀](../papers) — 原始論文的數學推導
