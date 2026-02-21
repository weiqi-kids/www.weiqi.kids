---
sidebar_position: 3
title: KataGo 的關鍵創新
description: KataGo 如何以 50 倍效率提升超越 AlphaGo
---

# KataGo 的關鍵創新

KataGo 是 David Wu 於 2019 年發表的開源圍棋 AI，以更少的資源達到更強的棋力。本文將深入解析其技術創新。

## 為什麼需要 KataGo？

### AlphaGo 的問題

| 問題 | 說明 |
|------|------|
| **閉源** | DeepMind 從未公開 AlphaGo 的程式碼 |
| **資源需求** | 需要數千個 TPU 進行訓練 |
| **無法重現** | 普通開發者無法自行訓練 |
| **功能有限** | 只輸出策略和勝率 |

### KataGo 的解決方案

| 特點 | 說明 |
|------|------|
| **完全開源** | MIT 授權，程式碼公開 |
| **高效訓練** | 30 GPU × 19 天 = 超越 Leela Zero |
| **功能豐富** | 勝率、目數、領地、多規則支援 |
| **持續更新** | 社群分散式訓練持續改進 |

---

## 50 倍效率提升的秘密

### 效率對比

| 系統 | 訓練資源 | 達到棋力 |
|------|---------|---------|
| AlphaGo Zero | 4 TPU × 3 天 | 超越 AlphaGo |
| Leela Zero | 社群分散式 × 1 年 | 接近 AlphaGo Zero |
| KataGo | 30 GPU × 19 天 | 超越 Leela Zero |

**效率提升**：約 **50 倍**（相對於 Leela Zero）

### 關鍵創新總覽

| 創新 | 效果 |
|------|------|
| 整合式多頭網路 | 減少計算量，增加輸出資訊 |
| 輔助訓練目標 | 加速學習，提升局面理解 |
| Playout Cap 隨機化 | 提升泛化能力 |
| 全局池化 | 更好的全局判斷 |
| 動態批次推理 | 提升 GPU 利用率 |

---

## 創新 1：整合式多頭網路

### 與 AlphaGo Zero 的差異

**AlphaGo Zero**：
```
輸入 → 共享卷積層 → Policy Head
                  → Value Head
```

**KataGo**：
```
輸入 → 共享卷積層 → Policy Head
                  → Value Head
                  → Score Head（目數）
                  → Ownership Head（領地）
```

### 多頭網路架構

```
                    棋盤輸入（19×19×22）
                         │
                    ┌────┴────┐
                    │ 初始卷積 │
                    └────┬────┘
                         │
                    ┌────┴────┐
                    │ 殘差塔   │（20-60 個殘差塊）
                    │ + 全局池化│
                    └────┬────┘
                         │
    ┌──────────┬─────────┼─────────┬──────────┐
    │          │         │         │          │
    ▼          ▼         ▼         ▼          ▼
 Policy     Value     Score   Ownership   其他輔助
  Head      Head      Head      Head        Heads
    │          │         │         │          │
    ▼          ▼         ▼         ▼          ▼
 361 機率   勝率      目數差    361 歸屬    各種輔助
            (-1~+1)   (目)     機率        預測
```

**動畫對應**：
- 🎬 E3：雙頭網路 ↔ 多任務學習
- 🎬 E11：殘差塔 ↔ 多級處理
- 🎬 D12：殘差連接 ↔ 電路並聯

### 多頭輸出的價值

| 輸出頭 | 功能 | 應用 |
|--------|------|------|
| **Policy** | 下一步的機率分布 | 引導 MCTS 搜索 |
| **Value** | 勝率預測 | 評估局面 |
| **Score** | 目數預測 | 細微形勢判斷 |
| **Ownership** | 領地預測 | 視覺化分析 |

---

## 創新 2：輔助訓練目標

### 傳統方法的問題

AlphaGo Zero 只訓練兩個目標：
- Policy：預測 MCTS 的搜索結果
- Value：預測最終勝負

**問題**：學習信號稀疏，收斂慢。

### KataGo 的解決方案

增加多個輔助訓練目標：

```python
loss = (
    policy_loss +              # 預測搜索結果
    value_loss +               # 預測勝負
    score_loss +               # 預測目數差
    ownership_loss +           # 預測每點歸屬
    auxiliary_loss_1 + ...     # 其他輔助目標
)
```

### 輔助目標列表

| 目標 | 說明 | 效果 |
|------|------|------|
| **Score** | 預測終局目數差 | 更精確的形勢判斷 |
| **Ownership** | 預測每點歸屬 | 更好的局部理解 |
| **Future Position** | 預測未來棋盤狀態 | 更好的推理能力 |
| **TD Lambda** | 多步時序差分 | 更穩定的價值估計 |

**動畫對應**：
- 🎬 J1：策略熵 ↔ 夏農熵
- 🎬 J2：KL 散度 ↔ 相對熵
- 🎬 H7：TD 學習 ↔ 增量估計

---

## 創新 3：Playout Cap 隨機化

### 傳統方法的問題

AlphaGo Zero 訓練時：
- 每局都用 **固定次數** 的 MCTS 搜索（如 800 次）
- 神經網路只學會「有 800 次搜索時的最佳下法」
- 可能過度依賴搜索

### KataGo 的解決方案

**隨機改變每局的搜索次數**：

```python
def get_playout_cap():
    # 隨機選擇搜索深度
    caps = [20, 50, 100, 200, 400, 600, 800, 1000]
    weights = [0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05]
    return random.choices(caps, weights)[0]
```

### 效果

| 搜索次數 | 學習效果 |
|---------|---------|
| 低（20-50） | 神經網路必須「靠自己」判斷 |
| 中（100-400） | 平衡搜索與直覺 |
| 高（600-1000） | 學習精確的戰術 |

**結果**：神經網路在各種搜索深度下都表現良好，泛化能力更強。

**動畫對應**：
- 🎬 C2：大數法則 ↔ 收斂
- 🎬 K6：SGD 噪聲 ↔ 隨機性
- 🎬 L1：過擬合 ↔ 過度適應

---

## 創新 4：全局池化（Global Pooling）

### 傳統 CNN 的問題

卷積神經網路只看**局部**特徵：
- 3×3 卷積核只看 9 個點
- 即使堆疊很多層，感受野仍有限
- 難以捕捉全局資訊（如整體目數、勢力範圍）

### KataGo 的解決方案

在殘差塊中加入**全局池化**：

```
局部卷積特徵（19×19×C）
         │
         ├──────────────────────────┐
         │                          │
         ▼                          ▼
    繼續卷積                  全局平均池化
         │                          │
         │                     (1×1×C)
         │                          │
         │                     全連接層
         │                          │
         └──────────┬───────────────┘
                    │
                    ▼ （concat 或 加法）
              融合後特徵
```

### 實作細節

```python
class GlobalPoolingBlock(nn.Module):
    def forward(self, x):
        # x: (batch, channels, 19, 19)

        # 局部路徑
        local = self.conv(x)

        # 全局路徑
        global_pool = x.mean(dim=[2, 3])  # (batch, channels)
        global_fc = self.fc(global_pool)  # (batch, channels')
        global_broadcast = global_fc.unsqueeze(2).unsqueeze(3)
        global_broadcast = global_broadcast.expand(-1, -1, 19, 19)

        # 融合
        return local + global_broadcast
```

**動畫對應**：
- 🎬 G1：高維表示 ↔ 向量空間
- 🎬 D11：池化 ↔ 降採樣
- 🎬 D14：注意力機制 ↔ 選擇性聚焦

---

## 創新 5：動態批次推理

### MCTS 的瓶頸

MCTS 的搜索過程：
1. 選擇節點
2. **神經網路推理**（最慢）
3. 回傳更新
4. 重複

**問題**：如果一次只推理一個位置，GPU 利用率很低。

### 解決方案：虛擬損失 + 批次處理

```python
class ParallelMCTS:
    def search(self, batch_size=8):
        nodes_to_evaluate = []

        # 同時選擇多個節點
        for _ in range(batch_size):
            node = self.select_with_virtual_loss()
            nodes_to_evaluate.append(node)

        # 批次推理
        states = [n.state for n in nodes_to_evaluate]
        policies, values = self.network.batch_evaluate(states)

        # 回傳更新（移除虛擬損失）
        for node, policy, value in zip(nodes_to_evaluate, policies, values):
            node.backpropagate(value)
            node.remove_virtual_loss()
```

**動畫對應**：
- 🎬 C9：虛擬損失 ↔ 排斥力
- 🎬 E9：分散式訓練 ↔ 並行計算

---

## 技術創新總結

### AlphaGo Zero vs KataGo

| 面向 | AlphaGo Zero | KataGo |
|------|-------------|--------|
| **網路架構** | 2 頭（Policy + Value） | **4+ 頭**（加 Score、Ownership） |
| **全局資訊** | 純卷積 | **全局池化** |
| **搜索深度** | 固定 | **隨機化** |
| **訓練目標** | 2 個 | **多個輔助目標** |
| **推理效率** | 單一推理 | **批次推理** |

### 效率提升來源

```
效率提升 ≈ 50x

├── 多頭網路共享計算      ~2x
├── 輔助目標加速學習      ~3x
├── Playout Cap 隨機化    ~2x
├── 全局池化更好的表示    ~2x
└── 其他優化             ~2x
```

---

## 實用功能

### 多規則支援

KataGo 支援多種圍棋規則：

| 規則 | 特點 |
|------|------|
| 中國規則 | 數子法 |
| 日本規則 | 數目法 |
| 韓國規則 | 類似日本 |
| AGA 規則 | 美國規則 |
| Tromp-Taylor | 電腦圍棋常用 |
| New Zealand | 紐西蘭規則 |

### 棋力調整

```json
{
  "maxVisits": 100,      // 降低搜索次數
  "humanSLProfile": {    // 模擬人類錯誤
    "rank": "5d",        // 模擬五段水平
    "noise": 0.1
  }
}
```

### Analysis Engine

KataGo 的 Analysis Engine 提供豐富的 JSON API：

```json
{
  "id": "query1",
  "moves": [["B", "Q16"], ["W", "D4"]],
  "rules": "chinese",
  "analyzeTurns": [0, 1, 2]
}
```

---

## 延伸閱讀

- [概念速查表](../concepts/) — 109 個動畫概念
- [30 分鐘跑起第一個圍棋 AI](../../hands-on/) — 動手安裝 KataGo
- [KataGo 原始碼導讀](../../deep-dive/source-code) — 深入程式碼
- [訓練自己的模型](../../deep-dive/training) — 從零開始訓練
