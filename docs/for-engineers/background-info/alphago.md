---
sidebar_position: 1
title: AlphaGo 論文解讀
---

# AlphaGo 論文解讀

本文深入解析 DeepMind 發表於 Nature 的經典論文《Mastering the game of Go with deep neural networks and tree search》，以及後續的 AlphaGo Zero 和 AlphaZero 論文。

## AlphaGo 的歷史意義

圍棋長期被視為人工智慧的「聖杯」挑戰。與西洋棋不同，圍棋的搜索空間極其龐大：

| 遊戲 | 平均分支因子 | 平均遊戲長度 | 狀態空間 |
|------|-------------|-------------|----------|
| 西洋棋 | ~35 | ~80 | ~10^47 |
| 圍棋 | ~250 | ~150 | ~10^170 |

傳統的暴力搜索方法在圍棋上完全不可行。2016 年 AlphaGo 擊敗李世乭，證明了深度學習與強化學習結合的強大威力。

### 里程碑事件

- **2015 年 10 月**：AlphaGo Fan 以 5:0 擊敗歐洲冠軍樊麾（職業二段）
- **2016 年 3 月**：AlphaGo Lee 以 4:1 擊敗世界冠軍李世乭（職業九段）
- **2017 年 5 月**：AlphaGo Master 以 3:0 擊敗世界排名第一的柯潔
- **2017 年 10 月**：AlphaGo Zero 發表，純自我對弈訓練，超越所有前代版本

## 核心技術架構

AlphaGo 的核心創新在於結合三個關鍵技術：

```mermaid
graph TD
    subgraph AlphaGo["AlphaGo 架構"]
        PolicyNet["Policy Network<br/>(落子策略)"]
        ValueNet["Value Network<br/>(勝率評估)"]
        MCTS["MCTS<br/>(蒙特卡羅樹搜索)"]

        PolicyNet --> MCTS
        ValueNet --> MCTS
    end
```

### Policy Network（策略網路）

Policy Network 負責預測每個位置的落子機率，用於指導搜索方向。

#### 網路架構

```mermaid
graph TD
    Input["輸入層：19×19×48 特徵平面"]
    Conv1["卷積層 1：5×5 卷積核，192 個濾波器"]
    Conv2_12["卷積層 2-12：3×3 卷積核，192 個濾波器"]
    Output["輸出層：19×19 機率分佈（softmax）"]

    Input --> Conv1
    Conv1 --> Conv2_12
    Conv2_12 --> Output
```

#### 輸入特徵

AlphaGo 使用 48 個特徵平面作為輸入：

| 特徵 | 平面數 | 描述 |
|------|--------|------|
| 棋子顏色 | 3 | 黑子、白子、空點 |
| 氣數 | 8 | 1氣、2氣、...、8氣以上 |
| 叫吃後氣數 | 8 | 吃子後會有多少氣 |
| 提子數 | 8 | 該位置可提多少子 |
| 打劫 | 1 | 是否為劫爭位置 |
| 落子合法性 | 1 | 該位置是否可落子 |
| 連續 1-8 手前的位置 | 8 | 前幾手的落子位置 |
| 輪到哪方下 | 1 | 當前輪到黑或白 |

#### 訓練方式

Policy Network 的訓練分為兩階段：

**第一階段：監督學習 (SL Policy Network)**
- 使用 KGS 圍棋伺服器的 3000 萬局棋譜
- 目標：預測人類棋手的下一手
- 達到 57% 的預測準確率

**第二階段：強化學習 (RL Policy Network)**
- 從 SL Policy Network 開始
- 與先前版本的自己對弈
- 使用 REINFORCE 演算法優化

```python
# 簡化的 Policy Gradient 更新
# reward: +1 勝利, -1 失敗
loss = -log(policy[action]) * reward
```

### Value Network（價值網路）

Value Network 評估當前局面的勝率，用於減少搜索深度。

#### 網路架構

```mermaid
graph TD
    Input["輸入層：19×19×48 特徵平面<br/>（與 Policy Network 相同）"]
    Conv["卷積層 1-12：與 Policy Network 相似"]
    FC["全連接層：256 個神經元"]
    Output["輸出層：1 個神經元<br/>（tanh，範圍 [-1, 1]）"]

    Input --> Conv
    Conv --> FC
    FC --> Output
```

#### 訓練方式

Value Network 使用 RL Policy Network 自我對弈產生的 3000 萬局面訓練：

- 從每局棋中隨機取樣一個局面
- 用最終勝負作為標籤
- 使用 MSE 損失函數

```python
# Value Network 訓練
value_prediction = value_network(position)
loss = (value_prediction - game_outcome) ** 2
```

**為什麼每局只取一個樣本？**

如果取多個樣本，同一局棋的相鄰局面會高度相關，導致過擬合。隨機取樣能確保訓練資料的多樣性。

## 蒙特卡羅樹搜索 (MCTS)

MCTS 是 AlphaGo 的決策核心，結合神經網路來高效搜索最佳著法。

### MCTS 四步驟

```mermaid
graph LR
    subgraph Step1["(1) Selection"]
        S1["選擇<br/>最佳路徑"]
    end
    subgraph Step2["(2) Expansion"]
        S2["擴展<br/>新節點"]
    end
    subgraph Step3["(3) Evaluation"]
        S3["神經網路<br/>評估"]
    end
    subgraph Step4["(4) Backpropagation"]
        S4["回傳<br/>更新"]
    end

    Step1 --> Step2
    Step2 --> Step3
    Step3 --> Step4
```

### 選擇公式 (PUCT)

AlphaGo 使用 PUCT (Predictor + UCT) 公式選擇要探索的分支：

```
a = argmax[Q(s,a) + u(s,a)]

u(s,a) = c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

其中：
- **Q(s,a)**：動作 a 的平均價值（exploitation）
- **P(s,a)**：Policy Network 預測的先驗機率
- **N(s)**：父節點的訪問次數
- **N(s,a)**：該動作的訪問次數
- **c_puct**：探索常數，平衡 exploration 與 exploitation

### 搜索過程詳解

1. **Selection**：從根節點開始，使用 PUCT 公式選擇動作，直到達到葉節點
2. **Expansion**：在葉節點展開新的子節點，用 Policy Network 初始化先驗機率
3. **Evaluation**：結合 Value Network 評估和快速走子模擬 (Rollout) 來評估價值
4. **Backpropagation**：將評估值沿路徑回傳，更新 Q 值和 N 值

### Rollout（快速走子）

AlphaGo（非 Zero 版）還使用一個小型快速策略網路進行模擬：

```
葉節點 → 快速隨機走子至終局 → 計算勝負
```

最終評估值結合 Value Network 和 Rollout：

```
V = λ * v_network + (1-λ) * v_rollout
```

AlphaGo 使用 λ = 0.5，給予兩者相等權重。

## Self-play 訓練方法

Self-play 是 AlphaGo 的核心訓練策略，讓 AI 透過與自己對弈來持續提升。

### 訓練循環

```mermaid
graph LR
    subgraph SelfPlay["Self-play 訓練循環"]
        CurrentModel["當前模型"]
        Play["自我對弈"]
        GenerateData["產生資料"]
        DataPool["資料池"]
        Train["訓練"]
        NewModel["新模型"]

        CurrentModel --> Play
        Play --> GenerateData
        GenerateData --> DataPool
        DataPool --> Train
        Train --> NewModel
        NewModel --> CurrentModel
    end
```

### 為什麼 Self-play 有效？

1. **無限資料**：不受人類棋譜數量限制
2. **自適應難度**：對手強度與自己同步提升
3. **探索創新**：不受人類固有思維模式限制
4. **目標明確**：直接優化勝率，而非模仿人類

## AlphaGo Zero 的改進

2017 年發表的 AlphaGo Zero 帶來了革命性的改進：

### 主要差異

| 特性 | AlphaGo | AlphaGo Zero |
|------|---------|--------------|
| 初始訓練 | 人類棋譜監督學習 | 完全從零開始 |
| 網路架構 | 分離的 Policy/Value | 單一雙頭網路 |
| 網路結構 | 普通 CNN | ResNet |
| 特徵工程 | 48 個手工特徵 | 17 個簡單特徵 |
| Rollout | 需要 | 不需要 |
| 訓練時間 | 數月 | 3 天超越人類 |

### 架構簡化

```mermaid
graph TD
    subgraph AlphaGoZero["AlphaGo Zero 雙頭網路"]
        Input["輸入：19×19×17（簡化特徵）"]
        ResNet["ResNet 主幹<br/>(40 個殘差塊)"]
        PolicyHead["Policy Head<br/>(19×19+1)"]
        ValueHead["Value Head<br/>([-1,1])"]

        Input --> ResNet
        ResNet --> PolicyHead
        ResNet --> ValueHead
    end
```

### 簡化的輸入特徵

AlphaGo Zero 僅使用 17 個特徵平面：

- 8 個平面：自己最近 8 手的棋子位置
- 8 個平面：對手最近 8 手的棋子位置
- 1 個平面：當前輪到哪方（全 0 或全 1）

### 訓練改進

1. **純 Self-play**：不使用任何人類資料
2. **直接使用 MCTS 機率作為訓練目標**：而非二元的勝負
3. **沒有 Rollout**：完全依賴 Value Network
4. **單一網路訓練**：Policy 和 Value 共享參數，互相增強

## AlphaZero 的通用化

2017 年底發表的 AlphaZero 將相同架構應用於圍棋、西洋棋和將棋：

### 關鍵特點

- **零領域知識**：除了遊戲規則外，不使用任何領域特定知識
- **統一架構**：同一套演算法適用於不同棋類
- **更快訓練**：
  - 圍棋：8 小時超越 AlphaGo Lee
  - 西洋棋：4 小時超越 Stockfish
  - 將棋：2 小時超越 Elmo

### 與 AlphaGo Zero 的差異

| 特性 | AlphaGo Zero | AlphaZero |
|------|-------------|-----------|
| 目標遊戲 | 僅圍棋 | 圍棋、西洋棋、將棋 |
| 對稱性利用 | 利用圍棋 8 重對稱 | 不假設對稱性 |
| 超參數調整 | 針對圍棋優化 | 通用設定 |
| 訓練方式 | 最佳模型自我對弈 | 最新模型自我對弈 |

## 實作重點

如果你想實作類似系統，以下是關鍵考量：

### 計算資源

AlphaGo 的訓練需要龐大的計算資源：

- **AlphaGo Lee**：176 GPU + 48 TPU
- **AlphaGo Zero**：4 TPU（訓練）+ 1 TPU（自我對弈）
- **AlphaZero**：5000 TPU（訓練）

### 關鍵超參數

```python
# MCTS 相關
num_simulations = 800     # 每手搜索模擬次數
c_puct = 1.5              # 探索常數
temperature = 1.0         # 選擇動作的溫度參數

# 訓練相關
batch_size = 2048
learning_rate = 0.01      # 含衰減
l2_regularization = 1e-4
```

### 常見問題

1. **訓練不穩定**：使用較小的學習率，增加 batch size
2. **過擬合**：確保訓練資料多樣性，使用正規化
3. **搜索效率**：優化 GPU 批次推理，並行化 MCTS

## 延伸閱讀

- [原始論文：Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961)
- [AlphaGo Zero 論文：Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270)
- [AlphaZero 論文：A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](https://www.science.org/doi/10.1126/science.aar6404)

理解 AlphaGo 的技術後，接下來讓我們看看 [KataGo 如何在此基礎上做出改進](./katago-paper.md)。
