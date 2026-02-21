---
sidebar_position: 2
title: AlphaGo 完整解析
description: 深入解析 AlphaGo 的神經網路架構與訓練方法
---

# AlphaGo 完整解析

AlphaGo 是 DeepMind 於 2015-2017 年開發的圍棋 AI，標誌著人工智慧的重大突破。本文將深入解析其技術細節。

## AlphaGo 的歷史意義

### 關鍵時刻

| 時間 | 事件 | 意義 |
|------|------|------|
| 2015.10 | 5:0 擊敗樊麾 | 首次擊敗職業棋手 |
| 2016.03 | 4:1 擊敗李世乭 | 超越世界頂尖 |
| 2017.05 | 3:0 擊敗柯潔 | 確立 AI 優勢 |
| 2017.10 | AlphaGo Zero | 不需人類棋譜 |

### 第 37 手「神之一手」

在對李世乭的第二盤，AlphaGo 第 37 手下出了人類從未想過的肩衝（五路）。

**專家反應**：
- 「這不是人類會下的棋」
- 「一定是 bug」
- 最終證明是致勝關鍵

這一手揭示了 AI 能發現人類數千年來忽略的下法。

---

## 神經網路架構詳解

### 輸入特徵

AlphaGo 將棋盤編碼為 **48 個特徵平面**：

| 特徵類型 | 平面數 | 說明 |
|---------|--------|------|
| 棋子位置 | 3 | 黑子、白子、空點 |
| 輪到誰下 | 1 | 當前執子方 |
| 氣數 | 8 | 1氣、2氣、...、≥8氣 |
| 吃子數 | 8 | 下此位置能吃多少子 |
| 歷史著法 | 8 | 最近 8 步的位置 |
| 劫 | 1 | 打劫位置 |
| ... | ... | 其他特徵 |

**動畫對應**：
- 🎬 A1：網格狀態 ↔ 離散網格
- 🎬 A8：狀態編碼 ↔ 三態系統
- 🎬 A10：歷史堆疊 ↔ 時間序列

### Policy Network（策略網路）

**架構**：
```
輸入層：19×19×48（棋盤特徵）
    │
    ▼
卷積層 1：192 filters, 5×5, ReLU
    │
    ▼
卷積層 2-12：192 filters, 3×3, ReLU（共 11 層）
    │
    ▼
卷積層 13：1 filter, 1×1
    │
    ▼
輸出層：19×19 = 361（每個位置的機率）
    │
    ▼
Softmax：正規化為機率分布
```

**動畫對應**：
- 🎬 D1：感知器 ↔ 閾值開關
- 🎬 D2：激活函數 ↔ 非線性響應
- 🎬 D9：卷積運算 ↔ 空間濾波
- 🎬 D10：特徵圖 ↔ 繞射圖案

### Value Network（價值網路）

**架構**：
```
輸入層：19×19×49（棋盤特徵 + 當前顏色）
    │
    ▼
卷積層 1-12：與 Policy Network 相同
    │
    ▼
卷積層 13：1 filter, 1×1
    │
    ▼
全連接層 1：256 neurons, ReLU
    │
    ▼
全連接層 2：1 neuron
    │
    ▼
Tanh：輸出 -1 到 +1（勝率）
```

**動畫對應**：
- 🎬 D4：損失地形 ↔ 損失曲面
- 🎬 D5：梯度下降 ↔ 球滾下山
- 🎬 D13：反向傳播 ↔ 波的反射

---

## 三階段訓練詳解

### 第一階段：監督學習（Supervised Learning）

**目標**：從人類棋譜學習下棋

**資料來源**：
- KGS 圍棋伺服器
- 約 3000 萬局棋譜
- 只使用業餘六段以上的棋譜

**訓練過程**：
```
for each 棋局:
    for each 落子:
        input = 棋盤狀態
        label = 人類下的位置

        prediction = PolicyNetwork(input)
        loss = CrossEntropy(prediction, label)

        更新網路參數（梯度下降）
```

**結果**：
- 準確率：57.0%（預測人類下一步）
- 棋力：約業餘三段

**動畫對應**：
- 🎬 D3：前向傳播 ↔ 前饋網路
- 🎬 D5：梯度下降 ↔ 球滾下山
- 🎬 D6：學習率效應 ↔ 欠阻尼/過阻尼

### 第二階段：強化學習（Reinforcement Learning）

**目標**：超越人類水平

**方法**：自我對弈 + 策略梯度

```python
# 策略梯度更新
for episode in self_play_games:
    states, actions, result = play_game(current_policy)

    for s, a in zip(states, actions):
        # 如果贏了，增加這些動作的機率
        # 如果輸了，減少這些動作的機率
        gradient = result * ∇log(P(a|s))
        policy.update(gradient)
```

**動畫對應**：
- 🎬 H4：策略梯度 ↔ 策略梯度法
- 🎬 H7：TD 學習 ↔ 增量估計
- 🎬 H8：優勢函數 ↔ 相對價值

### 第三階段：訓練 Value Network

**目標**：學習評估局面

**關鍵問題**：如何避免過擬合？

**解決方案**：
- 不直接使用完整棋局
- 從每局隨機取樣一個位置
- 使用該位置到終局的結果作為標籤

```python
# Value Network 訓練
for game in self_play_games:
    random_position = random.choice(game.positions)
    state = random_position.board
    value = game.final_result  # +1 或 -1

    prediction = ValueNetwork(state)
    loss = MSE(prediction, value)

    更新網路參數
```

**動畫對應**：
- 🎬 L1：過擬合 ↔ 過度適應
- 🎬 L2：正則化 ↔ 拉格朗日乘數

---

## MCTS 與神經網路的結合

### PUCT 公式

```
U(s,a) = c_puct × P(s,a) × √(Σ_b N(s,b)) / (1 + N(s,a))

選擇動作 = argmax_a [ Q(s,a) + U(s,a) ]
```

| 符號 | 意義 |
|------|------|
| Q(s,a) | 動作 a 的平均價值 |
| P(s,a) | Policy Network 預測的先驗機率 |
| N(s,a) | 動作 a 的訪問次數 |
| c_puct | 探索常數（通常約 1.5） |

**動畫對應**：
- 🎬 E4：PUCT 公式 ↔ 有偏擴散
- 🎬 C3：探索 vs 利用 ↔ 自由能權衡
- 🎬 C6：樹的成長 ↔ 晶體成長

### 搜索過程

```
function MCTS(root_state):
    for i in range(num_simulations):  # 通常數千次
        node = root

        # Selection：沿樹向下選擇
        while node.is_expanded:
            node = select_child(node)  # 用 PUCT

        # Expansion：展開新節點
        node.expand(PolicyNetwork(node.state))

        # Evaluation：用 Value Network 評估
        value = ValueNetwork(node.state)

        # Backpropagation：回傳更新
        while node:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent
            value = -value  # 交替視角

    # 選擇訪問次數最多的動作
    return argmax(root.children, key=lambda c: c.visit_count)
```

**動畫對應**：
- 🎬 C5：MCTS 四步驟 ↔ 樹的遍歷
- 🎬 C9：虛擬損失 ↔ 排斥力（用於並行）

---

## AlphaGo Zero 的進化

### 與原版的關鍵差異

| 面向 | AlphaGo | AlphaGo Zero |
|------|---------|--------------|
| 人類棋譜 | 需要 | 不需要 |
| 網路架構 | 分離 | 單一（雙頭） |
| 輸入特徵 | 48 平面 | 17 平面 |
| 殘差網路 | 無 | 有（40 層） |
| 訓練時間 | 數月 | 3 天 |

### 雙頭網路架構

```
輸入層：19×19×17
    │
    ▼
卷積層：256 filters, 3×3
    │
    ▼
殘差塔：40 個殘差塊
    │
    ├──────────────────────────┐
    │                          │
    ▼                          ▼
Policy Head                Value Head
卷積 + Softmax             卷積 + 全連接 + Tanh
    │                          │
    ▼                          ▼
361 個機率                  1 個數值
```

**動畫對應**：
- 🎬 E3：雙頭網路 ↔ 多任務學習
- 🎬 D12：殘差連接 ↔ 電路並聯
- 🎬 E11：殘差塔 ↔ 多級處理

### 從零開始的訓練

```
初始化：隨機權重
    │
    ├──→ 自我對弈（使用 MCTS）
    │        │
    │        ▼
    │    收集棋局資料
    │        │
    │        ▼
    └──← 訓練網路
           │
           ▼
        重複（約 500 萬局）
```

**動畫對應**：
- 🎬 E7：從零開始 ↔ 自組織
- 🎬 E5：自我對弈 ↔ 不動點收斂
- 🎬 E12：訓練課程 ↔ 課程學習

---

## 訓練曲線分析

### Elo 成長曲線

```
Elo Rating
    │
5000├                              ╭────────
    │                         ╭────╯
4000├                    ╭────╯
    │               ╭────╯
3000├          ╭────╯
    │     ╭────╯
2000├╭────╯
    │
1000├
    └────────────────────────────────────────
    0    10    20    30    40    50    60    小時
```

**觀察**：
- 前 10 小時：發現基本規則
- 10-20 小時：發現定式
- 20-40 小時：超越人類
- 40 小時後：持續緩慢進步

**動畫對應**：
- 🎬 E6：棋力曲線 ↔ S 曲線成長
- 🎬 F8：湧現能力 ↔ 相變

---

## 延伸閱讀

- [KataGo 的關鍵創新](../katago-innovations) — 如何用更少資源達到更強棋力
- [概念速查表](../concepts/) — 109 個動畫概念的完整列表
- [30 分鐘跑起第一個圍棋 AI](../../hands-on/) — 動手實作
