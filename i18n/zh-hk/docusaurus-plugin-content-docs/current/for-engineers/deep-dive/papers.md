---
sidebar_position: 11
title: 關鍵論文導讀
description: AlphaGo、AlphaZero、KataGo 等圍棋 AI 里程碑論文嘅重點解析
---

# 關鍵論文導讀

本文整理圍棋 AI 發展史上最重要嘅論文，提供快速理解嘅摘要同技術要點。

---

## 論文總覽

### 時間軸

```
2006  Coulom - MCTS 首次應用於圍棋
2016  Silver et al. - AlphaGo（Nature）
2017  Silver et al. - AlphaGo Zero（Nature）
2017  Silver et al. - AlphaZero
2019  Wu - KataGo
2020+ 各種改進同應用
```

### 閱讀建議

| 目標 | 建議論文 |
|------|---------|
| 了解基礎 | AlphaGo (2016) |
| 理解自我對弈 | AlphaGo Zero (2017) |
| 了解通用方法 | AlphaZero (2017) |
| 實作參考 | KataGo (2019) |

---

## 1. MCTS 嘅誕生（2006）

### 論文資訊

```
標題：Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search
作者：Rémi Coulom
發表：Computers and Games 2006
```

### 核心貢獻

首次將蒙地卡羅方法系統性地應用於圍棋：

```
之前：純隨機模擬，冇樹結構
之後：建構搜索樹 + UCB 選擇 + 回傳統計
```

### 關鍵概念

#### UCB1 公式

```
選擇分數 = 平均勝率 + C × √(ln(N) / n)

其中：
- N：父節點訪問次數
- n：子節點訪問次數
- C：探索常數
```

#### MCTS 四步驟

```
1. Selection：用 UCB 揀節點
2. Expansion：展開新節點
3. Simulation：隨機模擬到終局
4. Backpropagation：回傳勝負
```

### 影響

- 令圍棋 AI 達到業餘段位水平
- 成為之後所有圍棋 AI 嘅基礎
- UCB 概念影響咗 PUCT 嘅發展

---

## 2. AlphaGo（2016）

### 論文資訊

```
標題：Mastering the game of Go with deep neural networks and tree search
作者：Silver, D., Huang, A., Maddison, C.J., et al.
發表：Nature, 2016
DOI：10.1038/nature16961
```

### 核心貢獻

**首次結合深度學習同 MCTS**，擊敗人類世界冠軍。

### 系統架構

```
┌─────────────────────────────────────────────┐
│              AlphaGo 架構                    │
├─────────────────────────────────────────────┤
│                                             │
│   Policy Network (SL)                       │
│   ├── 輸入：棋盤狀態（48 個特徵平面）        │
│   ├── 架構：13 層 CNN                       │
│   ├── 輸出：361 個位置嘅機率                │
│   └── 訓練：3000 萬人類棋譜                 │
│                                             │
│   Policy Network (RL)                       │
│   ├── 由 SL Policy 初始化                   │
│   └── 自我對弈強化學習                      │
│                                             │
│   Value Network                             │
│   ├── 輸入：棋盤狀態                        │
│   ├── 輸出：單一勝率值                      │
│   └── 訓練：自我對弈產生嘅局面              │
│                                             │
│   MCTS                                      │
│   ├── 用 Policy Network 引導搜索            │
│   └── 用 Value Network + Rollout 評估       │
│                                             │
└─────────────────────────────────────────────┘
```

### 技術要點

#### 1. 監督學習 Policy Network

```python
# 輸入特徵（48 個平面）
- 己方棋子位置
- 對方棋子位置
- 氣嘅數量
- 提子後嘅狀態
- 合法手位置
- 最近幾手嘅位置
...
```

#### 2. 強化學習改進

```
SL Policy → 自我對弈 → RL Policy

RL Policy 比 SL Policy 強約 80% 勝率
```

#### 3. Value Network 訓練

```
防止過擬合嘅關鍵：
- 由每盤棋淨係攞一個位置
- 避免相似局面重複出現
```

#### 4. MCTS 整合

```
葉節點評估 = 0.5 × Value Network + 0.5 × Rollout

Rollout 用快速 Policy Network（準確率較低但速度快）
```

### 關鍵數據

| 項目 | 數值 |
|------|------|
| SL Policy 準確率 | 57% |
| RL Policy 對 SL Policy 勝率 | 80% |
| 訓練 GPU | 176 |
| 對局 GPU | 48 TPU |

---

## 3. AlphaGo Zero（2017）

### 論文資訊

```
標題：Mastering the game of Go without human knowledge
作者：Silver, D., Schrittwieser, J., Simonyan, K., et al.
發表：Nature, 2017
DOI：10.1038/nature24270
```

### 核心貢獻

**完全唔使人類棋譜**，由零開始自我學習。

### 同 AlphaGo 嘅差異

| 面向 | AlphaGo | AlphaGo Zero |
|------|---------|--------------|
| 人類棋譜 | 需要 | **唔使** |
| 網絡數量 | 4 個 | **1 個雙頭** |
| 輸入特徵 | 48 平面 | **17 平面** |
| Rollout | 用 | **唔用** |
| 殘差網絡 | 冇 | **有** |
| 訓練時間 | 幾個月 | **3 日** |

### 關鍵創新

#### 1. 單一雙頭網絡

```
              輸入（17 平面）
                   │
              ┌────┴────┐
              │ 殘差塔   │
              │ (19 or  │
              │  39 層) │
              └────┬────┘
           ┌──────┴──────┐
           │             │
        Policy         Value
        (361)          (1)
```

#### 2. 簡化輸入特徵

```python
# 淨係需要 17 個特徵平面
features = [
    current_player_stones,      # 己方棋子
    opponent_stones,            # 對方棋子
    history_1_player,           # 歷史狀態 1
    history_1_opponent,
    ...                         # 歷史狀態 2-7
    color_to_play               # 輪到邊個
]
```

#### 3. 純 Value Network 評估

```
唔再用 Rollout
葉節點評估 = Value Network 輸出

更簡潔、更快速
```

#### 4. 訓練流程

```
初始化隨機網絡
    │
    ▼
┌─────────────────────────────┐
│  自我對弈產生棋譜           │ ←─┐
└──────────────┬──────────────┘   │
               │                   │
               ▼                   │
┌─────────────────────────────┐   │
│  訓練神經網絡               │   │
│  - Policy: 最小化交叉熵      │   │
│  - Value: 最小化 MSE        │   │
└──────────────┬──────────────┘   │
               │                   │
               ▼                   │
┌─────────────────────────────┐   │
│  評估新網絡                 │   │
│  若較強則替換               │───┘
└─────────────────────────────┘
```

### 學習曲線

```
訓練時間    Elo
─────────────────
3 個鐘      初學者
24 個鐘     超越 AlphaGo Lee
72 個鐘     超越 AlphaGo Master
```

---

## 4. AlphaZero（2017）

### 論文資訊

```
標題：Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
作者：Silver, D., Hubert, T., Schrittwieser, J., et al.
發表：arXiv:1712.01815 (後發表於 Science, 2018)
```

### 核心貢獻

**通用化**：同一演算法應用於圍棋、西洋棋、將棋。

### 通用架構

```
輸入編碼（遊戲特定）→ 殘差網絡（通用）→ 雙頭輸出（通用）
```

### 跨遊戲適應

| 遊戲 | 輸入平面 | 動作空間 | 訓練時間 |
|------|---------|---------|---------|
| 圍棋 | 17 | 362 | 40 日 |
| 西洋棋 | 119 | 4672 | 9 個鐘 |
| 將棋 | 362 | 11259 | 12 個鐘 |

### MCTS 改進

#### PUCT 公式

```
選擇分數 = Q(s,a) + c(s) × P(s,a) × √N(s) / (1 + N(s,a))

c(s) = log((1 + N(s) + c_base) / c_base) + c_init
```

#### 探索噪聲

```python
# 根節點加入 Dirichlet 噪聲
P(s,a) = (1 - ε) × p_a + ε × η_a

η ~ Dir(α)
α = 0.03（圍棋）, 0.3（西洋棋）, 0.15（將棋）
```

---

## 5. KataGo（2019）

### 論文資訊

```
標題：Accelerating Self-Play Learning in Go
作者：David J. Wu
發表：arXiv:1902.10565
```

### 核心貢獻

**50 倍效率提升**，令個人開發者都可以訓練強大嘅圍棋 AI。

### 關鍵創新

#### 1. 輔助訓練目標

```
總損失 = Policy Loss + Value Loss +
         Score Loss + Ownership Loss + ...

輔助目標令網絡更快收斂
```

#### 2. 全局特徵

```python
# 全局池化層
global_features = global_avg_pool(conv_features)
# 同局部特徵結合
combined = concat(conv_features, broadcast(global_features))
```

#### 3. Playout Cap 隨機化

```
傳統：每次搜索固定 N 次
KataGo：N 由某個分佈隨機取樣

令網絡學識喺各種搜索深度下都表現良好
```

#### 4. 漸進式棋盤大小

```python
if training_step < 1000000:
    board_size = random.choice([9, 13, 19])
else:
    board_size = 19
```

### 效率比較

| 指標 | AlphaZero | KataGo |
|------|-----------|--------|
| 達到超人水平嘅 GPU 日 | 5000 | **100** |
| 效率提升 | 基準 | **50 倍** |

---

## 6. 延伸論文

### MuZero（2020）

```
標題：Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model
貢獻：學習環境動態模型，唔使遊戲規則
```

### EfficientZero（2021）

```
標題：Mastering Atari Games with Limited Data
貢獻：樣本效率大幅提升
```

### Gumbel AlphaZero（2022）

```
標題：Policy Improvement by Planning with Gumbel
貢獻：改進嘅策略改進方法
```

---

## 論文閱讀建議

### 入門順序

```
1. AlphaGo (2016) - 理解基本架構
2. AlphaGo Zero (2017) - 理解自我對弈
3. KataGo (2019) - 理解實作細節
```

### 進階順序

```
4. AlphaZero (2017) - 通用化
5. MuZero (2020) - 學習世界模型
6. MCTS 原始論文 - 理解基礎
```

### 閱讀技巧

1. **先睇摘要同結論**：快速掌握核心貢獻
2. **睇圖表**：理解整體架構
3. **睇方法部分**：理解技術細節
4. **睇附錄**：搵實作細節同超參數

---

## 資源連結

### 論文 PDF

| 論文 | 連結 |
|------|------|
| AlphaGo | [Nature](https://www.nature.com/articles/nature16961) |
| AlphaGo Zero | [Nature](https://www.nature.com/articles/nature24270) |
| AlphaZero | [Science](https://www.science.org/doi/10.1126/science.aar6404) |
| KataGo | [arXiv](https://arxiv.org/abs/1902.10565) |

### 開源實作

| 專案 | 連結 |
|------|------|
| KataGo | [GitHub](https://github.com/lightvector/KataGo) |
| Leela Zero | [GitHub](https://github.com/leela-zero/leela-zero) |
| MiniGo | [GitHub](https://github.com/tensorflow/minigo) |

---

## 延伸閱讀

- [神經網絡架構詳解](../neural-network) — 深入理解網絡設計
- [MCTS 實作細節](../mcts-implementation) — 搜索演算法實作
- [KataGo 訓練機制解析](../training) — 訓練流程詳解
