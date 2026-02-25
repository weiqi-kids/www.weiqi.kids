---
sidebar_position: 2
title: KataGo 原始碼導讀
description: KataGo 程式碼結構、核心模組與架構設計
---

# KataGo 原始碼導讀

本文帶你了解 KataGo 的程式碼結構，適合想深入研究或貢獻程式碼的工程師。

---

## 取得原始碼

```bash
git clone https://github.com/lightvector/KataGo.git
cd KataGo
```

---

## 目錄結構

```
KataGo/
├── cpp/                    # C++ 核心引擎
│   ├── main.cpp            # 主程式入口
│   ├── command/            # 指令處理
│   ├── core/               # 核心工具
│   ├── game/               # 圍棋規則
│   ├── search/             # MCTS 搜索
│   ├── neuralnet/          # 神經網路推理
│   ├── dataio/             # 資料 I/O
│   └── tests/              # 單元測試
│
├── python/                 # Python 訓練程式碼
│   ├── train.py            # 訓練主程式
│   ├── model.py            # 網路架構定義
│   ├── data/               # 資料處理
│   └── configs/            # 訓練設定
│
└── docs/                   # 文件
```

---

## 核心模組解析

### 1. game/ — 圍棋規則

圍棋規則的完整實作。

#### board.h / board.cpp

```cpp
// 棋盤狀態表示
class Board {
public:
    static constexpr int MAX_BOARD_SIZE = 19;

    // 棋盤狀態
    Color colors[MAX_ARR_SIZE];  // 每個位置的顏色
    Chain chains[MAX_ARR_SIZE];  // 棋串資訊

    // 核心操作
    bool playMove(Loc loc, Player pla);  // 下一步棋
    bool isLegal(Loc loc, Player pla);   // 判斷合法性
    void calculateArea(Color* area);      // 計算領地
};
```

**動畫對應**：
- 🎬 A2 晶格模型：棋盤的資料結構
- 🎬 A6 連通區域：棋串（Chain）的表示
- 🎬 A7 氣的計算：liberty 的追蹤

#### rules.h / rules.cpp

```cpp
// 多規則支援
struct Rules {
    enum KoRule { SIMPLE_KO, POSITIONAL_KO, SITUATIONAL_KO };
    enum ScoringRule { TERRITORY_SCORING, AREA_SCORING };
    enum TaxRule { NO_TAX, TAX_SEKI, TAX_ALL };

    KoRule koRule;
    ScoringRule scoringRule;
    TaxRule taxRule;
    float komi;

    // 規則名稱對應
    static Rules parseRules(const std::string& name);
};
```

支援的規則：
- `chinese`：中國規則（數子）
- `japanese`：日本規則（數目）
- `korean`：韓國規則
- `aga`：美國規則
- `tromp-taylor`：Tromp-Taylor 規則

---

### 2. search/ — MCTS 搜索

蒙地卡羅樹搜索的實作。

#### search.h / search.cpp

```cpp
class Search {
public:
    // 核心搜索
    void runWholeSearch(Player pla);

    // MCTS 步驟
    void selectNode();           // 選擇節點
    void expandNode();           // 擴展節點
    void evaluateNode();         // 神經網路評估
    void backpropValue();        // 回傳更新

    // 結果取得
    Loc getChosenMove();
    std::vector<MoveInfo> getSortedMoveInfos();
};
```

**動畫對應**：
- 🎬 C5 MCTS 四步驟：對應 select → expand → evaluate → backprop
- 🎬 E4 PUCT 公式：在 `selectNode()` 中實作

#### searchparams.h

```cpp
struct SearchParams {
    // 搜索控制
    int64_t maxVisits;          // 最大訪問次數
    double maxTime;             // 最大時間

    // PUCT 參數
    double cpuctExploration;    // 探索常數
    double cpuctBase;

    // 虛擬損失
    int virtualLoss;

    // 根節點雜訊
    double rootNoiseEnabled;
    double rootDirichletAlpha;
};
```

---

### 3. neuralnet/ — 神經網路推理

神經網路的推理引擎。

#### nninputs.h / nninputs.cpp

```cpp
// 神經網路輸入特徵
class NNInputs {
public:
    // 特徵平面
    static constexpr int NUM_FEATURES = 22;

    // 填充特徵
    static void fillFeatures(
        const Board& board,
        const BoardHistory& hist,
        float* features
    );
};
```

輸入特徵包括：
- 黑子位置、白子位置
- 氣數（1, 2, 3+）
- 歷史棋步
- 規則編碼

**動畫對應**：
- 🎬 A10 歷史堆疊：多幀輸入
- 🎬 A11 合法手遮罩：禁手過濾

#### nneval.h / nneval.cpp

```cpp
// 神經網路評估結果
struct NNOutput {
    // Policy 輸出（362 個位置，含 pass）
    float policyProbs[NNPos::MAX_NN_POLICY_SIZE];

    // Value 輸出
    float winProb;       // 勝率
    float lossProb;      // 敗率
    float noResultProb;  // 和棋率

    // 輔助輸出
    float scoreMean;     // 目數預測
    float scoreStdev;    // 目數標準差
    float lead;          // 領先目數

    // 領地預測
    float ownership[NNPos::MAX_BOARD_AREA];
};
```

**動畫對應**：
- 🎬 E1 策略網路：policyProbs
- 🎬 E2 價值網路：winProb, scoreMean
- 🎬 E3 雙頭網路：多輸出頭設計

---

### 4. command/ — 指令處理

不同運行模式的實作。

#### gtp.cpp

GTP（Go Text Protocol）模式的實作：

```cpp
void MainCmds::gtp(const std::vector<std::string>& args) {
    // 指令解析與執行
    while(true) {
        std::string line;
        std::getline(std::cin, line);

        if(line == "name") {
            respond("KataGo");
        }
        else if(line.find("play") == 0) {
            // 處理下棋指令
        }
        else if(line.find("genmove") == 0) {
            // 執行搜索並回傳最佳下法
        }
        // ... 其他指令
    }
}
```

#### analysis.cpp

Analysis Engine 的實作：

```cpp
void MainCmds::analysis(const std::vector<std::string>& args) {
    while(true) {
        // 讀取 JSON 請求
        std::string line;
        std::getline(std::cin, line);
        json query = json::parse(line);

        // 建立棋盤狀態
        Board board = setupBoard(query);

        // 執行分析
        Search search(...);
        search.runWholeSearch();

        // 輸出 JSON 回應
        json response = formatResponse(search);
        std::cout << response.dump() << std::endl;
    }
}
```

---

## Python 訓練程式碼

### model.py — 網路架構

```python
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 初始卷積
        self.initial_conv = nn.Conv2d(
            in_channels=config.input_features,
            out_channels=config.trunk_channels,
            kernel_size=3, padding=1
        )

        # 殘差塔
        self.trunk = nn.ModuleList([
            ResidualBlock(config.trunk_channels)
            for _ in range(config.num_blocks)
        ])

        # 輸出頭
        self.policy_head = PolicyHead(config)
        self.value_head = ValueHead(config)
        self.ownership_head = OwnershipHead(config)

    def forward(self, x):
        # 初始卷積
        x = self.initial_conv(x)

        # 殘差塔
        for block in self.trunk:
            x = block(x)

        # 多頭輸出
        policy = self.policy_head(x)
        value = self.value_head(x)
        ownership = self.ownership_head(x)

        return policy, value, ownership
```

**動畫對應**：
- 🎬 D9 卷積運算：Conv2d
- 🎬 D12 殘差連接：ResidualBlock
- 🎬 E11 殘差塔：trunk 結構

### train.py — 訓練循環

```python
def train_step(model, optimizer, batch):
    # 前向傳播
    policy_pred, value_pred, ownership_pred = model(batch.inputs)

    # 計算損失
    policy_loss = cross_entropy(policy_pred, batch.policy_target)
    value_loss = mse_loss(value_pred, batch.value_target)
    ownership_loss = mse_loss(ownership_pred, batch.ownership_target)

    total_loss = policy_loss + value_loss + ownership_loss

    # 反向傳播
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()
```

**動畫對應**：
- 🎬 D3 前向傳播：model(batch.inputs)
- 🎬 D13 反向傳播：total_loss.backward()
- 🎬 K3 Adam：optimizer.step()

---

## 關鍵演算法實作

### PUCT 選擇公式

```cpp
// search.cpp
double Search::getPUCTScore(const SearchNode* node, int moveIdx) {
    double Q = node->getChildValue(moveIdx);
    double P = node->getChildPolicy(moveIdx);
    double N_parent = node->visits;
    double N_child = node->getChildVisits(moveIdx);

    double exploration = params.cpuctExploration;
    double cpuct = exploration * sqrt(N_parent) / (1.0 + N_child);

    return Q + cpuct * P;
}
```

### 虛擬損失

```cpp
// 避免多執行緒選擇相同節點
void Search::applyVirtualLoss(SearchNode* node) {
    node->virtualLoss += params.virtualLoss;
}

void Search::removeVirtualLoss(SearchNode* node) {
    node->virtualLoss -= params.virtualLoss;
}
```

**動畫對應**：
- 🎬 C9 虛擬損失：並行搜索的技巧

---

## 編譯與除錯

### 編譯（Debug 模式）

```bash
cd cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=OPENCL -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

### 單元測試

```bash
./katago runtests
```

### 除錯技巧

```cpp
// 啟用詳細日誌
#define SEARCH_DEBUG 1

// 在搜索中加入斷點
if(node->visits > 1000) {
    // 設置斷點檢查搜索狀態
}
```

---

## 延伸閱讀

- [KataGo 訓練機制解析](../training) — 完整訓練流程
- [參與開源社群](../contributing) — 貢獻指南
- [概念速查表](/docs/animations/) — 109 個概念對照
