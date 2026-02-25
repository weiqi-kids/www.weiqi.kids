---
sidebar_position: 2
title: KataGo 源代码导读
description: KataGo 代码结构、核心模块与架构设计
---

# KataGo 源代码导读

本文带你了解 KataGo 的代码结构，适合想深入研究或贡献代码的工程师。

---

## 获取源代码

```bash
git clone https://github.com/lightvector/KataGo.git
cd KataGo
```

---

## 目录结构

```
KataGo/
├── cpp/                    # C++ 核心引擎
│   ├── main.cpp            # 主程序入口
│   ├── command/            # 指令处理
│   ├── core/               # 核心工具
│   ├── game/               # 围棋规则
│   ├── search/             # MCTS 搜索
│   ├── neuralnet/          # 神经网络推理
│   ├── dataio/             # 数据 I/O
│   └── tests/              # 单元测试
│
├── python/                 # Python 训练代码
│   ├── train.py            # 训练主程序
│   ├── model.py            # 网络架构定义
│   ├── data/               # 数据处理
│   └── configs/            # 训练配置
│
└── docs/                   # 文档
```

---

## 核心模块解析

### 1. game/ — 围棋规则

围棋规则的完整实现。

#### board.h / board.cpp

```cpp
// 棋盘状态表示
class Board {
public:
    static constexpr int MAX_BOARD_SIZE = 19;

    // 棋盘状态
    Color colors[MAX_ARR_SIZE];  // 每个位置的颜色
    Chain chains[MAX_ARR_SIZE];  // 棋串信息

    // 核心操作
    bool playMove(Loc loc, Player pla);  // 下一步棋
    bool isLegal(Loc loc, Player pla);   // 判断合法性
    void calculateArea(Color* area);      // 计算领地
};
```

**动画对应**：
- A2 晶格模型：棋盘的数据结构
- A6 连通区域：棋串（Chain）的表示
- A7 气的计算：liberty 的追踪

#### rules.h / rules.cpp

```cpp
// 多规则支持
struct Rules {
    enum KoRule { SIMPLE_KO, POSITIONAL_KO, SITUATIONAL_KO };
    enum ScoringRule { TERRITORY_SCORING, AREA_SCORING };
    enum TaxRule { NO_TAX, TAX_SEKI, TAX_ALL };

    KoRule koRule;
    ScoringRule scoringRule;
    TaxRule taxRule;
    float komi;

    // 规则名称对应
    static Rules parseRules(const std::string& name);
};
```

支持的规则：
- `chinese`：中国规则（数子）
- `japanese`：日本规则（数目）
- `korean`：韩国规则
- `aga`：美国规则
- `tromp-taylor`：Tromp-Taylor 规则

---

### 2. search/ — MCTS 搜索

蒙特卡洛树搜索的实现。

#### search.h / search.cpp

```cpp
class Search {
public:
    // 核心搜索
    void runWholeSearch(Player pla);

    // MCTS 步骤
    void selectNode();           // 选择节点
    void expandNode();           // 扩展节点
    void evaluateNode();         // 神经网络评估
    void backpropValue();        // 回传更新

    // 结果获取
    Loc getChosenMove();
    std::vector<MoveInfo> getSortedMoveInfos();
};
```

**动画对应**：
- C5 MCTS 四步骤：对应 select → expand → evaluate → backprop
- E4 PUCT 公式：在 `selectNode()` 中实现

#### searchparams.h

```cpp
struct SearchParams {
    // 搜索控制
    int64_t maxVisits;          // 最大访问次数
    double maxTime;             // 最大时间

    // PUCT 参数
    double cpuctExploration;    // 探索常数
    double cpuctBase;

    // 虚拟损失
    int virtualLoss;

    // 根节点噪声
    double rootNoiseEnabled;
    double rootDirichletAlpha;
};
```

---

### 3. neuralnet/ — 神经网络推理

神经网络的推理引擎。

#### nninputs.h / nninputs.cpp

```cpp
// 神经网络输入特征
class NNInputs {
public:
    // 特征平面
    static constexpr int NUM_FEATURES = 22;

    // 填充特征
    static void fillFeatures(
        const Board& board,
        const BoardHistory& hist,
        float* features
    );
};
```

输入特征包括：
- 黑子位置、白子位置
- 气数（1, 2, 3+）
- 历史棋步
- 规则编码

**动画对应**：
- A10 历史堆叠：多帧输入
- A11 合法手遮罩：禁手过滤

#### nneval.h / nneval.cpp

```cpp
// 神经网络评估结果
struct NNOutput {
    // Policy 输出（362 个位置，含 pass）
    float policyProbs[NNPos::MAX_NN_POLICY_SIZE];

    // Value 输出
    float winProb;       // 胜率
    float lossProb;      // 败率
    float noResultProb;  // 和棋率

    // 辅助输出
    float scoreMean;     // 目数预测
    float scoreStdev;    // 目数标准差
    float lead;          // 领先目数

    // 领地预测
    float ownership[NNPos::MAX_BOARD_AREA];
};
```

**动画对应**：
- E1 策略网络：policyProbs
- E2 价值网络：winProb, scoreMean
- E3 双头网络：多输出头设计

---

### 4. command/ — 指令处理

不同运行模式的实现。

#### gtp.cpp

GTP（Go Text Protocol）模式的实现：

```cpp
void MainCmds::gtp(const std::vector<std::string>& args) {
    // 指令解析与执行
    while(true) {
        std::string line;
        std::getline(std::cin, line);

        if(line == "name") {
            respond("KataGo");
        }
        else if(line.find("play") == 0) {
            // 处理下棋指令
        }
        else if(line.find("genmove") == 0) {
            // 执行搜索并返回最佳下法
        }
        // ... 其他指令
    }
}
```

#### analysis.cpp

Analysis Engine 的实现：

```cpp
void MainCmds::analysis(const std::vector<std::string>& args) {
    while(true) {
        // 读取 JSON 请求
        std::string line;
        std::getline(std::cin, line);
        json query = json::parse(line);

        // 建立棋盘状态
        Board board = setupBoard(query);

        // 执行分析
        Search search(...);
        search.runWholeSearch();

        // 输出 JSON 响应
        json response = formatResponse(search);
        std::cout << response.dump() << std::endl;
    }
}
```

---

## Python 训练代码

### model.py — 网络架构

```python
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 初始卷积
        self.initial_conv = nn.Conv2d(
            in_channels=config.input_features,
            out_channels=config.trunk_channels,
            kernel_size=3, padding=1
        )

        # 残差塔
        self.trunk = nn.ModuleList([
            ResidualBlock(config.trunk_channels)
            for _ in range(config.num_blocks)
        ])

        # 输出头
        self.policy_head = PolicyHead(config)
        self.value_head = ValueHead(config)
        self.ownership_head = OwnershipHead(config)

    def forward(self, x):
        # 初始卷积
        x = self.initial_conv(x)

        # 残差塔
        for block in self.trunk:
            x = block(x)

        # 多头输出
        policy = self.policy_head(x)
        value = self.value_head(x)
        ownership = self.ownership_head(x)

        return policy, value, ownership
```

**动画对应**：
- D9 卷积运算：Conv2d
- D12 残差连接：ResidualBlock
- E11 残差塔：trunk 结构

### train.py — 训练循环

```python
def train_step(model, optimizer, batch):
    # 前向传播
    policy_pred, value_pred, ownership_pred = model(batch.inputs)

    # 计算损失
    policy_loss = cross_entropy(policy_pred, batch.policy_target)
    value_loss = mse_loss(value_pred, batch.value_target)
    ownership_loss = mse_loss(ownership_pred, batch.ownership_target)

    total_loss = policy_loss + value_loss + ownership_loss

    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()
```

**动画对应**：
- D3 前向传播：model(batch.inputs)
- D13 反向传播：total_loss.backward()
- K3 Adam：optimizer.step()

---

## 关键算法实现

### PUCT 选择公式

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

### 虚拟损失

```cpp
// 避免多线程选择相同节点
void Search::applyVirtualLoss(SearchNode* node) {
    node->virtualLoss += params.virtualLoss;
}

void Search::removeVirtualLoss(SearchNode* node) {
    node->virtualLoss -= params.virtualLoss;
}
```

**动画对应**：
- C9 虚拟损失：并行搜索的技巧

---

## 编译与调试

### 编译（Debug 模式）

```bash
cd cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=OPENCL -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

### 单元测试

```bash
./katago runtests
```

### 调试技巧

```cpp
// 启用详细日志
#define SEARCH_DEBUG 1

// 在搜索中加入断点
if(node->visits > 1000) {
    // 设置断点检查搜索状态
}
```

---

## 延伸阅读

- [KataGo 训练机制解析](../training) — 完整训练流程
- [参与开源社区](../contributing) — 贡献指南
- [概念速查表](../../how-it-works/concepts/) — 109 个概念对照
