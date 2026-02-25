---
sidebar_position: 2
title: KataGoソースコード解説
description: KataGoのコード構造、コアモジュール、アーキテクチャ設計
---

# KataGoソースコード解説

本記事では、KataGoのコード構造を理解し、ソースコードを深く研究したり貢献したいエンジニアに役立つ内容を提供します。

---

## ソースコードの取得

```bash
git clone https://github.com/lightvector/KataGo.git
cd KataGo
```

---

## ディレクトリ構造

```
KataGo/
├── cpp/                    # C++コアエンジン
│   ├── main.cpp            # メインプログラムエントリ
│   ├── command/            # コマンド処理
│   ├── core/               # コアユーティリティ
│   ├── game/               # 囲碁ルール
│   ├── search/             # MCTS探索
│   ├── neuralnet/          # ニューラルネットワーク推論
│   ├── dataio/             # データI/O
│   └── tests/              # ユニットテスト
│
├── python/                 # Python訓練コード
│   ├── train.py            # 訓練メインプログラム
│   ├── model.py            # ネットワークアーキテクチャ定義
│   ├── data/               # データ処理
│   └── configs/            # 訓練設定
│
└── docs/                   # ドキュメント
```

---

## コアモジュール解析

### 1. game/ — 囲碁ルール

囲碁ルールの完全な実装。

#### board.h / board.cpp

```cpp
// 盤面状態の表現
class Board {
public:
    static constexpr int MAX_BOARD_SIZE = 19;

    // 盤面状態
    Color colors[MAX_ARR_SIZE];  // 各位置の色
    Chain chains[MAX_ARR_SIZE];  // 石群情報

    // コア操作
    bool playMove(Loc loc, Player pla);  // 一手打つ
    bool isLegal(Loc loc, Player pla);   // 合法性判定
    void calculateArea(Color* area);      // 地を計算
};
```

**アニメーション対応**：
- A2 格子モデル：盤面のデータ構造
- A6 連結領域：石群（Chain）の表現
- A7 呼吸点の計算：libertyの追跡

#### rules.h / rules.cpp

```cpp
// マルチルールサポート
struct Rules {
    enum KoRule { SIMPLE_KO, POSITIONAL_KO, SITUATIONAL_KO };
    enum ScoringRule { TERRITORY_SCORING, AREA_SCORING };
    enum TaxRule { NO_TAX, TAX_SEKI, TAX_ALL };

    KoRule koRule;
    ScoringRule scoringRule;
    TaxRule taxRule;
    float komi;

    // ルール名対応
    static Rules parseRules(const std::string& name);
};
```

サポートされるルール：
- `chinese`：中国ルール（地+石）
- `japanese`：日本ルール（地のみ）
- `korean`：韓国ルール
- `aga`：アメリカルール
- `tromp-taylor`：Tromp-Taylorルール

---

### 2. search/ — MCTS探索

モンテカルロ木探索の実装。

#### search.h / search.cpp

```cpp
class Search {
public:
    // コア探索
    void runWholeSearch(Player pla);

    // MCTSステップ
    void selectNode();           // ノード選択
    void expandNode();           // ノード展開
    void evaluateNode();         // ニューラルネットワーク評価
    void backpropValue();        // 逆伝播更新

    // 結果取得
    Loc getChosenMove();
    std::vector<MoveInfo> getSortedMoveInfos();
};
```

**アニメーション対応**：
- C5 MCTSの4ステップ：select → expand → evaluate → backpropに対応
- E4 PUCT公式：`selectNode()`内で実装

#### searchparams.h

```cpp
struct SearchParams {
    // 探索制御
    int64_t maxVisits;          // 最大訪問回数
    double maxTime;             // 最大時間

    // PUCTパラメータ
    double cpuctExploration;    // 探索定数
    double cpuctBase;

    // 仮想損失
    int virtualLoss;

    // ルートノードノイズ
    double rootNoiseEnabled;
    double rootDirichletAlpha;
};
```

---

### 3. neuralnet/ — ニューラルネットワーク推論

ニューラルネットワークの推論エンジン。

#### nninputs.h / nninputs.cpp

```cpp
// ニューラルネットワーク入力特徴
class NNInputs {
public:
    // 特徴平面
    static constexpr int NUM_FEATURES = 22;

    // 特徴を埋める
    static void fillFeatures(
        const Board& board,
        const BoardHistory& hist,
        float* features
    );
};
```

入力特徴に含まれる：
- 黒石位置、白石位置
- 呼吸点数（1, 2, 3+）
- 履歴の手
- ルールエンコーディング

**アニメーション対応**：
- A10 履歴スタッキング：マルチフレーム入力
- A11 合法手マスク：禁止手フィルタリング

#### nneval.h / nneval.cpp

```cpp
// ニューラルネットワーク評価結果
struct NNOutput {
    // Policy出力（362位置、パス含む）
    float policyProbs[NNPos::MAX_NN_POLICY_SIZE];

    // Value出力
    float winProb;       // 勝率
    float lossProb;      // 敗率
    float noResultProb;  // 引き分け率

    // 補助出力
    float scoreMean;     // 目数予測
    float scoreStdev;    // 目数標準偏差
    float lead;          // リード目数

    // 地予測
    float ownership[NNPos::MAX_BOARD_AREA];
};
```

**アニメーション対応**：
- E1 方策ネットワーク：policyProbs
- E2 価値ネットワーク：winProb, scoreMean
- E3 デュアルヘッドネットワーク：マルチ出力ヘッド設計

---

### 4. command/ — コマンド処理

異なる実行モードの実装。

#### gtp.cpp

GTP（Go Text Protocol）モードの実装：

```cpp
void MainCmds::gtp(const std::vector<std::string>& args) {
    // コマンドの解析と実行
    while(true) {
        std::string line;
        std::getline(std::cin, line);

        if(line == "name") {
            respond("KataGo");
        }
        else if(line.find("play") == 0) {
            // 着手コマンドを処理
        }
        else if(line.find("genmove") == 0) {
            // 探索を実行して最善手を返す
        }
        // ... その他のコマンド
    }
}
```

#### analysis.cpp

Analysis Engineの実装：

```cpp
void MainCmds::analysis(const std::vector<std::string>& args) {
    while(true) {
        // JSONリクエストを読み込む
        std::string line;
        std::getline(std::cin, line);
        json query = json::parse(line);

        // 盤面状態を構築
        Board board = setupBoard(query);

        // 分析を実行
        Search search(...);
        search.runWholeSearch();

        // JSONレスポンスを出力
        json response = formatResponse(search);
        std::cout << response.dump() << std::endl;
    }
}
```

---

## Python訓練コード

### model.py — ネットワークアーキテクチャ

```python
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 初期畳み込み
        self.initial_conv = nn.Conv2d(
            in_channels=config.input_features,
            out_channels=config.trunk_channels,
            kernel_size=3, padding=1
        )

        # 残差タワー
        self.trunk = nn.ModuleList([
            ResidualBlock(config.trunk_channels)
            for _ in range(config.num_blocks)
        ])

        # 出力ヘッド
        self.policy_head = PolicyHead(config)
        self.value_head = ValueHead(config)
        self.ownership_head = OwnershipHead(config)

    def forward(self, x):
        # 初期畳み込み
        x = self.initial_conv(x)

        # 残差タワー
        for block in self.trunk:
            x = block(x)

        # マルチヘッド出力
        policy = self.policy_head(x)
        value = self.value_head(x)
        ownership = self.ownership_head(x)

        return policy, value, ownership
```

**アニメーション対応**：
- D9 畳み込み演算：Conv2d
- D12 残差接続：ResidualBlock
- E11 残差タワー：trunk構造

### train.py — 訓練ループ

```python
def train_step(model, optimizer, batch):
    # 順伝播
    policy_pred, value_pred, ownership_pred = model(batch.inputs)

    # 損失を計算
    policy_loss = cross_entropy(policy_pred, batch.policy_target)
    value_loss = mse_loss(value_pred, batch.value_target)
    ownership_loss = mse_loss(ownership_pred, batch.ownership_target)

    total_loss = policy_loss + value_loss + ownership_loss

    # 逆伝播
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()
```

**アニメーション対応**：
- D3 順伝播：model(batch.inputs)
- D13 逆伝播：total_loss.backward()
- K3 Adam：optimizer.step()

---

## 重要アルゴリズム実装

### PUCT選択公式

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

### 仮想損失

```cpp
// マルチスレッドが同じノードを選択しないように
void Search::applyVirtualLoss(SearchNode* node) {
    node->virtualLoss += params.virtualLoss;
}

void Search::removeVirtualLoss(SearchNode* node) {
    node->virtualLoss -= params.virtualLoss;
}
```

**アニメーション対応**：
- C9 仮想損失：並列探索のテクニック

---

## コンパイルとデバッグ

### コンパイル（Debugモード）

```bash
cd cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=OPENCL -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

### ユニットテスト

```bash
./katago runtests
```

### デバッグテクニック

```cpp
// 詳細ログを有効化
#define SEARCH_DEBUG 1

// 探索にブレークポイントを追加
if(node->visits > 1000) {
    // ブレークポイントを設定して探索状態を確認
}
```

---

## 関連記事

- [KataGo訓練メカニズム解析](../training) — 完全な訓練フロー
- [オープンソースコミュニティへの参加](../contributing) — 貢献ガイド
- [概念クイックリファレンス](/docs/animations/) — 109の概念対照
