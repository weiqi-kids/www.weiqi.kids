---
sidebar_position: 2
title: KataGo Source Code Guide
description: KataGo code structure, core modules, and architecture design
---

# KataGo Source Code Guide

This article helps you understand KataGo's code structure, suitable for engineers who want to dive deeper or contribute code.

---

## Getting the Source Code

```bash
git clone https://github.com/lightvector/KataGo.git
cd KataGo
```

---

## Directory Structure

```
KataGo/
├── cpp/                    # C++ core engine
│   ├── main.cpp            # Main entry point
│   ├── command/            # Command handlers
│   ├── core/               # Core utilities
│   ├── game/               # Go rules
│   ├── search/             # MCTS search
│   ├── neuralnet/          # Neural network inference
│   ├── dataio/             # Data I/O
│   └── tests/              # Unit tests
│
├── python/                 # Python training code
│   ├── train.py            # Training main program
│   ├── model.py            # Network architecture definition
│   ├── data/               # Data processing
│   └── configs/            # Training configs
│
└── docs/                   # Documentation
```

---

## Core Module Analysis

### 1. game/ — Go Rules

Complete implementation of Go rules.

#### board.h / board.cpp

```cpp
// Board state representation
class Board {
public:
    static constexpr int MAX_BOARD_SIZE = 19;

    // Board state
    Color colors[MAX_ARR_SIZE];  // Color at each position
    Chain chains[MAX_ARR_SIZE];  // Chain information

    // Core operations
    bool playMove(Loc loc, Player pla);  // Play a move
    bool isLegal(Loc loc, Player pla);   // Check legality
    void calculateArea(Color* area);      // Calculate territory
};
```

**Animation correspondence**:
- A2 Lattice model: Board data structure
- A6 Connected region: Chain representation
- A7 Liberty calculation: Liberty tracking

#### rules.h / rules.cpp

```cpp
// Multi-rule support
struct Rules {
    enum KoRule { SIMPLE_KO, POSITIONAL_KO, SITUATIONAL_KO };
    enum ScoringRule { TERRITORY_SCORING, AREA_SCORING };
    enum TaxRule { NO_TAX, TAX_SEKI, TAX_ALL };

    KoRule koRule;
    ScoringRule scoringRule;
    TaxRule taxRule;
    float komi;

    // Rule name mapping
    static Rules parseRules(const std::string& name);
};
```

Supported rules:
- `chinese`: Chinese rules (area scoring)
- `japanese`: Japanese rules (territory scoring)
- `korean`: Korean rules
- `aga`: American rules
- `tromp-taylor`: Tromp-Taylor rules

---

### 2. search/ — MCTS Search

Monte Carlo Tree Search implementation.

#### search.h / search.cpp

```cpp
class Search {
public:
    // Core search
    void runWholeSearch(Player pla);

    // MCTS steps
    void selectNode();           // Select node
    void expandNode();           // Expand node
    void evaluateNode();         // Neural network evaluation
    void backpropValue();        // Backpropagation

    // Get results
    Loc getChosenMove();
    std::vector<MoveInfo> getSortedMoveInfos();
};
```

**Animation correspondence**:
- C5 MCTS four steps: Corresponds to select → expand → evaluate → backprop
- E4 PUCT formula: Implemented in `selectNode()`

#### searchparams.h

```cpp
struct SearchParams {
    // Search control
    int64_t maxVisits;          // Maximum visits
    double maxTime;             // Maximum time

    // PUCT parameters
    double cpuctExploration;    // Exploration constant
    double cpuctBase;

    // Virtual loss
    int virtualLoss;

    // Root noise
    double rootNoiseEnabled;
    double rootDirichletAlpha;
};
```

---

### 3. neuralnet/ — Neural Network Inference

Neural network inference engine.

#### nninputs.h / nninputs.cpp

```cpp
// Neural network input features
class NNInputs {
public:
    // Feature planes
    static constexpr int NUM_FEATURES = 22;

    // Fill features
    static void fillFeatures(
        const Board& board,
        const BoardHistory& hist,
        float* features
    );
};
```

Input features include:
- Black stone positions, White stone positions
- Liberty counts (1, 2, 3+)
- History moves
- Rule encoding

**Animation correspondence**:
- A10 History stacking: Multi-frame input
- A11 Legal move mask: Forbidden move filtering

#### nneval.h / nneval.cpp

```cpp
// Neural network evaluation result
struct NNOutput {
    // Policy output (362 positions, including pass)
    float policyProbs[NNPos::MAX_NN_POLICY_SIZE];

    // Value output
    float winProb;       // Win probability
    float lossProb;      // Loss probability
    float noResultProb;  // Draw probability

    // Auxiliary outputs
    float scoreMean;     // Score prediction
    float scoreStdev;    // Score standard deviation
    float lead;          // Lead in points

    // Territory prediction
    float ownership[NNPos::MAX_BOARD_AREA];
};
```

**Animation correspondence**:
- E1 Policy network: policyProbs
- E2 Value network: winProb, scoreMean
- E3 Dual-head network: Multi-head output design

---

### 4. command/ — Command Handlers

Implementation of different running modes.

#### gtp.cpp

GTP (Go Text Protocol) mode implementation:

```cpp
void MainCmds::gtp(const std::vector<std::string>& args) {
    // Command parsing and execution
    while(true) {
        std::string line;
        std::getline(std::cin, line);

        if(line == "name") {
            respond("KataGo");
        }
        else if(line.find("play") == 0) {
            // Handle play command
        }
        else if(line.find("genmove") == 0) {
            // Execute search and return best move
        }
        // ... other commands
    }
}
```

#### analysis.cpp

Analysis Engine implementation:

```cpp
void MainCmds::analysis(const std::vector<std::string>& args) {
    while(true) {
        // Read JSON request
        std::string line;
        std::getline(std::cin, line);
        json query = json::parse(line);

        // Setup board state
        Board board = setupBoard(query);

        // Execute analysis
        Search search(...);
        search.runWholeSearch();

        // Output JSON response
        json response = formatResponse(search);
        std::cout << response.dump() << std::endl;
    }
}
```

---

## Python Training Code

### model.py — Network Architecture

```python
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Initial convolution
        self.initial_conv = nn.Conv2d(
            in_channels=config.input_features,
            out_channels=config.trunk_channels,
            kernel_size=3, padding=1
        )

        # Residual tower
        self.trunk = nn.ModuleList([
            ResidualBlock(config.trunk_channels)
            for _ in range(config.num_blocks)
        ])

        # Output heads
        self.policy_head = PolicyHead(config)
        self.value_head = ValueHead(config)
        self.ownership_head = OwnershipHead(config)

    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)

        # Residual tower
        for block in self.trunk:
            x = block(x)

        # Multi-head output
        policy = self.policy_head(x)
        value = self.value_head(x)
        ownership = self.ownership_head(x)

        return policy, value, ownership
```

**Animation correspondence**:
- D9 Convolution operation: Conv2d
- D12 Residual connection: ResidualBlock
- E11 Residual tower: trunk structure

### train.py — Training Loop

```python
def train_step(model, optimizer, batch):
    # Forward pass
    policy_pred, value_pred, ownership_pred = model(batch.inputs)

    # Compute loss
    policy_loss = cross_entropy(policy_pred, batch.policy_target)
    value_loss = mse_loss(value_pred, batch.value_target)
    ownership_loss = mse_loss(ownership_pred, batch.ownership_target)

    total_loss = policy_loss + value_loss + ownership_loss

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()
```

**Animation correspondence**:
- D3 Forward propagation: model(batch.inputs)
- D13 Backward propagation: total_loss.backward()
- K3 Adam: optimizer.step()

---

## Key Algorithm Implementation

### PUCT Selection Formula

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

### Virtual Loss

```cpp
// Prevent multiple threads selecting same node
void Search::applyVirtualLoss(SearchNode* node) {
    node->virtualLoss += params.virtualLoss;
}

void Search::removeVirtualLoss(SearchNode* node) {
    node->virtualLoss -= params.virtualLoss;
}
```

**Animation correspondence**:
- C9 Virtual loss: Parallel search technique

---

## Compilation & Debugging

### Compilation (Debug Mode)

```bash
cd cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=OPENCL -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

### Unit Tests

```bash
./katago runtests
```

### Debugging Tips

```cpp
// Enable detailed logging
#define SEARCH_DEBUG 1

// Add breakpoint in search
if(node->visits > 1000) {
    // Set breakpoint to check search state
}
```

---

## Further Reading

- [KataGo Training Mechanism](../training) — Complete training process
- [Contributing to Open Source](../contributing) — Contribution guide
- [Concept Quick Reference](/docs/animations/) — 109 concept mappings
