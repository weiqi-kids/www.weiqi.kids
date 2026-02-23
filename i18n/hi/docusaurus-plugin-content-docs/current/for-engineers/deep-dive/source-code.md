---
sidebar_position: 2
title: KataGo स्रोत कोड गाइड
description: KataGo प्रोग्राम कोड संरचना, कोर मॉड्यूल और आर्किटेक्चर डिज़ाइन
---

# KataGo स्रोत कोड गाइड

यह लेख आपको KataGo की कोड संरचना समझने में मदद करता है, गहन अध्ययन या कोड योगदान करने वाले इंजीनियरों के लिए उपयुक्त।

---

## स्रोत कोड प्राप्त करें

```bash
git clone https://github.com/lightvector/KataGo.git
cd KataGo
```

---

## डायरेक्टरी संरचना

```
KataGo/
├── cpp/                    # C++ कोर इंजन
│   ├── main.cpp            # मुख्य प्रोग्राम एंट्री
│   ├── command/            # कमांड प्रोसेसिंग
│   ├── core/               # कोर टूल्स
│   ├── game/               # गो नियम
│   ├── search/             # MCTS सर्च
│   ├── neuralnet/          # न्यूरल नेटवर्क इन्फरेंस
│   ├── dataio/             # डेटा I/O
│   └── tests/              # यूनिट टेस्ट
│
├── python/                 # Python प्रशिक्षण कोड
│   ├── train.py            # प्रशिक्षण मुख्य प्रोग्राम
│   ├── model.py            # नेटवर्क आर्किटेक्चर परिभाषा
│   ├── data/               # डेटा प्रोसेसिंग
│   └── configs/            # प्रशिक्षण कॉन्फ़िगरेशन
│
└── docs/                   # डॉक्यूमेंटेशन
```

---

## कोर मॉड्यूल विश्लेषण

### 1. game/ — गो नियम

गो नियमों का पूर्ण कार्यान्वयन।

#### board.h / board.cpp

```cpp
// बोर्ड स्थिति प्रतिनिधित्व
class Board {
public:
    static constexpr int MAX_BOARD_SIZE = 19;

    // बोर्ड स्थिति
    Color colors[MAX_ARR_SIZE];  // प्रत्येक स्थान का रंग
    Chain chains[MAX_ARR_SIZE];  // पत्थर समूह जानकारी

    // कोर ऑपरेशन
    bool playMove(Loc loc, Player pla);  // एक चाल खेलें
    bool isLegal(Loc loc, Player pla);   // वैधता जांचें
    void calculateArea(Color* area);      // क्षेत्र गणना करें
};
```

**एनिमेशन संबंध**:
- A2 जाली मॉडल: बोर्ड की डेटा संरचना
- A6 कनेक्टेड क्षेत्र: पत्थर समूह (Chain) का प्रतिनिधित्व
- A7 लिबर्टी गणना: liberty की ट्रैकिंग

#### rules.h / rules.cpp

```cpp
// मल्टी-रूल समर्थन
struct Rules {
    enum KoRule { SIMPLE_KO, POSITIONAL_KO, SITUATIONAL_KO };
    enum ScoringRule { TERRITORY_SCORING, AREA_SCORING };
    enum TaxRule { NO_TAX, TAX_SEKI, TAX_ALL };

    KoRule koRule;
    ScoringRule scoringRule;
    TaxRule taxRule;
    float komi;

    // नियम नाम मैपिंग
    static Rules parseRules(const std::string& name);
};
```

समर्थित नियम:
- `chinese`: चीनी नियम (एरिया स्कोरिंग)
- `japanese`: जापानी नियम (टेरिटरी स्कोरिंग)
- `korean`: कोरियाई नियम
- `aga`: अमेरिकी नियम
- `tromp-taylor`: Tromp-Taylor नियम

---

### 2. search/ — MCTS सर्च

मोंटे कार्लो ट्री सर्च का कार्यान्वयन।

#### search.h / search.cpp

```cpp
class Search {
public:
    // कोर सर्च
    void runWholeSearch(Player pla);

    // MCTS चरण
    void selectNode();           // नोड चयन
    void expandNode();           // नोड विस्तार
    void evaluateNode();         // न्यूरल नेटवर्क मूल्यांकन
    void backpropValue();        // बैकप्रोप अपडेट

    // परिणाम प्राप्त करें
    Loc getChosenMove();
    std::vector<MoveInfo> getSortedMoveInfos();
};
```

**एनिमेशन संबंध**:
- C5 MCTS चार चरण: select → expand → evaluate → backprop से मेल खाता है
- E4 PUCT फॉर्मूला: `selectNode()` में कार्यान्वित

#### searchparams.h

```cpp
struct SearchParams {
    // सर्च नियंत्रण
    int64_t maxVisits;          // अधिकतम विज़िट
    double maxTime;             // अधिकतम समय

    // PUCT पैरामीटर
    double cpuctExploration;    // अन्वेषण स्थिरांक
    double cpuctBase;

    // वर्चुअल लॉस
    int virtualLoss;

    // रूट नोड नॉइज़
    double rootNoiseEnabled;
    double rootDirichletAlpha;
};
```

---

### 3. neuralnet/ — न्यूरल नेटवर्क इन्फरेंस

न्यूरल नेटवर्क का इन्फरेंस इंजन।

#### nninputs.h / nninputs.cpp

```cpp
// न्यूरल नेटवर्क इनपुट फीचर्स
class NNInputs {
public:
    // फीचर प्लेन
    static constexpr int NUM_FEATURES = 22;

    // फीचर्स भरें
    static void fillFeatures(
        const Board& board,
        const BoardHistory& hist,
        float* features
    );
};
```

इनपुट फीचर्स में शामिल:
- काले पत्थर स्थिति, सफेद पत्थर स्थिति
- लिबर्टी संख्या (1, 2, 3+)
- इतिहास चालें
- नियम एनकोडिंग

**एनिमेशन संबंध**:
- A10 इतिहास स्टैकिंग: मल्टी-फ्रेम इनपुट
- A11 वैध चाल मास्क: निषिद्ध चाल फ़िल्टरिंग

#### nneval.h / nneval.cpp

```cpp
// न्यूरल नेटवर्क मूल्यांकन परिणाम
struct NNOutput {
    // Policy आउटपुट (362 स्थान, pass सहित)
    float policyProbs[NNPos::MAX_NN_POLICY_SIZE];

    // Value आउटपुट
    float winProb;       // जीत दर
    float lossProb;      // हार दर
    float noResultProb;  // ड्रॉ दर

    // सहायक आउटपुट
    float scoreMean;     // अंक पूर्वानुमान
    float scoreStdev;    // अंक मानक विचलन
    float lead;          // बढ़त अंक

    // क्षेत्र पूर्वानुमान
    float ownership[NNPos::MAX_BOARD_AREA];
};
```

**एनिमेशन संबंध**:
- E1 पॉलिसी नेटवर्क: policyProbs
- E2 वैल्यू नेटवर्क: winProb, scoreMean
- E3 ड्यूअल-हेड नेटवर्क: मल्टी आउटपुट हेड डिज़ाइन

---

### 4. command/ — कमांड प्रोसेसिंग

विभिन्न रनिंग मोड का कार्यान्वयन।

#### gtp.cpp

GTP (Go Text Protocol) मोड का कार्यान्वयन:

```cpp
void MainCmds::gtp(const std::vector<std::string>& args) {
    // कमांड पार्सिंग और निष्पादन
    while(true) {
        std::string line;
        std::getline(std::cin, line);

        if(line == "name") {
            respond("KataGo");
        }
        else if(line.find("play") == 0) {
            // चाल कमांड प्रोसेस करें
        }
        else if(line.find("genmove") == 0) {
            // सर्च निष्पादित करें और सर्वश्रेष्ठ चाल लौटाएं
        }
        // ... अन्य कमांड
    }
}
```

#### analysis.cpp

Analysis Engine का कार्यान्वयन:

```cpp
void MainCmds::analysis(const std::vector<std::string>& args) {
    while(true) {
        // JSON अनुरोध पढ़ें
        std::string line;
        std::getline(std::cin, line);
        json query = json::parse(line);

        // बोर्ड स्थिति सेटअप करें
        Board board = setupBoard(query);

        // विश्लेषण निष्पादित करें
        Search search(...);
        search.runWholeSearch();

        // JSON प्रतिक्रिया आउटपुट करें
        json response = formatResponse(search);
        std::cout << response.dump() << std::endl;
    }
}
```

---

## Python प्रशिक्षण कोड

### model.py — नेटवर्क आर्किटेक्चर

```python
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        # प्रारंभिक कन्वोल्यूशन
        self.initial_conv = nn.Conv2d(
            in_channels=config.input_features,
            out_channels=config.trunk_channels,
            kernel_size=3, padding=1
        )

        # रेसिड्यूअल टावर
        self.trunk = nn.ModuleList([
            ResidualBlock(config.trunk_channels)
            for _ in range(config.num_blocks)
        ])

        # आउटपुट हेड
        self.policy_head = PolicyHead(config)
        self.value_head = ValueHead(config)
        self.ownership_head = OwnershipHead(config)

    def forward(self, x):
        # प्रारंभिक कन्वोल्यूशन
        x = self.initial_conv(x)

        # रेसिड्यूअल टावर
        for block in self.trunk:
            x = block(x)

        # मल्टी-हेड आउटपुट
        policy = self.policy_head(x)
        value = self.value_head(x)
        ownership = self.ownership_head(x)

        return policy, value, ownership
```

**एनिमेशन संबंध**:
- D9 कन्वोल्यूशन ऑपरेशन: Conv2d
- D12 रेसिड्यूअल कनेक्शन: ResidualBlock
- E11 रेसिड्यूअल टावर: trunk संरचना

### train.py — प्रशिक्षण लूप

```python
def train_step(model, optimizer, batch):
    # फॉरवर्ड प्रोपेगेशन
    policy_pred, value_pred, ownership_pred = model(batch.inputs)

    # लॉस गणना
    policy_loss = cross_entropy(policy_pred, batch.policy_target)
    value_loss = mse_loss(value_pred, batch.value_target)
    ownership_loss = mse_loss(ownership_pred, batch.ownership_target)

    total_loss = policy_loss + value_loss + ownership_loss

    # बैकप्रोपेगेशन
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()
```

**एनिमेशन संबंध**:
- D3 फॉरवर्ड प्रोपेगेशन: model(batch.inputs)
- D13 बैकप्रोपेगेशन: total_loss.backward()
- K3 Adam: optimizer.step()

---

## मुख्य एल्गोरिथम कार्यान्वयन

### PUCT चयन फॉर्मूला

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

### वर्चुअल लॉस

```cpp
// मल्टी-थ्रेड्स को समान नोड चुनने से रोकें
void Search::applyVirtualLoss(SearchNode* node) {
    node->virtualLoss += params.virtualLoss;
}

void Search::removeVirtualLoss(SearchNode* node) {
    node->virtualLoss -= params.virtualLoss;
}
```

**एनिमेशन संबंध**:
- C9 वर्चुअल लॉस: पैरेलल सर्च की तकनीक

---

## कंपाइल और डीबगिंग

### कंपाइल (Debug मोड)

```bash
cd cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=OPENCL -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

### यूनिट टेस्ट

```bash
./katago runtests
```

### डीबगिंग तकनीकें

```cpp
// विस्तृत लॉगिंग सक्षम करें
#define SEARCH_DEBUG 1

// सर्च में ब्रेकपॉइंट जोड़ें
if(node->visits > 1000) {
    // सर्च स्थिति जांचने के लिए ब्रेकपॉइंट सेट करें
}
```

---

## आगे पढ़ें

- [KataGo प्रशिक्षण तंत्र विश्लेषण](../training) — पूर्ण प्रशिक्षण प्रक्रिया
- [ओपन सोर्स समुदाय में भागीदारी](../contributing) — योगदान गाइड
- [अवधारणा त्वरित संदर्भ](../../how-it-works/concepts/) — 109 अवधारणाओं की तुलना
