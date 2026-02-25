---
sidebar_position: 2
title: Panduan Source Code KataGo
description: Struktur kode KataGo, modul inti, dan desain arsitektur
---

# Panduan Source Code KataGo

Artikel ini membantu Anda memahami struktur kode KataGo, cocok untuk engineer yang ingin mendalami penelitian atau berkontribusi kode.

---

## Mendapatkan Source Code

```bash
git clone https://github.com/lightvector/KataGo.git
cd KataGo
```

---

## Struktur Direktori

```
KataGo/
├── cpp/                    # Engine inti C++
│   ├── main.cpp            # Entry point program utama
│   ├── command/            # Penanganan perintah
│   ├── core/               # Utilitas inti
│   ├── game/               # Aturan Go
│   ├── search/             # Pencarian MCTS
│   ├── neuralnet/          # Inferensi neural network
│   ├── dataio/             # I/O data
│   └── tests/              # Unit test
│
├── python/                 # Kode pelatihan Python
│   ├── train.py            # Program pelatihan utama
│   ├── model.py            # Definisi arsitektur network
│   ├── data/               # Pemrosesan data
│   └── configs/            # Konfigurasi pelatihan
│
└── docs/                   # Dokumentasi
```

---

## Analisis Modul Inti

### 1. game/ — Aturan Go

Implementasi lengkap aturan Go.

#### board.h / board.cpp

```cpp
// Representasi status papan
class Board {
public:
    static constexpr int MAX_BOARD_SIZE = 19;

    // Status papan
    Color colors[MAX_ARR_SIZE];  // Warna setiap posisi
    Chain chains[MAX_ARR_SIZE];  // Informasi grup batu

    // Operasi inti
    bool playMove(Loc loc, Player pla);  // Mainkan satu langkah
    bool isLegal(Loc loc, Player pla);   // Periksa legalitas
    void calculateArea(Color* area);      // Hitung teritori
};
```

**Padanan Animasi**:
- A2 Model lattice: Struktur data papan
- A6 Area terhubung: Representasi Chain (grup batu)
- A7 Penghitungan liberty: Pelacakan liberty

#### rules.h / rules.cpp

```cpp
// Dukungan multi-aturan
struct Rules {
    enum KoRule { SIMPLE_KO, POSITIONAL_KO, SITUATIONAL_KO };
    enum ScoringRule { TERRITORY_SCORING, AREA_SCORING };
    enum TaxRule { NO_TAX, TAX_SEKI, TAX_ALL };

    KoRule koRule;
    ScoringRule scoringRule;
    TaxRule taxRule;
    float komi;

    // Pemetaan nama aturan
    static Rules parseRules(const std::string& name);
};
```

Aturan yang didukung:
- `chinese`: Aturan Tiongkok (penghitungan area)
- `japanese`: Aturan Jepang (penghitungan teritori)
- `korean`: Aturan Korea
- `aga`: Aturan Amerika
- `tromp-taylor`: Aturan Tromp-Taylor

---

### 2. search/ — Pencarian MCTS

Implementasi Monte Carlo Tree Search.

#### search.h / search.cpp

```cpp
class Search {
public:
    // Pencarian inti
    void runWholeSearch(Player pla);

    // Langkah MCTS
    void selectNode();           // Pilih node
    void expandNode();           // Ekspansi node
    void evaluateNode();         // Evaluasi neural network
    void backpropValue();        // Update backprop

    // Ambil hasil
    Loc getChosenMove();
    std::vector<MoveInfo> getSortedMoveInfos();
};
```

**Padanan Animasi**:
- C5 Empat langkah MCTS: Padanan select → expand → evaluate → backprop
- E4 Formula PUCT: Diimplementasikan di `selectNode()`

#### searchparams.h

```cpp
struct SearchParams {
    // Kontrol pencarian
    int64_t maxVisits;          // Maksimum kunjungan
    double maxTime;             // Maksimum waktu

    // Parameter PUCT
    double cpuctExploration;    // Konstanta eksplorasi
    double cpuctBase;

    // Virtual loss
    int virtualLoss;

    // Noise root node
    double rootNoiseEnabled;
    double rootDirichletAlpha;
};
```

---

### 3. neuralnet/ — Inferensi Neural Network

Engine inferensi neural network.

#### nninputs.h / nninputs.cpp

```cpp
// Fitur input neural network
class NNInputs {
public:
    // Feature plane
    static constexpr int NUM_FEATURES = 22;

    // Isi fitur
    static void fillFeatures(
        const Board& board,
        const BoardHistory& hist,
        float* features
    );
};
```

Fitur input mencakup:
- Posisi batu hitam, posisi batu putih
- Jumlah liberty (1, 2, 3+)
- Langkah historis
- Encoding aturan

**Padanan Animasi**:
- A10 Stack historis: Input multi-frame
- A11 Mask langkah legal: Filter langkah terlarang

#### nneval.h / nneval.cpp

```cpp
// Hasil evaluasi neural network
struct NNOutput {
    // Output Policy (362 posisi, termasuk pass)
    float policyProbs[NNPos::MAX_NN_POLICY_SIZE];

    // Output Value
    float winProb;       // Winrate
    float lossProb;      // Lossrate
    float noResultProb;  // Drawrate

    // Output tambahan
    float scoreMean;     // Prediksi skor
    float scoreStdev;    // Standar deviasi skor
    float lead;          // Poin unggul

    // Prediksi teritori
    float ownership[NNPos::MAX_BOARD_AREA];
};
```

**Padanan Animasi**:
- E1 Policy network: policyProbs
- E2 Value network: winProb, scoreMean
- E3 Dual-head network: Desain multi-output head

---

### 4. command/ — Penanganan Perintah

Implementasi mode operasi yang berbeda.

#### gtp.cpp

Implementasi mode GTP (Go Text Protocol):

```cpp
void MainCmds::gtp(const std::vector<std::string>& args) {
    // Parsing dan eksekusi perintah
    while(true) {
        std::string line;
        std::getline(std::cin, line);

        if(line == "name") {
            respond("KataGo");
        }
        else if(line.find("play") == 0) {
            // Tangani perintah bermain
        }
        else if(line.find("genmove") == 0) {
            // Jalankan pencarian dan kembalikan langkah terbaik
        }
        // ... perintah lainnya
    }
}
```

#### analysis.cpp

Implementasi Analysis Engine:

```cpp
void MainCmds::analysis(const std::vector<std::string>& args) {
    while(true) {
        // Baca request JSON
        std::string line;
        std::getline(std::cin, line);
        json query = json::parse(line);

        // Bangun status papan
        Board board = setupBoard(query);

        // Jalankan analisis
        Search search(...);
        search.runWholeSearch();

        // Output response JSON
        json response = formatResponse(search);
        std::cout << response.dump() << std::endl;
    }
}
```

---

## Kode Pelatihan Python

### model.py — Arsitektur Network

```python
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Konvolusi awal
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

        # Output head
        self.policy_head = PolicyHead(config)
        self.value_head = ValueHead(config)
        self.ownership_head = OwnershipHead(config)

    def forward(self, x):
        # Konvolusi awal
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

**Padanan Animasi**:
- D9 Operasi konvolusi: Conv2d
- D12 Koneksi residual: ResidualBlock
- E11 Residual tower: Struktur trunk

### train.py — Loop Pelatihan

```python
def train_step(model, optimizer, batch):
    # Forward pass
    policy_pred, value_pred, ownership_pred = model(batch.inputs)

    # Hitung loss
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

**Padanan Animasi**:
- D3 Forward pass: model(batch.inputs)
- D13 Backward pass: total_loss.backward()
- K3 Adam: optimizer.step()

---

## Implementasi Algoritma Kunci

### Formula Seleksi PUCT

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
// Hindari multi-thread memilih node yang sama
void Search::applyVirtualLoss(SearchNode* node) {
    node->virtualLoss += params.virtualLoss;
}

void Search::removeVirtualLoss(SearchNode* node) {
    node->virtualLoss -= params.virtualLoss;
}
```

**Padanan Animasi**:
- C9 Virtual loss: Teknik pencarian paralel

---

## Kompilasi dan Debugging

### Kompilasi (Mode Debug)

```bash
cd cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=OPENCL -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

### Unit Test

```bash
./katago runtests
```

### Tips Debugging

```cpp
// Aktifkan log detail
#define SEARCH_DEBUG 1

// Tambahkan breakpoint dalam pencarian
if(node->visits > 1000) {
    // Set breakpoint untuk memeriksa status pencarian
}
```

---

## Bacaan Lanjutan

- [Analisis Mekanisme Pelatihan KataGo](../training) — Alur pelatihan lengkap
- [Berkontribusi ke Komunitas Open Source](../contributing) — Panduan kontribusi
- [Lembar Referensi Konsep](../../how-it-works/concepts/) — Padanan 109 konsep
