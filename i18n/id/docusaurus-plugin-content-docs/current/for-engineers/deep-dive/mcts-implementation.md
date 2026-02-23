---
sidebar_position: 5
title: Detail Implementasi MCTS
description: Analisis mendalam implementasi Monte Carlo Tree Search, seleksi PUCT, dan teknik paralelisasi
---

# Detail Implementasi MCTS

Artikel ini menganalisis secara mendalam detail implementasi Monte Carlo Tree Search (MCTS) di KataGo, termasuk struktur data, strategi seleksi, dan teknik paralelisasi.

---

## Tinjauan Empat Langkah MCTS

```
┌─────────────────────────────────────────────────────┐
│                 Siklus Pencarian MCTS               │
├─────────────────────────────────────────────────────┤
│                                                     │
│   1. Selection      Seleksi: Turun pohon,           │
│         │           gunakan PUCT untuk memilih node │
│         ▼                                           │
│   2. Expansion      Ekspansi: Mencapai leaf node,   │
│         │           buat child node                 │
│         ▼                                           │
│   3. Evaluation     Evaluasi: Gunakan neural        │
│         │           network untuk mengevaluasi      │
│         ▼           leaf node                       │
│   4. Backprop       Backprop: Update statistik      │
│                     semua node di jalur             │
│                                                     │
│   Ulangi ribuan kali, pilih aksi dengan             │
│   kunjungan terbanyak                               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Struktur Data Node

### Data Inti

Setiap node MCTS perlu menyimpan:

```python
class MCTSNode:
    def __init__(self, state, parent=None, prior=0.0):
        # Informasi dasar
        self.state = state              # Status papan
        self.parent = parent            # Node induk
        self.children = {}              # Dictionary child node {action: node}
        self.action = None              # Aksi untuk mencapai node ini

        # Informasi statistik
        self.visit_count = 0            # N(s): Jumlah kunjungan
        self.value_sum = 0.0            # W(s): Total nilai
        self.prior = prior              # P(s,a): Probabilitas prior

        # Untuk pencarian paralel
        self.virtual_loss = 0           # Virtual loss
        self.is_expanded = False        # Apakah sudah diekspansi

    @property
    def value(self):
        """Q(s) = W(s) / N(s)"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
```

### Optimasi Memori

KataGo menggunakan berbagai teknik untuk mengurangi penggunaan memori:

```python
# Menggunakan array numpy bukan Python dict
class OptimizedNode:
    __slots__ = ['visit_count', 'value_sum', 'prior', 'children_indices']

    def __init__(self):
        self.visit_count = np.int32(0)
        self.value_sum = np.float32(0.0)
        self.prior = np.float32(0.0)
        self.children_indices = None  # Alokasi ditunda
```

---

## Selection: Seleksi PUCT

### Formula PUCT

```
Skor Seleksi = Q(s,a) + U(s,a)

Di mana:
Q(s,a) = W(s,a) / N(s,a)              # Nilai rata-rata
U(s,a) = c_puct × P(s,a) × √(N(s)) / (1 + N(s,a))  # Komponen eksplorasi
```

### Penjelasan Parameter

| Simbol | Arti | Nilai Tipikal |
|--------|------|---------------|
| Q(s,a) | Nilai rata-rata aksi a | [-1, +1] |
| P(s,a) | Probabilitas prior dari neural network | [0, 1] |
| N(s) | Jumlah kunjungan node induk | Integer |
| N(s,a) | Jumlah kunjungan aksi a | Integer |
| c_puct | Konstanta eksplorasi | 1.0 ~ 2.5 |

### Implementasi

```python
def select_child(self, c_puct=1.5):
    """Pilih child node dengan skor PUCT tertinggi"""
    best_score = -float('inf')
    best_action = None
    best_child = None

    # Akar kuadrat jumlah kunjungan node induk
    sqrt_parent_visits = math.sqrt(self.visit_count)

    for action, child in self.children.items():
        # Nilai Q (nilai rata-rata)
        if child.visit_count > 0:
            q_value = child.value_sum / child.visit_count
        else:
            q_value = 0.0

        # Nilai U (komponen eksplorasi)
        u_value = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)

        # Total skor
        score = q_value + u_value

        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    return best_action, best_child
```

### Keseimbangan Eksplorasi vs Eksploitasi

```
Tahap Awal: N(s,a) kecil
├── U(s,a) besar → Dominan eksplorasi
└── Aksi dengan probabilitas prior tinggi dieksplorasi lebih dulu

Tahap Akhir: N(s,a) besar
├── U(s,a) kecil → Dominan eksploitasi
└── Q(s,a) mendominasi, pilih aksi yang diketahui bagus
```

---

## Expansion: Ekspansi Node

### Kondisi Ekspansi

Saat mencapai leaf node, gunakan neural network untuk ekspansi:

```python
def expand(self, policy_probs, legal_moves):
    """Ekspansi node, buat child node untuk semua aksi legal"""
    for action in legal_moves:
        if action not in self.children:
            prior = policy_probs[action]  # Probabilitas dari neural network
            child_state = self.state.play(action)
            self.children[action] = MCTSNode(
                state=child_state,
                parent=self,
                prior=prior
            )

    self.is_expanded = True
```

### Filter Aksi Legal

```python
def get_legal_moves(state):
    """Dapatkan semua aksi legal"""
    legal = []
    for i in range(361):
        x, y = i // 19, i % 19
        if state.is_legal(x, y):
            legal.append(i)

    # Tambahkan pass
    legal.append(361)

    return legal
```

---

## Evaluation: Evaluasi Neural Network

### Evaluasi Tunggal

```python
def evaluate(self, state):
    """Gunakan neural network untuk mengevaluasi posisi"""
    # Encode fitur input
    features = encode_state(state)  # (22, 19, 19)
    features = torch.tensor(features).unsqueeze(0)  # (1, 22, 19, 19)

    # Inferensi neural network
    with torch.no_grad():
        output = self.network(features)

    policy = output['policy'][0].numpy()  # (362,)
    value = output['value'][0].item()     # scalar

    return policy, value
```

### Evaluasi Batch (Optimasi Kunci)

GPU paling efisien saat inferensi batch:

```python
class BatchedEvaluator:
    def __init__(self, network, batch_size=8):
        self.network = network
        self.batch_size = batch_size
        self.pending = []  # List (state, callback) yang menunggu evaluasi

    def request_evaluation(self, state, callback):
        """Minta evaluasi, otomatis eksekusi saat batch penuh"""
        self.pending.append((state, callback))

        if len(self.pending) >= self.batch_size:
            self.flush()

    def flush(self):
        """Eksekusi evaluasi batch"""
        if not self.pending:
            return

        # Siapkan input batch
        states = [s for s, _ in self.pending]
        features = torch.stack([encode_state(s) for s in states])

        # Inferensi batch
        with torch.no_grad():
            outputs = self.network(features)

        # Callback hasil
        for i, (_, callback) in enumerate(self.pending):
            policy = outputs['policy'][i].numpy()
            value = outputs['value'][i].item()
            callback(policy, value)

        self.pending.clear()
```

---

## Backpropagation: Update Balik

### Backprop Dasar

```python
def backpropagate(self, value):
    """Backprop dari leaf node ke root node, update informasi statistik"""
    node = self

    while node is not None:
        node.visit_count += 1
        node.value_sum += value

        # Perspektif bergantian: nilai lawan adalah kebalikannya
        value = -value

        node = node.parent
```

### Pentingnya Perspektif Bergantian

```
Perspektif Hitam: value = +0.6 (Hitam unggul)

Jalur backprop:
Leaf node (giliran Hitam): value_sum += +0.6
    ↑
Parent node (giliran Putih): value_sum += -0.6  ← Tidak menguntungkan untuk Putih
    ↑
Grandparent node (giliran Hitam): value_sum += +0.6
    ↑
...
```

---

## Paralelisasi: Virtual Loss

### Masalah

Saat multi-thread mencari secara bersamaan, mungkin semua memilih node yang sama:

```
Thread 1: Pilih node A (Q=0.6, N=100)
Thread 2: Pilih node A (Q=0.6, N=100) ← Duplikat!
Thread 3: Pilih node A (Q=0.6, N=100) ← Duplikat!
```

### Solusi: Virtual Loss

Saat memilih node, tambahkan "virtual loss" lebih dulu, membuat thread lain tidak mau memilihnya:

```python
VIRTUAL_LOSS = 3  # Nilai virtual loss

def select_with_virtual_loss(self):
    """Seleksi dengan virtual loss"""
    action, child = self.select_child()

    # Tambahkan virtual loss
    child.visit_count += VIRTUAL_LOSS
    child.value_sum -= VIRTUAL_LOSS  # Pura-pura kalah

    return action, child

def backpropagate_with_virtual_loss(self, value):
    """Hapus virtual loss saat backprop"""
    node = self

    while node is not None:
        # Hapus virtual loss
        node.visit_count -= VIRTUAL_LOSS
        node.value_sum += VIRTUAL_LOSS

        # Update normal
        node.visit_count += 1
        node.value_sum += value

        value = -value
        node = node.parent
```

### Efek

```
Thread 1: Pilih node A, tambahkan virtual loss
         Nilai Q node A turun sementara

Thread 2: Pilih node B (karena A terlihat lebih buruk)

Thread 3: Pilih node C

→ Thread berbeda mengeksplorasi cabang berbeda, meningkatkan efisiensi
```

---

## Implementasi Pencarian Lengkap

```python
class MCTS:
    def __init__(self, network, c_puct=1.5, num_simulations=800):
        self.network = network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.evaluator = BatchedEvaluator(network)

    def search(self, root_state):
        """Eksekusi pencarian MCTS"""
        root = MCTSNode(root_state)

        # Ekspansi root node
        policy, value = self.evaluate(root_state)
        legal_moves = get_legal_moves(root_state)
        root.expand(policy, legal_moves)

        # Eksekusi simulasi
        for _ in range(self.num_simulations):
            node = root
            path = [node]

            # Selection: Turun pohon
            while node.is_expanded and node.children:
                action, node = node.select_child(self.c_puct)
                path.append(node)

            # Expansion + Evaluation
            if not node.is_expanded:
                policy, value = self.evaluate(node.state)
                legal_moves = get_legal_moves(node.state)

                if legal_moves:
                    node.expand(policy, legal_moves)

            # Backpropagation
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += value
                value = -value

        # Pilih aksi dengan kunjungan terbanyak
        best_action = max(root.children.items(),
                         key=lambda x: x[1].visit_count)[0]

        return best_action

    def evaluate(self, state):
        features = encode_state(state)
        features = torch.tensor(features).unsqueeze(0)

        with torch.no_grad():
            output = self.network(features)

        return output['policy'][0].numpy(), output['value'][0].item()
```

---

## Teknik Lanjutan

### Noise Dirichlet

Tambahkan noise di root node saat pelatihan untuk meningkatkan eksplorasi:

```python
def add_dirichlet_noise(root, alpha=0.03, epsilon=0.25):
    """Tambahkan noise Dirichlet di root node"""
    noise = np.random.dirichlet([alpha] * len(root.children))

    for i, child in enumerate(root.children.values()):
        child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]
```

### Parameter Temperatur

Mengontrol keacakan pemilihan aksi:

```python
def select_action_with_temperature(root, temperature=1.0):
    """Pilih aksi berdasarkan jumlah kunjungan dan temperatur"""
    visits = np.array([c.visit_count for c in root.children.values()])
    actions = list(root.children.keys())

    if temperature == 0:
        # Pilihan greedy
        return actions[np.argmax(visits)]
    else:
        # Pilih berdasarkan distribusi probabilitas dari jumlah kunjungan
        probs = visits ** (1 / temperature)
        probs = probs / probs.sum()
        return np.random.choice(actions, p=probs)
```

### Penggunaan Ulang Pohon

Langkah baru bisa menggunakan ulang pohon pencarian sebelumnya:

```python
def reuse_tree(root, action):
    """Gunakan ulang subtree"""
    if action in root.children:
        new_root = root.children[action]
        new_root.parent = None
        return new_root
    else:
        return None  # Perlu buat pohon baru
```

---

## Ringkasan Optimasi Performa

| Teknik | Efek |
|--------|------|
| **Evaluasi Batch** | Utilisasi GPU dari 10% → 80%+ |
| **Virtual Loss** | Efisiensi multi-thread meningkat 3-5x |
| **Penggunaan Ulang Pohon** | Kurangi cold start, hemat 30%+ komputasi |
| **Memory Pool** | Kurangi overhead alokasi memori |

---

## Bacaan Lanjutan

- [Detail Arsitektur Neural Network](../neural-network) — Sumber fungsi evaluasi
- [Backend GPU dan Optimasi](../gpu-optimization) — Optimasi hardware untuk inferensi batch
- [Panduan Paper Kunci](../papers) — Dasar teori formula PUCT
