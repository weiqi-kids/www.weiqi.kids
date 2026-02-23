---
sidebar_position: 11
title: Panduan Paper Kunci
description: Analisis poin penting paper milestone AI Go seperti AlphaGo, AlphaZero, KataGo
---

# Panduan Paper Kunci

Artikel ini merangkum paper paling penting dalam sejarah pengembangan AI Go, menyediakan ringkasan dan poin teknis untuk pemahaman cepat.

---

## Gambaran Paper

### Timeline

```
2006  Coulom - MCTS pertama kali diterapkan pada Go
2016  Silver et al. - AlphaGo (Nature)
2017  Silver et al. - AlphaGo Zero (Nature)
2017  Silver et al. - AlphaZero
2019  Wu - KataGo
2020+ Berbagai perbaikan dan aplikasi
```

### Rekomendasi Membaca

| Tujuan | Paper yang Disarankan |
|--------|----------------------|
| Memahami dasar | AlphaGo (2016) |
| Memahami self-play | AlphaGo Zero (2017) |
| Memahami metode umum | AlphaZero (2017) |
| Referensi implementasi | KataGo (2019) |

---

## 1. Kelahiran MCTS (2006)

### Informasi Paper

```
Judul: Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search
Penulis: Rémi Coulom
Dipresentasikan: Computers and Games 2006
```

### Kontribusi Inti

Pertama kali menerapkan metode Monte Carlo secara sistematis pada Go:

```
Sebelumnya: Simulasi acak murni, tanpa struktur pohon
Sesudahnya: Membangun pohon pencarian + Seleksi UCB + Statistik backprop
```

### Konsep Kunci

#### Formula UCB1

```
Skor Seleksi = Rata-rata winrate + C × √(ln(N) / n)

Di mana:
- N: Jumlah kunjungan node induk
- n: Jumlah kunjungan node anak
- C: Konstanta eksplorasi
```

#### Empat Langkah MCTS

```
1. Selection: Pilih node menggunakan UCB
2. Expansion: Ekspansi node baru
3. Simulation: Simulasi acak sampai akhir permainan
4. Backpropagation: Backprop menang/kalah
```

### Dampak

- Membuat AI Go mencapai level dan amatir
- Menjadi dasar untuk semua AI Go selanjutnya
- Konsep UCB mempengaruhi pengembangan PUCT

---

## 2. AlphaGo (2016)

### Informasi Paper

```
Judul: Mastering the game of Go with deep neural networks and tree search
Penulis: Silver, D., Huang, A., Maddison, C.J., et al.
Dipublikasikan: Nature, 2016
DOI: 10.1038/nature16961
```

### Kontribusi Inti

**Pertama kali menggabungkan deep learning dengan MCTS**, mengalahkan juara dunia manusia.

### Arsitektur Sistem

```
┌─────────────────────────────────────────────┐
│              Arsitektur AlphaGo             │
├─────────────────────────────────────────────┤
│                                             │
│   Policy Network (SL)                       │
│   ├── Input: Status papan (48 feature plane)│
│   ├── Arsitektur: 13 layer CNN              │
│   ├── Output: Probabilitas 361 posisi       │
│   └── Pelatihan: 30 juta rekaman manusia    │
│                                             │
│   Policy Network (RL)                       │
│   ├── Diinisialisasi dari SL Policy         │
│   └── Reinforcement learning self-play      │
│                                             │
│   Value Network                             │
│   ├── Input: Status papan                   │
│   ├── Output: Nilai winrate tunggal         │
│   └── Pelatihan: Posisi dari self-play      │
│                                             │
│   MCTS                                      │
│   ├── Gunakan Policy Network untuk panduan  │
│   └── Gunakan Value Network + Rollout       │
│       untuk evaluasi                        │
│                                             │
└─────────────────────────────────────────────┘
```

### Poin Teknis

#### 1. Supervised Learning Policy Network

```python
# Fitur input (48 plane)
- Posisi batu sendiri
- Posisi batu lawan
- Jumlah liberty
- Status setelah penangkapan
- Posisi langkah legal
- Posisi beberapa langkah terakhir
...
```

#### 2. Perbaikan Reinforcement Learning

```
SL Policy → Self-play → RL Policy

RL Policy sekitar 80% lebih kuat dari SL Policy
```

#### 3. Pelatihan Value Network

```
Kunci mencegah overfitting:
- Hanya ambil satu posisi dari setiap permainan
- Hindari posisi serupa muncul berulang
```

#### 4. Integrasi MCTS

```
Evaluasi leaf node = 0.5 × Value Network + 0.5 × Rollout

Rollout menggunakan Policy Network cepat (akurasi lebih rendah tapi lebih cepat)
```

### Data Kunci

| Item | Nilai |
|------|-------|
| Akurasi SL Policy | 57% |
| Winrate RL Policy vs SL Policy | 80% |
| GPU Pelatihan | 176 |
| GPU Pertandingan | 48 TPU |

---

## 3. AlphaGo Zero (2017)

### Informasi Paper

```
Judul: Mastering the game of Go without human knowledge
Penulis: Silver, D., Schrittwieser, J., Simonyan, K., et al.
Dipublikasikan: Nature, 2017
DOI: 10.1038/nature24270
```

### Kontribusi Inti

**Tidak memerlukan rekaman manusia sama sekali**, belajar sendiri dari nol.

### Perbedaan dengan AlphaGo

| Aspek | AlphaGo | AlphaGo Zero |
|-------|---------|--------------|
| Rekaman manusia | Diperlukan | **Tidak diperlukan** |
| Jumlah network | 4 | **1 dual-head** |
| Fitur input | 48 plane | **17 plane** |
| Rollout | Digunakan | **Tidak digunakan** |
| Residual network | Tidak | **Ya** |
| Waktu pelatihan | Berbulan-bulan | **3 hari** |

### Inovasi Kunci

#### 1. Single Dual-Head Network

```
              Input (17 plane)
                   │
              ┌────┴────┐
              │ Residual│
              │  Tower  │
              │ (19 atau│
              │  39 layer)│
              └────┬────┘
           ┌──────┴──────┐
           │             │
        Policy         Value
        (361)          (1)
```

#### 2. Fitur Input yang Disederhanakan

```python
# Hanya 17 feature plane
features = [
    current_player_stones,      # Batu sendiri
    opponent_stones,            # Batu lawan
    history_1_player,           # Status historis 1
    history_1_opponent,
    ...                         # Status historis 2-7
    color_to_play               # Giliran siapa
]
```

#### 3. Evaluasi Pure Value Network

```
Tidak lagi menggunakan Rollout
Evaluasi leaf node = Output Value Network

Lebih ringkas, lebih cepat
```

#### 4. Alur Pelatihan

```
Inisialisasi network acak
    │
    ▼
┌─────────────────────────────┐
│  Self-play menghasilkan     │ ←─┐
│  rekaman permainan          │   │
└──────────────┬──────────────┘   │
               │                   │
               ▼                   │
┌─────────────────────────────┐   │
│  Latih neural network       │   │
│  - Policy: Minimisasi       │   │
│    cross entropy            │   │
│  - Value: Minimisasi MSE    │   │
└──────────────┬──────────────┘   │
               │                   │
               ▼                   │
┌─────────────────────────────┐   │
│  Evaluasi network baru      │   │
│  Jika lebih kuat ganti      │───┘
└─────────────────────────────┘
```

### Kurva Pembelajaran

```
Waktu Pelatihan    Elo
─────────────────────
3 jam              Pemula
24 jam             Melampaui AlphaGo Lee
72 jam             Melampaui AlphaGo Master
```

---

## 4. AlphaZero (2017)

### Informasi Paper

```
Judul: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
Penulis: Silver, D., Hubert, T., Schrittwieser, J., et al.
Dipublikasikan: arXiv:1712.01815 (kemudian dipublikasikan di Science, 2018)
```

### Kontribusi Inti

**Generalisasi**: Algoritma yang sama diterapkan pada Go, catur, dan shogi.

### Arsitektur Umum

```
Encoding Input (game-specific) → Residual Network (umum) → Dual-head Output (umum)
```

### Adaptasi Lintas Game

| Game | Plane Input | Ruang Aksi | Waktu Pelatihan |
|------|-------------|------------|-----------------|
| Go | 17 | 362 | 40 hari |
| Catur | 119 | 4672 | 9 jam |
| Shogi | 362 | 11259 | 12 jam |

### Perbaikan MCTS

#### Formula PUCT

```
Skor Seleksi = Q(s,a) + c(s) × P(s,a) × √N(s) / (1 + N(s,a))

c(s) = log((1 + N(s) + c_base) / c_base) + c_init
```

#### Noise Eksplorasi

```python
# Tambahkan noise Dirichlet di root node
P(s,a) = (1 - ε) × p_a + ε × η_a

η ~ Dir(α)
α = 0.03 (Go), 0.3 (catur), 0.15 (shogi)
```

---

## 5. KataGo (2019)

### Informasi Paper

```
Judul: Accelerating Self-Play Learning in Go
Penulis: David J. Wu
Dipublikasikan: arXiv:1902.10565
```

### Kontribusi Inti

**Peningkatan efisiensi 50x**, memungkinkan pengembang individu melatih AI Go yang kuat.

### Inovasi Kunci

#### 1. Target Pelatihan Tambahan

```
Total Loss = Policy Loss + Value Loss +
             Score Loss + Ownership Loss + ...

Target tambahan membuat network konvergen lebih cepat
```

#### 2. Fitur Global

```python
# Layer global pooling
global_features = global_avg_pool(conv_features)
# Gabungkan dengan fitur lokal
combined = concat(conv_features, broadcast(global_features))
```

#### 3. Randomisasi Playout Cap

```
Tradisional: Setiap pencarian N kali tetap
KataGo: N diambil sampel acak dari distribusi tertentu

Membuat network belajar tampil baik di berbagai kedalaman pencarian
```

#### 4. Ukuran Papan Progresif

```python
if training_step < 1000000:
    board_size = random.choice([9, 13, 19])
else:
    board_size = 19
```

### Perbandingan Efisiensi

| Metrik | AlphaZero | KataGo |
|--------|-----------|--------|
| GPU-hari untuk mencapai level superhuman | 5000 | **100** |
| Peningkatan efisiensi | Baseline | **50x** |

---

## 6. Paper Lanjutan

### MuZero (2020)

```
Judul: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model
Kontribusi: Mempelajari model dinamika lingkungan, tidak memerlukan aturan game
```

### EfficientZero (2021)

```
Judul: Mastering Atari Games with Limited Data
Kontribusi: Peningkatan efisiensi sampel yang signifikan
```

### Gumbel AlphaZero (2022)

```
Judul: Policy Improvement by Planning with Gumbel
Kontribusi: Metode policy improvement yang ditingkatkan
```

---

## Saran Membaca Paper

### Urutan untuk Pemula

```
1. AlphaGo (2016) - Memahami arsitektur dasar
2. AlphaGo Zero (2017) - Memahami self-play
3. KataGo (2019) - Memahami detail implementasi
```

### Urutan Lanjutan

```
4. AlphaZero (2017) - Generalisasi
5. MuZero (2020) - Mempelajari model dunia
6. Paper MCTS asli - Memahami dasar
```

### Tips Membaca

1. **Baca abstrak dan kesimpulan dulu**: Pahami cepat kontribusi inti
2. **Lihat gambar dan tabel**: Pahami arsitektur keseluruhan
3. **Baca bagian metode**: Pahami detail teknis
4. **Lihat lampiran**: Temukan detail implementasi dan hyperparameter

---

## Link Sumber Daya

### PDF Paper

| Paper | Link |
|-------|------|
| AlphaGo | [Nature](https://www.nature.com/articles/nature16961) |
| AlphaGo Zero | [Nature](https://www.nature.com/articles/nature24270) |
| AlphaZero | [Science](https://www.science.org/doi/10.1126/science.aar6404) |
| KataGo | [arXiv](https://arxiv.org/abs/1902.10565) |

### Implementasi Open Source

| Proyek | Link |
|--------|------|
| KataGo | [GitHub](https://github.com/lightvector/KataGo) |
| Leela Zero | [GitHub](https://github.com/leela-zero/leela-zero) |
| MiniGo | [GitHub](https://github.com/tensorflow/minigo) |

---

## Bacaan Lanjutan

- [Detail Arsitektur Neural Network](../neural-network) — Memahami mendalam desain network
- [Detail Implementasi MCTS](../mcts-implementation) — Implementasi algoritma pencarian
- [Analisis Mekanisme Pelatihan KataGo](../training) — Detail alur pelatihan
