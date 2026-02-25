---
sidebar_position: 1
title: Untuk Peneliti Mendalam
description: "Panduan topik lanjutan: neural network, MCTS, pelatihan, optimasi, deployment"
---

# Untuk Peneliti Mendalam

Bagian ini ditujukan untuk engineer yang ingin mendalami AI Go, mencakup implementasi teknis, dasar teori, dan aplikasi praktis.

---

## Daftar Artikel

### Teknologi Inti

| Artikel | Deskripsi |
|---------|-----------|
| [Detail Arsitektur Neural Network](./neural-network) | Residual network KataGo, fitur input, desain multi-head output |
| [Detail Implementasi MCTS](./mcts-implementation) | Seleksi PUCT, virtual loss, evaluasi batch, paralelisasi |
| [Analisis Mekanisme Pelatihan KataGo](./training) | Self-play, fungsi loss, siklus pelatihan |

### Optimasi Performa

| Artikel | Deskripsi |
|---------|-----------|
| [Backend GPU dan Optimasi](./gpu-optimization) | Perbandingan dan tuning backend CUDA, OpenCL, Metal |
| [Kuantisasi Model dan Deployment](./quantization-deploy) | FP16, INT8, TensorRT, deployment multi-platform |
| [Evaluasi dan Benchmark](./evaluation) | Rating Elo, pengujian pertandingan, metode statistik SPRT |

### Topik Lanjutan

| Artikel | Deskripsi |
|---------|-----------|
| [Arsitektur Pelatihan Terdistribusi](./distributed-training) | Self-play Worker, pengumpulan data, rilis model |
| [Aturan Kustom dan Varian](./custom-rules) | Aturan Tiongkok, Jepang, AGA, varian ukuran papan |
| [Panduan Paper Kunci](./papers) | Analisis poin penting paper AlphaGo, AlphaZero, KataGo |

### Open Source dan Implementasi

| Artikel | Deskripsi |
|---------|-----------|
| [Panduan Source Code KataGo](./source-code) | Struktur direktori, modul inti, gaya kode |
| [Berkontribusi ke Komunitas Open Source](./contributing) | Cara kontribusi, pelatihan terdistribusi, partisipasi komunitas |
| [Membangun AI Go dari Nol](./build-from-scratch) | Implementasi bertahap AlphaGo Zero versi sederhana |

---

## Apa yang Ingin Anda Lakukan?

| Tujuan | Jalur yang Disarankan |
|--------|----------------------|
| Memahami desain neural network | [Detail Arsitektur Neural Network](./neural-network) → [Detail Implementasi MCTS](./mcts-implementation) |
| Mengoptimasi performa eksekusi | [Backend GPU dan Optimasi](./gpu-optimization) → [Kuantisasi Model dan Deployment](./quantization-deploy) |
| Meneliti metode pelatihan | [Analisis Mekanisme Pelatihan KataGo](./training) → [Arsitektur Pelatihan Terdistribusi](./distributed-training) |
| Memahami prinsip paper | [Panduan Paper Kunci](./papers) → [Detail Arsitektur Neural Network](./neural-network) |
| Menulis kode sendiri | [Membangun AI Go dari Nol](./build-from-scratch) → [Panduan Source Code KataGo](./source-code) |
| Berkontribusi ke proyek open source | [Berkontribusi ke Komunitas Open Source](./contributing) → [Panduan Source Code KataGo](./source-code) |

---

## Indeks Konsep Lanjutan

Saat mendalami penelitian, Anda akan menemui konsep-konsep lanjutan berikut:

### Seri F: Penskalaan (8 konsep)

| No | Konsep Go | Padanan Fisika/Matematika |
|----|-----------|---------------------------|
| F1 | Ukuran papan vs kompleksitas | Penskalaan kompleksitas |
| F2 | Ukuran network vs kekuatan | Penskalaan kapasitas |
| F3 | Waktu pelatihan vs hasil | Hukum diminishing returns |
| F4 | Jumlah data vs generalisasi | Kompleksitas sampel |
| F5 | Penskalaan sumber daya komputasi | Hukum penskalaan |
| F6 | Hukum penskalaan neural | Hubungan log-log |
| F7 | Pelatihan batch besar | Batch kritis |
| F8 | Efisiensi parameter | Batas kompresi |

### Seri G: Dimensi (6 konsep)

| No | Konsep Go | Padanan Fisika/Matematika |
|----|-----------|---------------------------|
| G1 | Representasi dimensi tinggi | Ruang vektor |
| G2 | Kutukan dimensionalitas | Dilema dimensi tinggi |
| G3 | Hipotesis manifold | Manifold dimensi rendah |
| G4 | Representasi antara | Ruang laten |
| G5 | Pemisahan fitur | Komponen independen |
| G6 | Arah semantik | Aljabar geometris |

### Seri H: Reinforcement Learning (9 konsep)

| No | Konsep Go | Padanan Fisika/Matematika |
|----|-----------|---------------------------|
| H1 | MDP | Rantai Markov |
| H2 | Persamaan Bellman | Pemrograman dinamis |
| H3 | Iterasi nilai | Teorema titik tetap |
| H4 | Gradien kebijakan | Optimasi stokastik |
| H5 | Experience replay | Importance sampling |
| H6 | Faktor diskon | Preferensi waktu |
| H7 | Pembelajaran TD | Estimasi inkremental |
| H8 | Fungsi advantage | Pengurangan varians baseline |
| H9 | Clipping PPO | Trust region |

### Seri K: Metode Optimasi (6 konsep)

| No | Konsep Go | Padanan Fisika/Matematika |
|----|-----------|---------------------------|
| K1 | SGD | Aproksimasi stokastik |
| K2 | Momentum | Inersia |
| K3 | Adam | Step size adaptif |
| K4 | Decay learning rate | Annealing |
| K5 | Gradient clipping | Batasan saturasi |
| K6 | Noise SGD | Perturbasi stokastik |

### Seri L: Generalisasi dan Stabilitas (5 konsep)

| No | Konsep Go | Padanan Fisika/Matematika |
|----|-----------|---------------------------|
| L1 | Overfitting | Over-adaptasi |
| L2 | Regularisasi | Optimasi terkendala |
| L3 | Dropout | Aktivasi sparse |
| L4 | Augmentasi data | Pelanggaran simetri |
| L5 | Early stopping | Penghentian optimal |

---

## Kebutuhan Hardware

### Membaca dan Belajar

Tidak ada kebutuhan khusus, komputer apapun bisa digunakan.

### Melatih Model

| Skala | Hardware yang Disarankan | Waktu Pelatihan |
|-------|--------------------------|-----------------|
| Mini (b6c96) | GTX 1060 6GB | Beberapa jam |
| Kecil (b10c128) | RTX 3060 12GB | 1-2 hari |
| Sedang (b18c384) | RTX 4090 24GB | 1-2 minggu |
| Penuh (b40c256) | Cluster multi-GPU | Beberapa minggu |

### Kontribusi Pelatihan Terdistribusi

- Komputer apapun dengan GPU bisa berpartisipasi
- Disarankan minimal GTX 1060 atau setara
- Membutuhkan koneksi internet yang stabil

---

## Mulai Membaca

**Rekomendasi untuk memulai:**

- Ingin memahami prinsip? → [Detail Arsitektur Neural Network](./neural-network)
- Ingin praktik langsung? → [Membangun AI Go dari Nol](./build-from-scratch)
- Ingin membaca paper? → [Panduan Paper Kunci](./papers)
