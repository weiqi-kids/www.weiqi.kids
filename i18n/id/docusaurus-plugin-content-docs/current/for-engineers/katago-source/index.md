---
sidebar_position: 2
title: Panduan Praktis KataGo
---

# Panduan Praktis KataGo

Bab ini akan membimbing Anda dari instalasi hingga penggunaan KataGo yang sebenarnya, mencakup semua pengetahuan operasi praktis. Baik Anda ingin mengintegrasikan KataGo ke aplikasi Anda sendiri, atau ingin meneliti kode sumbernya secara mendalam, ini adalah titik awal Anda.

## Mengapa Memilih KataGo?

Di antara banyak AI Go, KataGo adalah pilihan terbaik saat ini, alasannya sebagai berikut:

| Keunggulan | Penjelasan |
|------|------|
| **Kemampuan bermain terkuat** | Terus mempertahankan level tertinggi dalam pengujian terbuka |
| **Fitur terlengkap** | Prediksi poin, analisis wilayah, dukungan multi-aturan |
| **Sepenuhnya open source** | Lisensi MIT, bebas digunakan dan dimodifikasi |
| **Terus diperbarui** | Pengembangan aktif dan dukungan komunitas |
| **Dokumentasi lengkap** | Dokumentasi resmi detail, sumber komunitas kaya |
| **Dukungan multi-platform** | Linux, macOS, Windows semuanya dapat dijalankan |

## Isi Bab Ini

### [Instalasi dan Konfigurasi](./setup.md)

Membangun lingkungan KataGo dari nol:

- Kebutuhan sistem dan saran hardware
- Langkah instalasi untuk setiap platform (macOS / Linux / Windows)
- Panduan unduh dan pemilihan model
- Penjelasan detail file konfigurasi

### [Perintah Umum](./commands.md)

Menguasai cara penggunaan KataGo:

- Pengenalan protokol GTP (Go Text Protocol)
- Perintah GTP umum dan contoh
- Cara penggunaan Analysis Engine
- Penjelasan lengkap JSON API

### [Arsitektur Kode Sumber](./architecture.md)

Memahami detail implementasi KataGo secara mendalam:

- Ikhtisar struktur direktori proyek
- Analisis arsitektur neural network
- Detail implementasi search engine
- Gambaran umum proses pelatihan

## Mulai Cepat

Jika Anda hanya ingin coba KataGo dengan cepat, berikut cara paling sederhana:

### macOS (Menggunakan Homebrew)

```bash
# Instalasi
brew install katago

# Unduh model (pilih model yang lebih kecil untuk pengujian)
curl -L -o kata-b18c384.bin.gz \
  https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz

# Jalankan mode GTP
katago gtp -model kata-b18c384.bin.gz -config gtp_example.cfg
```

### Linux (Versi Precompiled)

```bash
# Unduh versi precompiled
wget https://github.com/lightvector/KataGo/releases/download/v1.15.3/katago-v1.15.3-opencl-linux-x64.zip

# Ekstrak
unzip katago-v1.15.3-opencl-linux-x64.zip

# Unduh model
wget https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz

# Jalankan
./katago gtp -model kata-b18c384nbt-*.bin.gz -config default_gtp.cfg
```

### Verifikasi Instalasi

Setelah berhasil dijalankan, Anda akan melihat prompt GTP. Coba masukkan perintah berikut:

```
name
= KataGo

version
= 1.15.3

boardsize 19
=

genmove black
= Q16
```

## Panduan Skenario Penggunaan

Berdasarkan kebutuhan Anda, berikut urutan membaca dan fokus yang disarankan:

### Skenario 1: Integrasi ke Aplikasi Go

Anda ingin menggunakan KataGo sebagai mesin AI dalam aplikasi Go Anda sendiri.

**Fokus membaca**:
1. [Instalasi dan Konfigurasi](./setup.md) - Memahami kebutuhan deployment
2. [Perintah Umum](./commands.md) - Terutama bagian Analysis Engine

**Pengetahuan kunci**:
- Gunakan mode Analysis Engine bukan mode GTP
- Komunikasi dengan KataGo melalui JSON API
- Sesuaikan parameter pencarian berdasarkan hardware

### Skenario 2: Membangun Server Bermain

Anda ingin membangun server yang memungkinkan pengguna bermain melawan AI.

**Fokus membaca**:
1. [Instalasi dan Konfigurasi](./setup.md) - Bagian pengaturan GPU
2. [Perintah Umum](./commands.md) - Bagian protokol GTP

**Pengetahuan kunci**:
- Gunakan mode GTP untuk bermain
- Strategi deployment multi-instance
- Metode penyesuaian kemampuan bermain

### Skenario 3: Meneliti Algoritma AI

Anda ingin meneliti implementasi KataGo secara mendalam, mungkin ingin memodifikasi atau bereksperimen.

**Fokus membaca**:
1. [Arsitektur Kode Sumber](./architecture.md) - Baca lengkap dengan teliti
2. Semua pembahasan makalah di bab latar belakang pengetahuan

**Pengetahuan kunci**:
- Struktur kode C++
- Detail arsitektur neural network
- Cara implementasi MCTS

### Skenario 4: Melatih Model Sendiri

Anda ingin melatih dari awal atau fine-tune model KataGo.

**Fokus membaca**:
1. [Arsitektur Kode Sumber](./architecture.md) - Bagian proses pelatihan
2. [Pembahasan Makalah KataGo](../background-info/katago-paper.md)

**Pengetahuan kunci**:
- Format data pelatihan
- Penggunaan skrip pelatihan
- Pengaturan hyperparameter

## Saran Hardware

KataGo dapat berjalan di berbagai hardware, tetapi perbedaan performa sangat besar:

| Konfigurasi Hardware | Performa yang Diharapkan | Skenario Cocok |
|---------|---------|---------|
| **GPU High-end** (RTX 4090) | ~2000 playouts/sec | Analisis top, pencarian cepat |
| **GPU Mid-range** (RTX 3060) | ~500 playouts/sec | Analisis umum, bermain |
| **GPU Entry-level** (GTX 1650) | ~100 playouts/sec | Penggunaan dasar |
| **Apple Silicon** (M1/M2) | ~200-400 playouts/sec | Pengembangan macOS |
| **CPU murni** | ~10-30 playouts/sec | Belajar, testing |

:::tip
Bahkan dengan hardware yang lebih lambat, KataGo masih dapat memberikan analisis yang berharga. Mengurangi jumlah pencarian akan menurunkan presisi, tetapi untuk pengajaran dan pembelajaran biasanya sudah cukup.
:::

## Pertanyaan Umum

### Apa perbedaan KataGo dengan Leela Zero?

| Aspek | KataGo | Leela Zero |
|------|--------|------------|
| Kemampuan bermain | Lebih kuat | Lebih lemah |
| Fitur | Kaya (poin, wilayah) | Dasar |
| Multi-aturan | Didukung | Tidak didukung |
| Status pengembangan | Aktif | Mode pemeliharaan |
| Efisiensi pelatihan | Tinggi | Lebih rendah |

### Apakah perlu GPU?

Tidak wajib, tetapi sangat disarankan:
- **Ada GPU**: Dapat melakukan analisis cepat, mendapat hasil berkualitas tinggi
- **Tanpa GPU**: Dapat menggunakan backend Eigen, tetapi lebih lambat

### Perbedaan file model?

| Ukuran Model | Ukuran File | Kemampuan | Kecepatan |
|---------|---------|------|------|
| b10c128 | ~20 MB | Sedang | Tercepat |
| b18c384 | ~140 MB | Kuat | Cepat |
| b40c256 | ~250 MB | Sangat kuat | Sedang |
| b60c320 | ~500 MB | Terkuat | Lambat |

Biasanya disarankan menggunakan b18c384 atau b40c256, menyeimbangkan kemampuan dan kecepatan.

## Sumber Terkait

- [KataGo GitHub](https://github.com/lightvector/KataGo)
- [Website Pelatihan KataGo](https://katagotraining.org/)
- [Komunitas Discord KataGo](https://discord.gg/bqkZAz3)
- [Lizzie](https://github.com/featurecat/lizzie) - GUI yang digunakan bersama KataGo

Siap? Mari kita mulai dari [Instalasi dan Konfigurasi](./setup.md)!

