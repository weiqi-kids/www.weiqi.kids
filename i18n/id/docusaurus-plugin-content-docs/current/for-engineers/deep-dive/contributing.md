---
sidebar_position: 4
title: Berkontribusi ke Komunitas Open Source
description: Bergabung dengan komunitas open source KataGo, kontribusi daya komputasi atau kode
---

# Berkontribusi ke Komunitas Open Source

KataGo adalah proyek open source yang aktif, dengan berbagai cara untuk berpartisipasi dan berkontribusi.

---

## Gambaran Cara Kontribusi

| Cara | Kesulitan | Kebutuhan |
|------|-----------|-----------|
| **Kontribusi daya komputasi** | Rendah | Komputer dengan GPU |
| **Lapor masalah** | Rendah | Akun GitHub |
| **Perbaiki dokumentasi** | Sedang | Familiar dengan konten teknis |
| **Kontribusi kode** | Tinggi | Kemampuan pengembangan C++/Python |

---

## Kontribusi Daya Komputasi: Pelatihan Terdistribusi

### Pengenalan KataGo Training

KataGo Training adalah jaringan pelatihan terdistribusi global:

- Relawan menyumbangkan daya komputasi GPU untuk menjalankan self-play
- Data self-play diunggah ke server pusat
- Server melatih model baru secara berkala
- Model baru didistribusikan ke relawan untuk melanjutkan permainan

Situs resmi: https://katagotraining.org/

### Langkah Partisipasi

#### 1. Buat Akun

Kunjungi https://katagotraining.org/ untuk mendaftar akun.

#### 2. Unduh KataGo

```bash
# Unduh versi terbaru
wget https://github.com/lightvector/KataGo/releases/download/v1.15.3/katago-v1.15.3-cuda11.1-linux-x64.zip
unzip katago-v1.15.3-cuda11.1-linux-x64.zip
```

#### 3. Konfigurasi Mode Contribute

```bash
# Jalankan pertama kali akan memandu Anda untuk mengkonfigurasi
./katago contribute -username YOUR_USERNAME -password YOUR_PASSWORD
```

Sistem akan otomatis:
- Mengunduh model terbaru
- Menjalankan self-play
- Mengunggah data permainan

#### 4. Jalankan di Background

```bash
# Gunakan screen atau tmux untuk menjalankan di background
screen -S katago
./katago contribute -username YOUR_USERNAME -password YOUR_PASSWORD
# Ctrl+A, D untuk keluar dari screen
```

### Statistik Kontribusi

Anda dapat melihat di https://katagotraining.org/contributions/:
- Peringkat kontribusi Anda
- Total jumlah permainan yang dikontribusikan
- Model yang baru dilatih

---

## Melaporkan Masalah

### Di Mana Melaporkan

- **GitHub Issues**: https://github.com/lightvector/KataGo/issues
- **Discord**: https://discord.gg/bqkZAz3

### Laporan Masalah yang Baik Mencakup

1. **Versi KataGo**: `katago version`
2. **Sistem Operasi**: Windows/Linux/macOS
3. **Hardware**: Model GPU, memori
4. **Pesan error lengkap**: Salin log lengkap
5. **Langkah reproduksi**: Bagaimana memicu masalah ini

### Contoh

```markdown
## Deskripsi Masalah
Error memori tidak cukup saat menjalankan benchmark

## Lingkungan
- Versi KataGo: 1.15.3
- Sistem Operasi: Ubuntu 22.04
- GPU: RTX 3060 12GB
- Model: kata-b40c256.bin.gz

## Pesan Error
```
CUDA error: out of memory
```

## Langkah Reproduksi
1. Jalankan `katago benchmark -model kata-b40c256.bin.gz`
2. Tunggu sekitar 30 detik
3. Error muncul
```

---

## Memperbaiki Dokumentasi

### Lokasi Dokumentasi

- **README**: `README.md`
- **Dokumentasi GTP**: `docs/GTP_Extensions.md`
- **Dokumentasi Analysis**: `docs/Analysis_Engine.md`
- **Dokumentasi Pelatihan**: `python/README.md`

### Alur Kontribusi

1. Fork proyek
2. Buat branch baru
3. Edit dokumentasi
4. Submit Pull Request

```bash
git clone https://github.com/YOUR_USERNAME/KataGo.git
cd KataGo
git checkout -b improve-docs
# Edit dokumentasi
git add .
git commit -m "Improve documentation for Analysis Engine"
git push origin improve-docs
# Buat Pull Request di GitHub
```

---

## Kontribusi Kode

### Pengaturan Lingkungan Pengembangan

```bash
# Clone proyek
git clone https://github.com/lightvector/KataGo.git
cd KataGo

# Kompilasi (mode Debug)
cd cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=OPENCL -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# Jalankan test
./katago runtests
```

### Gaya Kode

KataGo menggunakan gaya kode berikut:

**C++**:
- Indentasi 2 spasi
- Kurung kurawal di baris yang sama
- Nama variabel menggunakan camelCase
- Nama kelas menggunakan PascalCase

```cpp
class ExampleClass {
public:
  void exampleMethod() {
    int localVariable = 0;
    if(condition) {
      doSomething();
    }
  }
};
```

**Python**:
- Mengikuti PEP 8
- Indentasi 4 spasi

### Area Kontribusi

| Area | Lokasi File | Keahlian yang Diperlukan |
|------|-------------|-------------------------|
| Engine inti | `cpp/` | C++, CUDA/OpenCL |
| Program pelatihan | `python/` | Python, PyTorch |
| Protokol GTP | `cpp/command/gtp.cpp` | C++ |
| Analysis API | `cpp/command/analysis.cpp` | C++, JSON |
| Test | `cpp/tests/` | C++ |

### Alur Pull Request

1. **Buat Issue**: Diskusikan perubahan yang ingin Anda buat terlebih dahulu
2. **Fork & Clone**: Buat branch Anda sendiri
3. **Develop & Test**: Pastikan semua test lulus
4. **Submit PR**: Jelaskan konten perubahan secara detail
5. **Code Review**: Tanggapi feedback dari maintainer
6. **Merge**: Maintainer menggabungkan kode Anda

### Contoh PR

```markdown
## Deskripsi Perubahan
Menambahkan dukungan untuk aturan New Zealand

## Konten Perubahan
- Menambahkan aturan NEW_ZEALAND di rules.cpp
- Mengupdate perintah GTP untuk mendukung `kata-set-rules nz`
- Menambahkan unit test

## Hasil Test
- Semua test yang ada lulus
- Test baru lulus

## Issue Terkait
Fixes #123
```

---

## Sumber Daya Komunitas

### Link Resmi

| Sumber Daya | Link |
|-------------|------|
| GitHub | https://github.com/lightvector/KataGo |
| Discord | https://discord.gg/bqkZAz3 |
| Jaringan Pelatihan | https://katagotraining.org/ |

### Forum Diskusi

- **Discord**: Diskusi real-time, tanya jawab teknis
- **GitHub Discussions**: Diskusi panjang, proposal fitur
- **Reddit r/baduk**: Diskusi umum AI Go

### Proyek Terkait

| Proyek | Deskripsi | Link |
|--------|-----------|------|
| KaTrain | Alat analisis pengajaran | github.com/sanderland/katrain |
| Lizzie | Antarmuka analisis | github.com/featurecat/lizzie |
| Sabaki | Editor rekaman | sabaki.yichuanshen.de |
| BadukAI | Analisis online | baduk.ai |

---

## Pengakuan dan Penghargaan

### Daftar Kontributor

Semua kontributor akan tercantum di:
- Halaman GitHub Contributors
- Papan peringkat kontribusi KataGo Training

### Hasil Pembelajaran

Manfaat dari berpartisipasi dalam proyek open source:
- Belajar arsitektur sistem AI tingkat industri
- Berkomunikasi dengan pengembang di seluruh dunia
- Mengumpulkan catatan kontribusi open source
- Memahami mendalam teknologi AI Go

---

## Bacaan Lanjutan

- [Panduan Source Code](../source-code) — Memahami struktur kode
- [Analisis Mekanisme Pelatihan KataGo](../training) — Eksperimen pelatihan lokal
- [Memahami AI Go dalam Satu Artikel](../../how-it-works/) — Prinsip teknis
