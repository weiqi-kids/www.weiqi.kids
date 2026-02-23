---
sidebar_position: 6
title: Backend GPU dan Optimasi
description: Panduan perbandingan dan tuning performa backend CUDA, OpenCL, Metal KataGo
---

# Backend GPU dan Optimasi

Artikel ini memperkenalkan berbagai backend GPU yang didukung KataGo, perbedaan performa, dan cara melakukan tuning untuk mendapatkan performa optimal.

---

## Gambaran Backend

KataGo mendukung empat backend komputasi:

| Backend | Dukungan Hardware | Performa | Kesulitan Instalasi |
|---------|-------------------|----------|---------------------|
| **CUDA** | GPU NVIDIA | Terbaik | Sedang |
| **OpenCL** | GPU NVIDIA/AMD/Intel | Baik | Mudah |
| **Metal** | Apple Silicon | Baik | Mudah |
| **Eigen** | CPU saja | Lebih lambat | Paling mudah |

---

## Backend CUDA

### Skenario Penggunaan

- GPU NVIDIA (GTX seri 10 ke atas)
- Membutuhkan performa tertinggi
- Memiliki lingkungan pengembangan CUDA

### Kebutuhan Instalasi

```bash
# Periksa versi CUDA
nvcc --version

# Periksa cuDNN
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR
```

| Komponen | Versi yang Disarankan |
|----------|----------------------|
| CUDA | 11.x atau 12.x |
| cuDNN | 8.x |
| Driver | 470+ |

### Kompilasi

```bash
cd KataGo/cpp
mkdir build && cd build

cmake .. -DUSE_BACKEND=CUDA \
         -DCUDNN_INCLUDE_DIR=/usr/local/cuda/include \
         -DCUDNN_LIBRARY=/usr/local/cuda/lib64/libcudnn.so

make -j$(nproc)
```

### Karakteristik Performa

- **Tensor Cores**: Mendukung akselerasi FP16 (seri RTX)
- **Inferensi Batch**: Utilisasi GPU optimal
- **Manajemen Memori**: Kontrol VRAM yang presisi

---

## Backend OpenCL

### Skenario Penggunaan

- GPU AMD
- GPU terintegrasi Intel
- GPU NVIDIA (tanpa lingkungan CUDA)
- Deployment lintas platform

### Kebutuhan Instalasi

```bash
# Linux - Instal paket pengembangan OpenCL
sudo apt install ocl-icd-opencl-dev

# Periksa perangkat OpenCL yang tersedia
clinfo
```

### Kompilasi

```bash
cmake .. -DUSE_BACKEND=OPENCL
make -j$(nproc)
```

### Pilihan Driver

| Tipe GPU | Driver yang Disarankan |
|----------|------------------------|
| AMD | ROCm atau AMDGPU-PRO |
| Intel | intel-opencl-icd |
| NVIDIA | nvidia-opencl-icd |

### Tuning Performa

```ini
# config.cfg
openclDeviceToUse = 0          # Nomor GPU
openclUseFP16 = auto           # Half precision (jika didukung)
openclUseFP16Storage = true    # Penyimpanan FP16
```

---

## Backend Metal

### Skenario Penggunaan

- Apple Silicon (M1/M2/M3)
- Sistem macOS

### Kompilasi

```bash
cmake .. -DUSE_BACKEND=METAL
make -j$(sysctl -n hw.ncpu)
```

### Optimasi Apple Silicon

Arsitektur unified memory Apple Silicon memiliki keunggulan khusus:

```ini
# Pengaturan yang disarankan untuk Apple Silicon
numNNServerThreadsPerModel = 1
nnMaxBatchSize = 16
numSearchThreads = 4
```

### Perbandingan Performa

| Chip | Performa Relatif |
|------|------------------|
| M1 | ~RTX 2060 |
| M1 Pro/Max | ~RTX 3060 |
| M2 | ~RTX 2070 |
| M3 Pro/Max | ~RTX 3070 |

---

## Backend Eigen (CPU Saja)

### Skenario Penggunaan

- Lingkungan tanpa GPU
- Pengujian cepat
- Penggunaan intensitas rendah

### Kompilasi

```bash
sudo apt install libeigen3-dev
cmake .. -DUSE_BACKEND=EIGEN
make -j$(nproc)
```

### Ekspektasi Performa

```
CPU single-core: ~10-30 playouts/sec
CPU multi-core: ~50-150 playouts/sec
GPU (mid-range): ~1000-3000 playouts/sec
```

---

## Parameter Tuning Performa

### Parameter Inti

```ini
# config.cfg

# === Pengaturan Neural Network ===
# Nomor GPU (untuk multi-GPU)
nnDeviceIdxs = 0

# Jumlah thread inferensi per model
numNNServerThreadsPerModel = 2

# Ukuran batch maksimum
nnMaxBatchSize = 16

# Ukuran cache (2^N posisi)
nnCacheSizePowerOfTwo = 20

# === Pengaturan Pencarian ===
# Jumlah thread pencarian
numSearchThreads = 8

# Maksimum kunjungan per langkah
maxVisits = 800
```

### Panduan Tuning Parameter

#### nnMaxBatchSize

```
Terlalu kecil: Utilisasi GPU rendah, latensi inferensi tinggi
Terlalu besar: VRAM tidak cukup, waktu tunggu lama

Nilai yang disarankan:
- 4GB VRAM: 8-12
- 8GB VRAM: 16-24
- 16GB+ VRAM: 32-64
```

#### numSearchThreads

```
Terlalu sedikit: Tidak bisa memenuhi kebutuhan GPU
Terlalu banyak: Bottleneck CPU, tekanan memori

Nilai yang disarankan:
- 1-2x jumlah core CPU
- Mendekati nnMaxBatchSize
```

#### numNNServerThreadsPerModel

```
CUDA: 1-2
OpenCL: 1-2
Eigen: Jumlah core CPU
```

### Tuning Memori

```ini
# Kurangi penggunaan VRAM
nnMaxBatchSize = 8
nnCacheSizePowerOfTwo = 18

# Tingkatkan penggunaan VRAM (tingkatkan performa)
nnMaxBatchSize = 32
nnCacheSizePowerOfTwo = 22
```

---

## Konfigurasi Multi-GPU

### Multi-GPU Satu Mesin

```ini
# Gunakan GPU 0 dan GPU 1
nnDeviceIdxs = 0,1

# Jumlah thread per GPU
numNNServerThreadsPerModel = 2
```

### Load Balancing

```ini
# Alokasikan bobot berdasarkan performa GPU
# GPU 0 lebih kuat, alokasikan lebih banyak pekerjaan
nnDeviceIdxs = 0,0,1
```

---

## Benchmark

### Menjalankan Benchmark

```bash
katago benchmark -model model.bin.gz -config config.cfg
```

### Interpretasi Output

```
GPU 0: NVIDIA GeForce RTX 3080
Threads: 8, Batch Size: 16

Benchmark results:
- Neural net evals/second: 2847.3
- Playouts/second: 4521.8
- Time per move (1000 visits): 0.221 sec

Memory usage:
- Peak GPU memory: 2.1 GB
- Peak system memory: 1.3 GB
```

### Data Performa Umum

| GPU | Model | Playouts/detik |
|-----|-------|----------------|
| RTX 3060 | b18c384 | ~2500 |
| RTX 3080 | b18c384 | ~4500 |
| RTX 4090 | b18c384 | ~8000 |
| M1 Pro | b18c384 | ~1500 |
| M2 Max | b18c384 | ~2200 |

---

## Akselerasi TensorRT

### Skenario Penggunaan

- GPU NVIDIA
- Mengejar performa ekstrem
- Dapat menerima waktu inisialisasi lebih lama

### Cara Mengaktifkan

```bash
# Aktifkan saat kompilasi
cmake .. -DUSE_BACKEND=CUDA -DUSE_TENSORRT=ON

# Atau gunakan versi pre-compiled
katago-tensorrt
```

### Peningkatan Performa

```
CUDA standar: 100%
TensorRT FP32: +20-30%
TensorRT FP16: +50-80% (seri RTX)
TensorRT INT8: +100-150% (perlu kalibrasi)
```

### Catatan

- Startup pertama perlu kompilasi engine TensorRT (beberapa menit)
- GPU berbeda perlu kompilasi ulang
- FP16/INT8 mungkin sedikit mengurangi presisi

---

## Masalah Umum

### GPU Tidak Terdeteksi

```bash
# Periksa status GPU
nvidia-smi  # NVIDIA
rocm-smi    # AMD
clinfo      # OpenCL

# KataGo list GPU yang tersedia
katago gpuinfo
```

### VRAM Tidak Cukup

```ini
# Gunakan model lebih kecil
# b18c384 → b10c128

# Kurangi batch size
nnMaxBatchSize = 4

# Kurangi cache
nnCacheSizePowerOfTwo = 16
```

### Performa Tidak Sesuai Harapan

1. Pastikan menggunakan backend yang benar (CUDA > OpenCL > Eigen)
2. Periksa apakah `numSearchThreads` cukup
3. Pastikan GPU tidak digunakan program lain
4. Gunakan perintah `benchmark` untuk konfirmasi performa

---

## Checklist Optimasi Performa

- [ ] Pilih backend yang benar (CUDA/OpenCL/Metal)
- [ ] Instal driver GPU terbaru
- [ ] Sesuaikan `nnMaxBatchSize` dengan VRAM
- [ ] Sesuaikan `numSearchThreads` dengan CPU
- [ ] Jalankan `benchmark` untuk konfirmasi performa
- [ ] Monitor utilisasi GPU (seharusnya > 80%)

---

## Bacaan Lanjutan

- [Detail Implementasi MCTS](../mcts-implementation) — Sumber kebutuhan inferensi batch
- [Kuantisasi Model dan Deployment](../quantization-deploy) — Optimasi performa lebih lanjut
- [Panduan Instalasi Lengkap](../../hands-on/setup) — Langkah instalasi per platform
