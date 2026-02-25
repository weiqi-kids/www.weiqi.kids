---
sidebar_position: 6
title: GPUバックエンドと最適化
description: KataGoのCUDA、OpenCL、Metalバックエンド比較とパフォーマンスチューニングガイド
---

# GPUバックエンドと最適化

本記事では、KataGoがサポートする各種GPUバックエンド、パフォーマンスの違い、最適なパフォーマンスを得るためのチューニング方法を紹介します。

---

## バックエンド概要

KataGoは4種類の計算バックエンドをサポートしています：

| バックエンド | ハードウェアサポート | パフォーマンス | インストール難易度 |
|------|---------|------|---------|
| **CUDA** | NVIDIA GPU | 最高 | 中程度 |
| **OpenCL** | NVIDIA/AMD/Intel GPU | 良好 | 簡単 |
| **Metal** | Apple Silicon | 良好 | 簡単 |
| **Eigen** | CPU専用 | 遅い | 最も簡単 |

---

## CUDAバックエンド

### 適用シナリオ

- NVIDIA GPU（GTX 10シリーズ以上）
- 最高パフォーマンスが必要
- CUDA開発環境がある

### インストール要件

```bash
# CUDAバージョンを確認
nvcc --version

# cuDNNを確認
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR
```

| コンポーネント | 推奨バージョン |
|------|---------|
| CUDA | 11.x または 12.x |
| cuDNN | 8.x |
| ドライバ | 470以上 |

### コンパイル

```bash
cd KataGo/cpp
mkdir build && cd build

cmake .. -DUSE_BACKEND=CUDA \
         -DCUDNN_INCLUDE_DIR=/usr/local/cuda/include \
         -DCUDNN_LIBRARY=/usr/local/cuda/lib64/libcudnn.so

make -j$(nproc)
```

### パフォーマンス特性

- **Tensor Cores**：FP16アクセラレーションをサポート（RTXシリーズ）
- **バッチ推論**：GPU使用率が最適
- **メモリ管理**：VRAM使用量を精密に制御可能

---

## OpenCLバックエンド

### 適用シナリオ

- AMD GPU
- Intel内蔵グラフィックス
- NVIDIA GPU（CUDA環境なし）
- クロスプラットフォームデプロイ

### インストール要件

```bash
# Linux - OpenCL開発キットをインストール
sudo apt install ocl-icd-opencl-dev

# 利用可能なOpenCLデバイスを確認
clinfo
```

### コンパイル

```bash
cmake .. -DUSE_BACKEND=OPENCL
make -j$(nproc)
```

### ドライバの選択

| GPUタイプ | 推奨ドライバ |
|---------|---------|
| AMD | ROCm または AMDGPU-PRO |
| Intel | intel-opencl-icd |
| NVIDIA | nvidia-opencl-icd |

### パフォーマンスチューニング

```ini
# config.cfg
openclDeviceToUse = 0          # GPU番号
openclUseFP16 = auto           # 半精度（サポート時）
openclUseFP16Storage = true    # FP16ストレージ
```

---

## Metalバックエンド

### 適用シナリオ

- Apple Silicon（M1/M2/M3）
- macOSシステム

### コンパイル

```bash
cmake .. -DUSE_BACKEND=METAL
make -j$(sysctl -n hw.ncpu)
```

### Apple Silicon最適化

Apple Siliconのユニファイドメモリアーキテクチャには特別な利点があります：

```ini
# Apple Silicon推奨設定
numNNServerThreadsPerModel = 1
nnMaxBatchSize = 16
numSearchThreads = 4
```

### パフォーマンス比較

| チップ | 相対パフォーマンス |
|------|---------|
| M1 | ~RTX 2060 |
| M1 Pro/Max | ~RTX 3060 |
| M2 | ~RTX 2070 |
| M3 Pro/Max | ~RTX 3070 |

---

## Eigenバックエンド（CPU専用）

### 適用シナリオ

- GPU環境なし
- クイックテスト
- 軽度の使用

### コンパイル

```bash
sudo apt install libeigen3-dev
cmake .. -DUSE_BACKEND=EIGEN
make -j$(nproc)
```

### パフォーマンス予測

```
CPUシングルコア：~10-30 playouts/秒
CPUマルチコア：~50-150 playouts/秒
GPU（ミドルレンジ）：~1000-3000 playouts/秒
```

---

## パフォーマンスチューニングパラメータ

### コアパラメータ

```ini
# config.cfg

# === ニューラルネットワーク設定 ===
# GPU番号（マルチGPU時に使用）
nnDeviceIdxs = 0

# モデルごとの推論スレッド数
numNNServerThreadsPerModel = 2

# 最大バッチサイズ
nnMaxBatchSize = 16

# キャッシュサイズ（2^N個の位置）
nnCacheSizePowerOfTwo = 20

# === 探索設定 ===
# 探索スレッド数
numSearchThreads = 8

# 1手あたりの最大訪問回数
maxVisits = 800
```

### パラメータチューニングガイド

#### nnMaxBatchSize

```
小さすぎ：GPU使用率が低い、推論遅延が高い
大きすぎ：VRAM不足、待機時間が長い

推奨値：
- 4GB VRAM: 8-12
- 8GB VRAM: 16-24
- 16GB以上 VRAM: 32-64
```

#### numSearchThreads

```
少なすぎ：GPUを十分に活用できない
多すぎ：CPUボトルネック、メモリ圧力

推奨値：
- CPUコア数の1-2倍
- nnMaxBatchSizeに近い値
```

#### numNNServerThreadsPerModel

```
CUDA：1-2
OpenCL：1-2
Eigen：CPUコア数
```

### メモリチューニング

```ini
# VRAM使用量を削減
nnMaxBatchSize = 8
nnCacheSizePowerOfTwo = 18

# VRAM使用量を増やす（パフォーマンス向上）
nnMaxBatchSize = 32
nnCacheSizePowerOfTwo = 22
```

---

## マルチGPU設定

### シングルマシン・マルチGPU

```ini
# GPU 0とGPU 1を使用
nnDeviceIdxs = 0,1

# 各GPUのスレッド数
numNNServerThreadsPerModel = 2
```

### 負荷分散

```ini
# GPUパフォーマンスに応じて重みを割り当て
# GPU 0の方が高性能で、より多くの作業を割り当て
nnDeviceIdxs = 0,0,1
```

---

## ベンチマーク

### ベンチマークの実行

```bash
katago benchmark -model model.bin.gz -config config.cfg
```

### 出力の解釈

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

### 一般的なパフォーマンスデータ

| GPU | モデル | Playouts/秒 |
|-----|------|------------|
| RTX 3060 | b18c384 | ~2500 |
| RTX 3080 | b18c384 | ~4500 |
| RTX 4090 | b18c384 | ~8000 |
| M1 Pro | b18c384 | ~1500 |
| M2 Max | b18c384 | ~2200 |

---

## TensorRTアクセラレーション

### 適用シナリオ

- NVIDIA GPU
- 最高パフォーマンスを追求
- 長い初期化時間を許容できる

### 有効化方法

```bash
# コンパイル時に有効化
cmake .. -DUSE_BACKEND=CUDA -DUSE_TENSORRT=ON

# またはプリコンパイル版を使用
katago-tensorrt
```

### パフォーマンス向上

```
標準CUDA：100%
TensorRT FP32：+20-30%
TensorRT FP16：+50-80%（RTXシリーズ）
TensorRT INT8：+100-150%（キャリブレーションが必要）
```

### 注意事項

- 初回起動時にTensorRTエンジンのコンパイルが必要（数分）
- 異なるGPUでは再コンパイルが必要
- FP16/INT8は精度がわずかに低下する可能性あり

---

## よくある問題

### GPUが検出されない

```bash
# GPUステータスを確認
nvidia-smi  # NVIDIA
rocm-smi    # AMD
clinfo      # OpenCL

# KataGoで利用可能なGPUをリスト
katago gpuinfo
```

### VRAM不足

```ini
# より小さいモデルを使用
# b18c384 → b10c128

# バッチサイズを削減
nnMaxBatchSize = 4

# キャッシュを削減
nnCacheSizePowerOfTwo = 16
```

### パフォーマンスが期待通りでない

1. 正しいバックエンドを使用しているか確認（CUDA > OpenCL > Eigen）
2. `numSearchThreads`が十分か確認
3. GPUが他のプログラムに占有されていないか確認
4. `benchmark`コマンドでパフォーマンスを確認

---

## パフォーマンス最適化チェックリスト

- [ ] 正しいバックエンドを選択（CUDA/OpenCL/Metal）
- [ ] 最新のGPUドライバをインストール
- [ ] `nnMaxBatchSize`をVRAMに合わせて調整
- [ ] `numSearchThreads`をCPUに合わせて調整
- [ ] `benchmark`でパフォーマンスを確認
- [ ] GPU使用率を監視（80%以上が目標）

---

## 関連記事

- [MCTS実装詳細](../mcts-implementation) — バッチ推論が必要な理由
- [モデル量子化とデプロイメント](../quantization-deploy) — さらなるパフォーマンス最適化
- [完全インストールガイド](../../hands-on/setup) — 各プラットフォームのインストール手順
