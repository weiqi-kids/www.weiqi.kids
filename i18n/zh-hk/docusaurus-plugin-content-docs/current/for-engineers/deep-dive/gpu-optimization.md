---
sidebar_position: 6
title: GPU 後端與優化
description: KataGo 嘅 CUDA、OpenCL、Metal 後端比較同效能調校指南
---

# GPU 後端與優化

本文介紹 KataGo 支援嘅各種 GPU 後端、效能差異，以及點樣調校以獲得最佳效能。

---

## 後端總覽

KataGo 支援四種計算後端：

| 後端 | 硬件支援 | 效能 | 安裝難度 |
|------|---------|------|---------|
| **CUDA** | NVIDIA GPU | 最好 | 中等 |
| **OpenCL** | NVIDIA/AMD/Intel GPU | 良好 | 簡單 |
| **Metal** | Apple Silicon | 良好 | 簡單 |
| **Eigen** | 純 CPU | 較慢 | 最簡單 |

---

## CUDA 後端

### 適用場景

- NVIDIA GPU（GTX 10 系列以上）
- 需要最高效能
- 有 CUDA 開發環境

### 安裝需求

```bash
# 檢查 CUDA 版本
nvcc --version

# 檢查 cuDNN
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR
```

| 元件 | 建議版本 |
|------|---------|
| CUDA | 11.x 或 12.x |
| cuDNN | 8.x |
| 驅動程式 | 470+ |

### 編譯

```bash
cd KataGo/cpp
mkdir build && cd build

cmake .. -DUSE_BACKEND=CUDA \
         -DCUDNN_INCLUDE_DIR=/usr/local/cuda/include \
         -DCUDNN_LIBRARY=/usr/local/cuda/lib64/libcudnn.so

make -j$(nproc)
```

### 效能特點

- **Tensor Cores**：支援 FP16 加速（RTX 系列）
- **批次推理**：GPU 利用率最好
- **記憶體管理**：可以精細控制 VRAM 用量

---

## OpenCL 後端

### 適用場景

- AMD GPU
- Intel 內顯
- NVIDIA GPU（冇 CUDA 環境）
- 跨平台部署

### 安裝需求

```bash
# Linux - 安裝 OpenCL 開發套件
sudo apt install ocl-icd-opencl-dev

# 檢查可用嘅 OpenCL 裝置
clinfo
```

### 編譯

```bash
cmake .. -DUSE_BACKEND=OPENCL
make -j$(nproc)
```

### 驅動程式選擇

| GPU 類型 | 建議驅動 |
|---------|---------|
| AMD | ROCm 或 AMDGPU-PRO |
| Intel | intel-opencl-icd |
| NVIDIA | nvidia-opencl-icd |

### 效能調校

```ini
# config.cfg
openclDeviceToUse = 0          # GPU 編號
openclUseFP16 = auto           # 半精度（支援嗰陣）
openclUseFP16Storage = true    # FP16 儲存
```

---

## Metal 後端

### 適用場景

- Apple Silicon（M1/M2/M3）
- macOS 系統

### 編譯

```bash
cmake .. -DUSE_BACKEND=METAL
make -j$(sysctl -n hw.ncpu)
```

### Apple Silicon 優化

Apple Silicon 嘅統一記憶體架構有特殊優勢：

```ini
# Apple Silicon 建議設定
numNNServerThreadsPerModel = 1
nnMaxBatchSize = 16
numSearchThreads = 4
```

### 效能比較

| 晶片 | 相對效能 |
|------|---------|
| M1 | ~RTX 2060 |
| M1 Pro/Max | ~RTX 3060 |
| M2 | ~RTX 2070 |
| M3 Pro/Max | ~RTX 3070 |

---

## Eigen 後端（純 CPU）

### 適用場景

- 冇 GPU 環境
- 快速測試
- 低強度使用

### 編譯

```bash
sudo apt install libeigen3-dev
cmake .. -DUSE_BACKEND=EIGEN
make -j$(nproc)
```

### 效能預期

```
CPU 單核：~10-30 playouts/sec
CPU 多核：~50-150 playouts/sec
GPU（中階）：~1000-3000 playouts/sec
```

---

## 效能調校參數

### 核心參數

```ini
# config.cfg

# === 神經網絡設定 ===
# GPU 編號（多 GPU 嗰陣用）
nnDeviceIdxs = 0

# 每個模型嘅推理執行緒數
numNNServerThreadsPerModel = 2

# 最大批次大小
nnMaxBatchSize = 16

# 快取大小（2^N 個位置）
nnCacheSizePowerOfTwo = 20

# === 搜索設定 ===
# 搜索執行緒數
numSearchThreads = 8

# 每手最大訪問次數
maxVisits = 800
```

### 參數調校指南

#### nnMaxBatchSize

```
太細：GPU 利用率低，推理延遲高
太大：VRAM 唔夠，等待時間長

建議值：
- 4GB VRAM: 8-12
- 8GB VRAM: 16-24
- 16GB+ VRAM: 32-64
```

#### numSearchThreads

```
太少：餵唔飽 GPU
太多：CPU 瓶頸、記憶體壓力

建議值：
- CPU 核心數嘅 1-2 倍
- 同 nnMaxBatchSize 相近
```

#### numNNServerThreadsPerModel

```
CUDA：1-2
OpenCL：1-2
Eigen：CPU 核心數
```

### 記憶體調校

```ini
# 減少 VRAM 用量
nnMaxBatchSize = 8
nnCacheSizePowerOfTwo = 18

# 增加 VRAM 用量（提升效能）
nnMaxBatchSize = 32
nnCacheSizePowerOfTwo = 22
```

---

## 多 GPU 設定

### 單機多 GPU

```ini
# 用 GPU 0 同 GPU 1
nnDeviceIdxs = 0,1

# 每個 GPU 嘅執行緒數
numNNServerThreadsPerModel = 2
```

### 負載平衡

```ini
# 根據 GPU 效能分配權重
# GPU 0 效能較強，分配更多工作
nnDeviceIdxs = 0,0,1
```

---

## 基準測試

### 執行基準測試

```bash
katago benchmark -model model.bin.gz -config config.cfg
```

### 輸出解讀

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

### 常見效能數據

| GPU | 模型 | Playouts/秒 |
|-----|------|------------|
| RTX 3060 | b18c384 | ~2500 |
| RTX 3080 | b18c384 | ~4500 |
| RTX 4090 | b18c384 | ~8000 |
| M1 Pro | b18c384 | ~1500 |
| M2 Max | b18c384 | ~2200 |

---

## TensorRT 加速

### 適用場景

- NVIDIA GPU
- 追求極致效能
- 可以接受較長嘅初始化時間

### 啟用方式

```bash
# 編譯嗰陣啟用
cmake .. -DUSE_BACKEND=CUDA -DUSE_TENSORRT=ON

# 或者用預編譯版本
katago-tensorrt
```

### 效能提升

```
標準 CUDA：100%
TensorRT FP32：+20-30%
TensorRT FP16：+50-80%（RTX 系列）
TensorRT INT8：+100-150%（需要校準）
```

### 注意事項

- 首次啟動需要編譯 TensorRT 引擎（幾分鐘）
- 唔同 GPU 需要重新編譯
- FP16/INT8 可能略微降低精度

---

## 常見問題

### GPU 未被偵測

```bash
# 檢查 GPU 狀態
nvidia-smi  # NVIDIA
rocm-smi    # AMD
clinfo      # OpenCL

# KataGo 列出可用 GPU
katago gpuinfo
```

### VRAM 唔夠

```ini
# 用較細嘅模型
# b18c384 → b10c128

# 減少批次大小
nnMaxBatchSize = 4

# 減少快取
nnCacheSizePowerOfTwo = 16
```

### 效能唔如預期

1. 確認用啱後端（CUDA > OpenCL > Eigen）
2. 檢查 `numSearchThreads` 係咪夠
3. 確認 GPU 冇俾其他程式佔用
4. 用 `benchmark` 指令確認效能

---

## 效能優化檢查清單

- [ ] 揀啱後端（CUDA/OpenCL/Metal）
- [ ] 安裝最新嘅 GPU 驅動程式
- [ ] 調整 `nnMaxBatchSize` 符合 VRAM
- [ ] 調整 `numSearchThreads` 符合 CPU
- [ ] 執行 `benchmark` 確認效能
- [ ] 監控 GPU 使用率（應該 > 80%）

---

## 延伸閱讀

- [MCTS 實作細節](../mcts-implementation) — 批次推理嘅需求來源
- [模型量化與部署](../quantization-deploy) — 進一步嘅效能優化
- [完整安裝指南](../../hands-on/setup) — 各平台安裝步驟
