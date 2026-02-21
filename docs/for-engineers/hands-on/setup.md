---
sidebar_position: 2
title: 完整安裝指南
description: 各平台詳細安裝步驟、模型選擇與設定檔說明
---

# KataGo 完整安裝指南

本文詳細介紹在各平台上安裝 KataGo 的完整步驟。

## 系統需求

### 硬體需求

#### GPU（推薦）

| GPU 類型 | 支援狀態 | 建議後端 |
|---------|---------|---------|
| NVIDIA（CUDA） | 最佳支援 | CUDA |
| NVIDIA（無 CUDA） | 良好支援 | OpenCL |
| AMD | 良好支援 | OpenCL |
| Intel 內顯 | 基本支援 | OpenCL |
| Apple Silicon | 良好支援 | Metal / OpenCL |

#### CPU 模式

如果沒有合適的 GPU，可以使用 Eigen 後端純 CPU 運行：
- 效能較低（約 10-30 playouts/sec）
- 適合學習、測試和低強度使用
- 需要 AVX2 指令集支援

### 後端選擇

```
你有 NVIDIA GPU？
├─ 是 → 使用 CUDA 後端（最佳效能）
└─ 否 → 你有其他 GPU（AMD/Intel/Apple）？
         ├─ 是 → 使用 OpenCL 後端
         └─ 否 → 使用 Eigen 後端（純 CPU）
```

---

## macOS 安裝

### 方法 1：Homebrew（推薦）

```bash
brew install katago
katago version
```

### 方法 2：從原始碼編譯

```bash
# 安裝依賴
brew install cmake

# 克隆原始碼
git clone https://github.com/lightvector/KataGo.git
cd KataGo/cpp
mkdir build && cd build

# OpenCL 後端（推薦）
cmake .. -DUSE_BACKEND=OPENCL
make -j$(sysctl -n hw.ncpu)

# 測試
./katago version
```

#### Apple Silicon 特別說明

M1/M2/M3 Mac 建議使用 OpenCL 後端：

```bash
cmake .. -DUSE_BACKEND=OPENCL
```

Metal 後端（實驗性）：

```bash
cmake .. -DUSE_BACKEND=METAL
```

---

## Linux 安裝

### 方法 1：預編譯版本（推薦）

```bash
# OpenCL 版本（通用）
wget https://github.com/lightvector/KataGo/releases/download/v1.15.3/katago-v1.15.3-opencl-linux-x64.zip
unzip katago-v1.15.3-opencl-linux-x64.zip
chmod +x katago

# CUDA 版本（NVIDIA GPU）
wget https://github.com/lightvector/KataGo/releases/download/v1.15.3/katago-v1.15.3-cuda11.1-linux-x64.zip
```

### 方法 2：從原始碼編譯

#### CUDA 後端

```bash
sudo apt update
sudo apt install cmake g++ libzip-dev

git clone https://github.com/lightvector/KataGo.git
cd KataGo/cpp
mkdir build && cd build

cmake .. -DUSE_BACKEND=CUDA
make -j$(nproc)
```

#### OpenCL 後端

```bash
sudo apt install cmake g++ libzip-dev ocl-icd-opencl-dev

# 安裝 OpenCL 驅動
# AMD: sudo apt install mesa-opencl-icd
# Intel: sudo apt install intel-opencl-icd

cmake .. -DUSE_BACKEND=OPENCL
make -j$(nproc)
```

#### Eigen 後端（純 CPU）

```bash
sudo apt install cmake g++ libzip-dev libeigen3-dev

cmake .. -DUSE_BACKEND=EIGEN
make -j$(nproc)
```

---

## Windows 安裝

### 方法 1：預編譯版本（推薦）

1. 前往 [KataGo Releases](https://github.com/lightvector/KataGo/releases)
2. 下載適合的版本：
   - CUDA：`katago-v1.15.3-cuda11.1-windows-x64.zip`
   - OpenCL：`katago-v1.15.3-opencl-windows-x64.zip`
   - CPU：`katago-v1.15.3-eigen-windows-x64.zip`
3. 解壓縮並測試：`katago.exe version`

### 方法 2：從原始碼編譯

1. 安裝 Visual Studio 2019/2022（含 C++ 工具）
2. 安裝 CMake
3. 如果使用 CUDA，安裝 CUDA Toolkit

```cmd
git clone https://github.com/lightvector/KataGo.git
cd KataGo\cpp
mkdir build && cd build

cmake .. -G "Visual Studio 17 2022" -A x64 -DUSE_BACKEND=CUDA
cmake --build . --config Release
```

---

## 模型選擇

### 下載位置

官方模型：https://katagotraining.org/

```bash
# b18c384（推薦，平衡）
curl -L -o kata-b18c384.bin.gz \
  "https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz"

# b40c256（較強）
curl -L -o kata-b40c256.bin.gz \
  "https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b40c256-s11840935168-d2898845681.bin.gz"
```

### 模型比較

| 模型 | 檔案大小 | 棋力 | 適用場景 |
|------|---------|------|---------|
| b10c128 | ~20 MB | 業餘高段 | CPU、快速測試 |
| b18c384 | ~140 MB | 職業水準 | 一般 GPU |
| b40c256 | ~250 MB | 超人水準 | 高階 GPU |
| b60c320 | ~500 MB | 頂級超人 | 頂級 GPU |

---

## 設定檔

### 重要設定項目

建立 `my_config.cfg`：

```ini
# 搜索設定
maxVisits = 500
numSearchThreads = 4

# 規則
rules = chinese
komi = 7.5

# GPU 設定
nnDeviceIdxs = 0

# 日誌
logDir = ./logs
logToStderr = false
```

### 常用設定說明

| 設定項 | 說明 | 建議值 |
|--------|------|--------|
| `maxVisits` | 每手最大搜索次數 | 500-2000 |
| `numSearchThreads` | CPU 執行緒數 | CPU 核心數 |
| `nnDeviceIdxs` | GPU 編號 | 0 |
| `rules` | 圍棋規則 | chinese/japanese |

---

## 驗證安裝

### 基準測試

```bash
katago benchmark -model kata-b18c384.bin.gz -v 500
```

### 列出 GPU

```bash
katago gpuinfo
```

### 測試 GTP

```bash
katago gtp -model kata-b18c384.bin.gz
```

輸入 `name` 和 `version` 確認正常運作。

---

## 常見問題

### GPU 相關

**找不到 GPU**：

```bash
clinfo  # 檢查 OpenCL
katago gpuinfo  # 檢查 KataGo 可見的 GPU
```

**CUDA 初始化失敗**：
- 確認 CUDA 版本與 KataGo 編譯版本相符
- 更新 GPU 驅動程式

### 記憶體相關

**記憶體不足**：

```ini
nnMaxBatchSize = 8
nnCacheSizePowerOfTwo = 20
```

### 效能相關

**速度太慢**：
1. 確認使用 GPU 而非 CPU
2. 減少 `numSearchThreads`
3. 使用較小的模型

---

## 延伸閱讀

- [基本使用](../basic-usage) — GTP 與 Analysis Engine
- [整合到你的專案](../integration) — API 範例
