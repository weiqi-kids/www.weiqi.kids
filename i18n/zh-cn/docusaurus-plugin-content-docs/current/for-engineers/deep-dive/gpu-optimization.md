---
sidebar_position: 6
title: GPU 后端与优化
description: KataGo 的 CUDA、OpenCL、Metal 后端比较与性能调优指南
---

# GPU 后端与优化

本文介绍 KataGo 支持的各种 GPU 后端、性能差异，以及如何调优以获得最佳性能。

---

## 后端总览

KataGo 支持四种计算后端：

| 后端 | 硬件支持 | 性能 | 安装难度 |
|------|---------|------|---------|
| **CUDA** | NVIDIA GPU | 最佳 | 中等 |
| **OpenCL** | NVIDIA/AMD/Intel GPU | 良好 | 简单 |
| **Metal** | Apple Silicon | 良好 | 简单 |
| **Eigen** | 纯 CPU | 较慢 | 最简单 |

---

## CUDA 后端

### 适用场景

- NVIDIA GPU（GTX 10 系列以上）
- 需要最高性能
- 有 CUDA 开发环境

### 安装需求

```bash
# 检查 CUDA 版本
nvcc --version

# 检查 cuDNN
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR
```

| 组件 | 建议版本 |
|------|---------|
| CUDA | 11.x 或 12.x |
| cuDNN | 8.x |
| 驱动程序 | 470+ |

### 编译

```bash
cd KataGo/cpp
mkdir build && cd build

cmake .. -DUSE_BACKEND=CUDA \
         -DCUDNN_INCLUDE_DIR=/usr/local/cuda/include \
         -DCUDNN_LIBRARY=/usr/local/cuda/lib64/libcudnn.so

make -j$(nproc)
```

### 性能特点

- **Tensor Cores**：支持 FP16 加速（RTX 系列）
- **批量推理**：GPU 利用率最佳
- **显存管理**：可精细控制 VRAM 用量

---

## OpenCL 后端

### 适用场景

- AMD GPU
- Intel 核显
- NVIDIA GPU（无 CUDA 环境）
- 跨平台部署

### 安装需求

```bash
# Linux - 安装 OpenCL 开发套件
sudo apt install ocl-icd-opencl-dev

# 检查可用的 OpenCL 设备
clinfo
```

### 编译

```bash
cmake .. -DUSE_BACKEND=OPENCL
make -j$(nproc)
```

### 驱动程序选择

| GPU 类型 | 建议驱动 |
|---------|---------|
| AMD | ROCm 或 AMDGPU-PRO |
| Intel | intel-opencl-icd |
| NVIDIA | nvidia-opencl-icd |

### 性能调优

```ini
# config.cfg
openclDeviceToUse = 0          # GPU 编号
openclUseFP16 = auto           # 半精度（支持时）
openclUseFP16Storage = true    # FP16 存储
```

---

## Metal 后端

### 适用场景

- Apple Silicon（M1/M2/M3）
- macOS 系统

### 编译

```bash
cmake .. -DUSE_BACKEND=METAL
make -j$(sysctl -n hw.ncpu)
```

### Apple Silicon 优化

Apple Silicon 的统一内存架构有特殊优势：

```ini
# Apple Silicon 建议配置
numNNServerThreadsPerModel = 1
nnMaxBatchSize = 16
numSearchThreads = 4
```

### 性能比较

| 芯片 | 相对性能 |
|------|---------|
| M1 | ~RTX 2060 |
| M1 Pro/Max | ~RTX 3060 |
| M2 | ~RTX 2070 |
| M3 Pro/Max | ~RTX 3070 |

---

## Eigen 后端（纯 CPU）

### 适用场景

- 无 GPU 环境
- 快速测试
- 低强度使用

### 编译

```bash
sudo apt install libeigen3-dev
cmake .. -DUSE_BACKEND=EIGEN
make -j$(nproc)
```

### 性能预期

```
CPU 单核：~10-30 playouts/sec
CPU 多核：~50-150 playouts/sec
GPU（中端）：~1000-3000 playouts/sec
```

---

## 性能调优参数

### 核心参数

```ini
# config.cfg

# === 神经网络配置 ===
# GPU 编号（多 GPU 时使用）
nnDeviceIdxs = 0

# 每个模型的推理线程数
numNNServerThreadsPerModel = 2

# 最大批量大小
nnMaxBatchSize = 16

# 缓存大小（2^N 个位置）
nnCacheSizePowerOfTwo = 20

# === 搜索配置 ===
# 搜索线程数
numSearchThreads = 8

# 每手最大访问次数
maxVisits = 800
```

### 参数调优指南

#### nnMaxBatchSize

```
太小：GPU 利用率低，推理延迟高
太大：显存不足，等待时间长

建议值：
- 4GB VRAM: 8-12
- 8GB VRAM: 16-24
- 16GB+ VRAM: 32-64
```

#### numSearchThreads

```
太少：无法喂饱 GPU
太多：CPU 瓶颈、内存压力

建议值：
- CPU 核心数的 1-2 倍
- 与 nnMaxBatchSize 相近
```

#### numNNServerThreadsPerModel

```
CUDA：1-2
OpenCL：1-2
Eigen：CPU 核心数
```

### 显存调优

```ini
# 减少显存用量
nnMaxBatchSize = 8
nnCacheSizePowerOfTwo = 18

# 增加显存用量（提升性能）
nnMaxBatchSize = 32
nnCacheSizePowerOfTwo = 22
```

---

## 多 GPU 配置

### 单机多 GPU

```ini
# 使用 GPU 0 和 GPU 1
nnDeviceIdxs = 0,1

# 每个 GPU 的线程数
numNNServerThreadsPerModel = 2
```

### 负载均衡

```ini
# 根据 GPU 性能分配权重
# GPU 0 性能较强，分配更多工作
nnDeviceIdxs = 0,0,1
```

---

## 基准测试

### 执行基准测试

```bash
katago benchmark -model model.bin.gz -config config.cfg
```

### 输出解读

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

### 常见性能数据

| GPU | 模型 | Playouts/秒 |
|-----|------|------------|
| RTX 3060 | b18c384 | ~2500 |
| RTX 3080 | b18c384 | ~4500 |
| RTX 4090 | b18c384 | ~8000 |
| M1 Pro | b18c384 | ~1500 |
| M2 Max | b18c384 | ~2200 |

---

## TensorRT 加速

### 适用场景

- NVIDIA GPU
- 追求极致性能
- 可接受较长的初始化时间

### 启用方式

```bash
# 编译时启用
cmake .. -DUSE_BACKEND=CUDA -DUSE_TENSORRT=ON

# 或使用预编译版本
katago-tensorrt
```

### 性能提升

```
标准 CUDA：100%
TensorRT FP32：+20-30%
TensorRT FP16：+50-80%（RTX 系列）
TensorRT INT8：+100-150%（需要校准）
```

### 注意事项

- 首次启动需要编译 TensorRT 引擎（数分钟）
- 不同 GPU 需要重新编译
- FP16/INT8 可能略微降低精度

---

## 常见问题

### GPU 未被检测

```bash
# 检查 GPU 状态
nvidia-smi  # NVIDIA
rocm-smi    # AMD
clinfo      # OpenCL

# KataGo 列出可用 GPU
katago gpuinfo
```

### 显存不足

```ini
# 使用较小的模型
# b18c384 → b10c128

# 减少批量大小
nnMaxBatchSize = 4

# 减少缓存
nnCacheSizePowerOfTwo = 16
```

### 性能不如预期

1. 确认使用正确的后端（CUDA > OpenCL > Eigen）
2. 检查 `numSearchThreads` 是否足够
3. 确认 GPU 没有被其他程序占用
4. 使用 `benchmark` 命令确认性能

---

## 性能优化检查清单

- [ ] 选择正确的后端（CUDA/OpenCL/Metal）
- [ ] 安装最新的 GPU 驱动程序
- [ ] 调整 `nnMaxBatchSize` 符合显存
- [ ] 调整 `numSearchThreads` 符合 CPU
- [ ] 执行 `benchmark` 确认性能
- [ ] 监控 GPU 使用率（应 > 80%）

---

## 延伸阅读

- [MCTS 实现细节](../mcts-implementation) — 批量推理的需求来源
- [模型量化与部署](../quantization-deploy) — 进一步的性能优化
- [完整安装指南](../../hands-on/setup) — 各平台安装步骤
