---
sidebar_position: 6
title: GPU Backend & Optimization
description: KataGo's CUDA, OpenCL, Metal backend comparison and performance tuning guide
---

# GPU Backend & Optimization

This article introduces the various GPU backends supported by KataGo, performance differences, and how to tune for optimal performance.

---

## Backend Overview

KataGo supports four compute backends:

| Backend | Hardware Support | Performance | Installation Difficulty |
|---------|-----------------|-------------|------------------------|
| **CUDA** | NVIDIA GPU | Best | Medium |
| **OpenCL** | NVIDIA/AMD/Intel GPU | Good | Easy |
| **Metal** | Apple Silicon | Good | Easy |
| **Eigen** | CPU only | Slower | Easiest |

---

## CUDA Backend

### Use Cases

- NVIDIA GPU (GTX 10 series and above)
- Need maximum performance
- Have CUDA development environment

### Installation Requirements

```bash
# Check CUDA version
nvcc --version

# Check cuDNN
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR
```

| Component | Recommended Version |
|-----------|-------------------|
| CUDA | 11.x or 12.x |
| cuDNN | 8.x |
| Driver | 470+ |

### Compilation

```bash
cd KataGo/cpp
mkdir build && cd build

cmake .. -DUSE_BACKEND=CUDA \
         -DCUDNN_INCLUDE_DIR=/usr/local/cuda/include \
         -DCUDNN_LIBRARY=/usr/local/cuda/lib64/libcudnn.so

make -j$(nproc)
```

### Performance Characteristics

- **Tensor Cores**: Support FP16 acceleration (RTX series)
- **Batch inference**: Best GPU utilization
- **Memory management**: Fine-grained VRAM control

---

## OpenCL Backend

### Use Cases

- AMD GPU
- Intel integrated graphics
- NVIDIA GPU (no CUDA environment)
- Cross-platform deployment

### Installation Requirements

```bash
# Linux - Install OpenCL development kit
sudo apt install ocl-icd-opencl-dev

# Check available OpenCL devices
clinfo
```

### Compilation

```bash
cmake .. -DUSE_BACKEND=OPENCL
make -j$(nproc)
```

### Driver Selection

| GPU Type | Recommended Driver |
|----------|-------------------|
| AMD | ROCm or AMDGPU-PRO |
| Intel | intel-opencl-icd |
| NVIDIA | nvidia-opencl-icd |

### Performance Tuning

```ini
# config.cfg
openclDeviceToUse = 0          # GPU number
openclUseFP16 = auto           # Half precision (when supported)
openclUseFP16Storage = true    # FP16 storage
```

---

## Metal Backend

### Use Cases

- Apple Silicon (M1/M2/M3)
- macOS system

### Compilation

```bash
cmake .. -DUSE_BACKEND=METAL
make -j$(sysctl -n hw.ncpu)
```

### Apple Silicon Optimization

Apple Silicon's unified memory architecture has unique advantages:

```ini
# Apple Silicon recommended settings
numNNServerThreadsPerModel = 1
nnMaxBatchSize = 16
numSearchThreads = 4
```

### Performance Comparison

| Chip | Relative Performance |
|------|---------------------|
| M1 | ~RTX 2060 |
| M1 Pro/Max | ~RTX 3060 |
| M2 | ~RTX 2070 |
| M3 Pro/Max | ~RTX 3070 |

---

## Eigen Backend (CPU Only)

### Use Cases

- No GPU environment
- Quick testing
- Light usage

### Compilation

```bash
sudo apt install libeigen3-dev
cmake .. -DUSE_BACKEND=EIGEN
make -j$(nproc)
```

### Performance Expectations

```
CPU single core: ~10-30 playouts/sec
CPU multi-core: ~50-150 playouts/sec
GPU (mid-range): ~1000-3000 playouts/sec
```

---

## Performance Tuning Parameters

### Core Parameters

```ini
# config.cfg

# === Neural Network Settings ===
# GPU number (for multi-GPU)
nnDeviceIdxs = 0

# Inference threads per model
numNNServerThreadsPerModel = 2

# Maximum batch size
nnMaxBatchSize = 16

# Cache size (2^N positions)
nnCacheSizePowerOfTwo = 20

# === Search Settings ===
# Search threads
numSearchThreads = 8

# Max visits per move
maxVisits = 800
```

### Parameter Tuning Guide

#### nnMaxBatchSize

```
Too small: Low GPU utilization, high inference latency
Too large: VRAM insufficient, long wait times

Recommended values:
- 4GB VRAM: 8-12
- 8GB VRAM: 16-24
- 16GB+ VRAM: 32-64
```

#### numSearchThreads

```
Too few: Cannot feed GPU enough work
Too many: CPU bottleneck, memory pressure

Recommended values:
- 1-2x CPU core count
- Close to nnMaxBatchSize
```

#### numNNServerThreadsPerModel

```
CUDA: 1-2
OpenCL: 1-2
Eigen: CPU core count
```

### Memory Tuning

```ini
# Reduce VRAM usage
nnMaxBatchSize = 8
nnCacheSizePowerOfTwo = 18

# Increase VRAM usage (better performance)
nnMaxBatchSize = 32
nnCacheSizePowerOfTwo = 22
```

---

## Multi-GPU Configuration

### Single Machine Multi-GPU

```ini
# Use GPU 0 and GPU 1
nnDeviceIdxs = 0,1

# Threads per GPU
numNNServerThreadsPerModel = 2
```

### Load Balancing

```ini
# Allocate weights based on GPU performance
# GPU 0 is stronger, assign more work
nnDeviceIdxs = 0,0,1
```

---

## Benchmarking

### Run Benchmark

```bash
katago benchmark -model model.bin.gz -config config.cfg
```

### Output Interpretation

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

### Common Performance Data

| GPU | Model | Playouts/sec |
|-----|-------|--------------|
| RTX 3060 | b18c384 | ~2500 |
| RTX 3080 | b18c384 | ~4500 |
| RTX 4090 | b18c384 | ~8000 |
| M1 Pro | b18c384 | ~1500 |
| M2 Max | b18c384 | ~2200 |

---

## TensorRT Acceleration

### Use Cases

- NVIDIA GPU
- Pursuing maximum performance
- Can accept longer initialization time

### Enabling

```bash
# Enable during compilation
cmake .. -DUSE_BACKEND=CUDA -DUSE_TENSORRT=ON

# Or use precompiled version
katago-tensorrt
```

### Performance Improvement

```
Standard CUDA: 100%
TensorRT FP32: +20-30%
TensorRT FP16: +50-80% (RTX series)
TensorRT INT8: +100-150% (requires calibration)
```

### Notes

- First launch needs to compile TensorRT engine (several minutes)
- Different GPUs need recompilation
- FP16/INT8 may slightly reduce accuracy

---

## Common Issues

### GPU Not Detected

```bash
# Check GPU status
nvidia-smi  # NVIDIA
rocm-smi    # AMD
clinfo      # OpenCL

# KataGo list available GPUs
katago gpuinfo
```

### VRAM Insufficient

```ini
# Use smaller model
# b18c384 → b10c128

# Reduce batch size
nnMaxBatchSize = 4

# Reduce cache
nnCacheSizePowerOfTwo = 16
```

### Performance Below Expectations

1. Confirm using correct backend (CUDA > OpenCL > Eigen)
2. Check if `numSearchThreads` is sufficient
3. Confirm GPU is not occupied by other programs
4. Use `benchmark` command to verify performance

---

## Performance Optimization Checklist

- [ ] Select correct backend (CUDA/OpenCL/Metal)
- [ ] Install latest GPU drivers
- [ ] Adjust `nnMaxBatchSize` to match VRAM
- [ ] Adjust `numSearchThreads` to match CPU
- [ ] Run `benchmark` to verify performance
- [ ] Monitor GPU utilization (should be > 80%)

---

## Further Reading

- [MCTS Implementation Details](../mcts-implementation) — Source of batch inference needs
- [Model Quantization & Deployment](../quantization-deploy) — Further performance optimization
- [Complete Installation Guide](../../hands-on/setup) — Platform-specific installation steps
