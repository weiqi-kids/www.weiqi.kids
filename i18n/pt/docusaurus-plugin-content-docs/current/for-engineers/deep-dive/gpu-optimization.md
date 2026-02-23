---
sidebar_position: 6
title: Backend GPU e Otimização
description: Comparação e guia de otimização de desempenho para backends CUDA, OpenCL e Metal do KataGo
---

# Backend GPU e Otimização

Este artigo apresenta os vários backends de GPU suportados pelo KataGo, diferenças de desempenho e como ajustar para obter o melhor desempenho.

---

## Visão Geral dos Backends

O KataGo suporta quatro backends de computação:

| Backend | Suporte de Hardware | Desempenho | Dificuldade de Instalação |
|---------|---------------------|------------|---------------------------|
| **CUDA** | GPU NVIDIA | Melhor | Média |
| **OpenCL** | GPU NVIDIA/AMD/Intel | Bom | Simples |
| **Metal** | Apple Silicon | Bom | Simples |
| **Eigen** | CPU pura | Mais lento | Mais simples |

---

## Backend CUDA

### Cenários de Uso

- GPU NVIDIA (GTX série 10 e superior)
- Necessita do melhor desempenho
- Tem ambiente de desenvolvimento CUDA

### Requisitos de Instalação

```bash
# Verificar versão do CUDA
nvcc --version

# Verificar cuDNN
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR
```

| Componente | Versão Recomendada |
|------------|-------------------|
| CUDA | 11.x ou 12.x |
| cuDNN | 8.x |
| Driver | 470+ |

### Compilação

```bash
cd KataGo/cpp
mkdir build && cd build

cmake .. -DUSE_BACKEND=CUDA \
         -DCUDNN_INCLUDE_DIR=/usr/local/cuda/include \
         -DCUDNN_LIBRARY=/usr/local/cuda/lib64/libcudnn.so

make -j$(nproc)
```

### Características de Desempenho

- **Tensor Cores**: Suporta aceleração FP16 (série RTX)
- **Inferência em lote**: Melhor utilização da GPU
- **Gerenciamento de memória**: Controle fino do uso de VRAM

---

## Backend OpenCL

### Cenários de Uso

- GPU AMD
- GPU integrada Intel
- GPU NVIDIA (sem ambiente CUDA)
- Implantação multiplataforma

### Requisitos de Instalação

```bash
# Linux - Instalar kit de desenvolvimento OpenCL
sudo apt install ocl-icd-opencl-dev

# Verificar dispositivos OpenCL disponíveis
clinfo
```

### Compilação

```bash
cmake .. -DUSE_BACKEND=OPENCL
make -j$(nproc)
```

### Escolha de Driver

| Tipo de GPU | Driver Recomendado |
|-------------|-------------------|
| AMD | ROCm ou AMDGPU-PRO |
| Intel | intel-opencl-icd |
| NVIDIA | nvidia-opencl-icd |

### Ajuste de Desempenho

```ini
# config.cfg
openclDeviceToUse = 0          # Número da GPU
openclUseFP16 = auto           # Meia precisão (quando suportado)
openclUseFP16Storage = true    # Armazenamento FP16
```

---

## Backend Metal

### Cenários de Uso

- Apple Silicon (M1/M2/M3)
- Sistema macOS

### Compilação

```bash
cmake .. -DUSE_BACKEND=METAL
make -j$(sysctl -n hw.ncpu)
```

### Otimização para Apple Silicon

A arquitetura de memória unificada do Apple Silicon tem vantagens especiais:

```ini
# Configuração recomendada para Apple Silicon
numNNServerThreadsPerModel = 1
nnMaxBatchSize = 16
numSearchThreads = 4
```

### Comparação de Desempenho

| Chip | Desempenho Relativo |
|------|---------------------|
| M1 | ~RTX 2060 |
| M1 Pro/Max | ~RTX 3060 |
| M2 | ~RTX 2070 |
| M3 Pro/Max | ~RTX 3070 |

---

## Backend Eigen (CPU Pura)

### Cenários de Uso

- Sem ambiente GPU
- Testes rápidos
- Uso de baixa intensidade

### Compilação

```bash
sudo apt install libeigen3-dev
cmake .. -DUSE_BACKEND=EIGEN
make -j$(nproc)
```

### Expectativa de Desempenho

```
CPU single-core: ~10-30 playouts/seg
CPU multi-core: ~50-150 playouts/seg
GPU (gama média): ~1000-3000 playouts/seg
```

---

## Parâmetros de Ajuste de Desempenho

### Parâmetros Principais

```ini
# config.cfg

# === Configuração de Rede Neural ===
# Número da GPU (para múltiplas GPUs)
nnDeviceIdxs = 0

# Threads de inferência por modelo
numNNServerThreadsPerModel = 2

# Tamanho máximo do lote
nnMaxBatchSize = 16

# Tamanho do cache (2^N posições)
nnCacheSizePowerOfTwo = 20

# === Configuração de Busca ===
# Threads de busca
numSearchThreads = 8

# Máximo de visitas por jogada
maxVisits = 800
```

### Guia de Ajuste de Parâmetros

#### nnMaxBatchSize

```
Muito pequeno: Baixa utilização da GPU, alta latência de inferência
Muito grande: VRAM insuficiente, tempo de espera longo

Valores recomendados:
- 4GB VRAM: 8-12
- 8GB VRAM: 16-24
- 16GB+ VRAM: 32-64
```

#### numSearchThreads

```
Muito poucos: Não consegue alimentar a GPU
Muitos: Gargalo de CPU, pressão de memória

Valores recomendados:
- 1-2x o número de núcleos de CPU
- Similar ao nnMaxBatchSize
```

#### numNNServerThreadsPerModel

```
CUDA: 1-2
OpenCL: 1-2
Eigen: Número de núcleos de CPU
```

### Ajuste de Memória

```ini
# Reduzir uso de VRAM
nnMaxBatchSize = 8
nnCacheSizePowerOfTwo = 18

# Aumentar uso de VRAM (melhorar desempenho)
nnMaxBatchSize = 32
nnCacheSizePowerOfTwo = 22
```

---

## Configuração Multi-GPU

### Múltiplas GPUs em uma Máquina

```ini
# Usar GPU 0 e GPU 1
nnDeviceIdxs = 0,1

# Threads por GPU
numNNServerThreadsPerModel = 2
```

### Balanceamento de Carga

```ini
# Distribuir peso conforme desempenho da GPU
# GPU 0 é mais potente, recebe mais trabalho
nnDeviceIdxs = 0,0,1
```

---

## Benchmark

### Executar Benchmark

```bash
katago benchmark -model model.bin.gz -config config.cfg
```

### Interpretação da Saída

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

### Dados de Desempenho Comuns

| GPU | Modelo | Playouts/seg |
|-----|--------|--------------|
| RTX 3060 | b18c384 | ~2500 |
| RTX 3080 | b18c384 | ~4500 |
| RTX 4090 | b18c384 | ~8000 |
| M1 Pro | b18c384 | ~1500 |
| M2 Max | b18c384 | ~2200 |

---

## Aceleração TensorRT

### Cenários de Uso

- GPU NVIDIA
- Busca por desempenho máximo
- Aceita tempo de inicialização mais longo

### Como Habilitar

```bash
# Habilitar na compilação
cmake .. -DUSE_BACKEND=CUDA -DUSE_TENSORRT=ON

# Ou usar versão pré-compilada
katago-tensorrt
```

### Ganho de Desempenho

```
CUDA padrão: 100%
TensorRT FP32: +20-30%
TensorRT FP16: +50-80% (série RTX)
TensorRT INT8: +100-150% (requer calibração)
```

### Observações

- Primeira inicialização requer compilar engine TensorRT (alguns minutos)
- GPUs diferentes requerem recompilação
- FP16/INT8 podem reduzir levemente a precisão

---

## Problemas Comuns

### GPU Não Detectada

```bash
# Verificar status da GPU
nvidia-smi  # NVIDIA
rocm-smi    # AMD
clinfo      # OpenCL

# KataGo lista GPUs disponíveis
katago gpuinfo
```

### VRAM Insuficiente

```ini
# Usar modelo menor
# b18c384 → b10c128

# Reduzir tamanho do lote
nnMaxBatchSize = 4

# Reduzir cache
nnCacheSizePowerOfTwo = 16
```

### Desempenho Abaixo do Esperado

1. Confirmar uso do backend correto (CUDA > OpenCL > Eigen)
2. Verificar se `numSearchThreads` é suficiente
3. Confirmar que a GPU não está ocupada por outros programas
4. Usar comando `benchmark` para confirmar desempenho

---

## Checklist de Otimização de Desempenho

- [ ] Escolher o backend correto (CUDA/OpenCL/Metal)
- [ ] Instalar drivers de GPU mais recentes
- [ ] Ajustar `nnMaxBatchSize` conforme VRAM
- [ ] Ajustar `numSearchThreads` conforme CPU
- [ ] Executar `benchmark` para confirmar desempenho
- [ ] Monitorar utilização da GPU (deve ser > 80%)

---

## Leitura Adicional

- [Detalhes de Implementação do MCTS](../mcts-implementation) — Origem da demanda por inferência em lote
- [Quantização e Implantação de Modelos](../quantization-deploy) — Otimizações avançadas de desempenho
- [Guia Completo de Instalação](../../hands-on/setup) — Passos de instalação para cada plataforma
