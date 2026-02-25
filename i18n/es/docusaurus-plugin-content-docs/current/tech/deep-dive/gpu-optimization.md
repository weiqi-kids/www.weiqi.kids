---
sidebar_position: 6
title: Backend GPU y optimización
description: Comparación de backends CUDA, OpenCL, Metal de KataGo y guía de ajuste de rendimiento
---

# Backend GPU y optimización

Este artículo presenta los diferentes backends GPU que soporta KataGo, las diferencias de rendimiento y cómo ajustar para obtener el mejor rendimiento.

---

## Visión general de backends

KataGo soporta cuatro backends de cálculo:

| Backend | Soporte de hardware | Rendimiento | Dificultad de instalación |
|---------|---------------------|-------------|---------------------------|
| **CUDA** | GPU NVIDIA | Óptimo | Media |
| **OpenCL** | GPU NVIDIA/AMD/Intel | Bueno | Fácil |
| **Metal** | Apple Silicon | Bueno | Fácil |
| **Eigen** | Solo CPU | Más lento | Más fácil |

---

## Backend CUDA

### Escenarios de uso

- GPU NVIDIA (serie GTX 10 o superior)
- Necesidad de máximo rendimiento
- Entorno de desarrollo CUDA disponible

### Requisitos de instalación

```bash
# Verificar versión de CUDA
nvcc --version

# Verificar cuDNN
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR
```

| Componente | Versión recomendada |
|------------|---------------------|
| CUDA | 11.x o 12.x |
| cuDNN | 8.x |
| Driver | 470+ |

### Compilación

```bash
cd KataGo/cpp
mkdir build && cd build

cmake .. -DUSE_BACKEND=CUDA \
         -DCUDNN_INCLUDE_DIR=/usr/local/cuda/include \
         -DCUDNN_LIBRARY=/usr/local/cuda/lib64/libcudnn.so

make -j$(nproc)
```

### Características de rendimiento

- **Tensor Cores**: Soporte de aceleración FP16 (serie RTX)
- **Inferencia por lotes**: Mejor utilización de GPU
- **Gestión de memoria**: Control fino del uso de VRAM

---

## Backend OpenCL

### Escenarios de uso

- GPU AMD
- GPU integrada Intel
- GPU NVIDIA (sin entorno CUDA)
- Despliegue multiplataforma

### Requisitos de instalación

```bash
# Linux - Instalar kit de desarrollo OpenCL
sudo apt install ocl-icd-opencl-dev

# Verificar dispositivos OpenCL disponibles
clinfo
```

### Compilación

```bash
cmake .. -DUSE_BACKEND=OPENCL
make -j$(nproc)
```

### Selección de driver

| Tipo de GPU | Driver recomendado |
|-------------|-------------------|
| AMD | ROCm o AMDGPU-PRO |
| Intel | intel-opencl-icd |
| NVIDIA | nvidia-opencl-icd |

### Ajuste de rendimiento

```ini
# config.cfg
openclDeviceToUse = 0          # Número de GPU
openclUseFP16 = auto           # Media precisión (si es compatible)
openclUseFP16Storage = true    # Almacenamiento FP16
```

---

## Backend Metal

### Escenarios de uso

- Apple Silicon (M1/M2/M3)
- Sistema macOS

### Compilación

```bash
cmake .. -DUSE_BACKEND=METAL
make -j$(sysctl -n hw.ncpu)
```

### Optimización para Apple Silicon

La arquitectura de memoria unificada de Apple Silicon tiene ventajas especiales:

```ini
# Configuración recomendada para Apple Silicon
numNNServerThreadsPerModel = 1
nnMaxBatchSize = 16
numSearchThreads = 4
```

### Comparación de rendimiento

| Chip | Rendimiento relativo |
|------|---------------------|
| M1 | ~RTX 2060 |
| M1 Pro/Max | ~RTX 3060 |
| M2 | ~RTX 2070 |
| M3 Pro/Max | ~RTX 3070 |

---

## Backend Eigen (solo CPU)

### Escenarios de uso

- Sin entorno GPU
- Pruebas rápidas
- Uso de baja intensidad

### Compilación

```bash
sudo apt install libeigen3-dev
cmake .. -DUSE_BACKEND=EIGEN
make -j$(nproc)
```

### Expectativa de rendimiento

```
CPU un núcleo: ~10-30 playouts/seg
CPU multi-núcleo: ~50-150 playouts/seg
GPU (gama media): ~1000-3000 playouts/seg
```

---

## Parámetros de ajuste de rendimiento

### Parámetros principales

```ini
# config.cfg

# === Configuración de red neuronal ===
# Número de GPU (para multi-GPU)
nnDeviceIdxs = 0

# Hilos de inferencia por modelo
numNNServerThreadsPerModel = 2

# Tamaño máximo de lote
nnMaxBatchSize = 16

# Tamaño de caché (2^N posiciones)
nnCacheSizePowerOfTwo = 20

# === Configuración de búsqueda ===
# Número de hilos de búsqueda
numSearchThreads = 8

# Visitas máximas por movimiento
maxVisits = 800
```

### Guía de ajuste de parámetros

#### nnMaxBatchSize

```
Muy pequeño: Baja utilización de GPU, alta latencia de inferencia
Muy grande: VRAM insuficiente, tiempo de espera largo

Valores recomendados:
- 4GB VRAM: 8-12
- 8GB VRAM: 16-24
- 16GB+ VRAM: 32-64
```

#### numSearchThreads

```
Muy pocos: No puede alimentar la GPU
Muchos: Cuello de botella en CPU, presión de memoria

Valores recomendados:
- 1-2 veces el número de núcleos de CPU
- Similar a nnMaxBatchSize
```

#### numNNServerThreadsPerModel

```
CUDA: 1-2
OpenCL: 1-2
Eigen: Número de núcleos de CPU
```

### Ajuste de memoria

```ini
# Reducir uso de VRAM
nnMaxBatchSize = 8
nnCacheSizePowerOfTwo = 18

# Aumentar uso de VRAM (mejorar rendimiento)
nnMaxBatchSize = 32
nnCacheSizePowerOfTwo = 22
```

---

## Configuración multi-GPU

### Multi-GPU en una máquina

```ini
# Usar GPU 0 y GPU 1
nnDeviceIdxs = 0,1

# Hilos por GPU
numNNServerThreadsPerModel = 2
```

### Balanceo de carga

```ini
# Asignar pesos según rendimiento de GPU
# GPU 0 más potente, asignar más trabajo
nnDeviceIdxs = 0,0,1
```

---

## Benchmarking

### Ejecutar benchmark

```bash
katago benchmark -model model.bin.gz -config config.cfg
```

### Interpretación de salida

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

### Datos de rendimiento comunes

| GPU | Modelo | Playouts/seg |
|-----|--------|--------------|
| RTX 3060 | b18c384 | ~2500 |
| RTX 3080 | b18c384 | ~4500 |
| RTX 4090 | b18c384 | ~8000 |
| M1 Pro | b18c384 | ~1500 |
| M2 Max | b18c384 | ~2200 |

---

## Aceleración TensorRT

### Escenarios de uso

- GPU NVIDIA
- Búsqueda de rendimiento extremo
- Aceptar tiempo de inicialización más largo

### Método de activación

```bash
# Activar al compilar
cmake .. -DUSE_BACKEND=CUDA -DUSE_TENSORRT=ON

# O usar versión precompilada
katago-tensorrt
```

### Mejora de rendimiento

```
CUDA estándar: 100%
TensorRT FP32: +20-30%
TensorRT FP16: +50-80% (serie RTX)
TensorRT INT8: +100-150% (requiere calibración)
```

### Notas

- Primera ejecución necesita compilar el motor TensorRT (varios minutos)
- Diferentes GPUs necesitan recompilar
- FP16/INT8 pueden reducir ligeramente la precisión

---

## Problemas comunes

### GPU no detectada

```bash
# Verificar estado de GPU
nvidia-smi  # NVIDIA
rocm-smi    # AMD
clinfo      # OpenCL

# KataGo listar GPUs disponibles
katago gpuinfo
```

### VRAM insuficiente

```ini
# Usar modelo más pequeño
# b18c384 → b10c128

# Reducir tamaño de lote
nnMaxBatchSize = 4

# Reducir caché
nnCacheSizePowerOfTwo = 16
```

### Rendimiento inferior a lo esperado

1. Confirmar que se usa el backend correcto (CUDA > OpenCL > Eigen)
2. Verificar que `numSearchThreads` sea suficiente
3. Confirmar que la GPU no está ocupada por otros programas
4. Usar comando `benchmark` para confirmar rendimiento

---

## Lista de verificación de optimización de rendimiento

- [ ] Seleccionar el backend correcto (CUDA/OpenCL/Metal)
- [ ] Instalar los drivers de GPU más recientes
- [ ] Ajustar `nnMaxBatchSize` según VRAM
- [ ] Ajustar `numSearchThreads` según CPU
- [ ] Ejecutar `benchmark` para confirmar rendimiento
- [ ] Monitorear utilización de GPU (debería ser > 80%)

---

## Lectura adicional

- [Detalles de implementación de MCTS](../mcts-implementation) — Origen de la necesidad de inferencia por lotes
- [Cuantización y despliegue de modelos](../quantization-deploy) — Optimización de rendimiento adicional
- [Guía de instalación completa](../../hands-on/setup) — Pasos de instalación para cada plataforma
