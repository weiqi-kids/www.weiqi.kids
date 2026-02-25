---
sidebar_position: 6
title: GPU 백엔드와 최적화
description: KataGo의 CUDA, OpenCL, Metal 백엔드 비교 및 성능 튜닝 가이드
---

# GPU 백엔드와 최적화

이 문서는 KataGo가 지원하는 다양한 GPU 백엔드, 성능 차이 및 최적 성능을 위한 튜닝 방법을 소개합니다.

---

## 백엔드 개요

KataGo는 네 가지 계산 백엔드를 지원합니다:

| 백엔드 | 하드웨어 지원 | 성능 | 설치 난이도 |
|------|---------|------|---------|
| **CUDA** | NVIDIA GPU | 최상 | 중간 |
| **OpenCL** | NVIDIA/AMD/Intel GPU | 양호 | 쉬움 |
| **Metal** | Apple Silicon | 양호 | 쉬움 |
| **Eigen** | 순수 CPU | 느림 | 가장 쉬움 |

---

## CUDA 백엔드

### 적용 시나리오

- NVIDIA GPU (GTX 10 시리즈 이상)
- 최고 성능 필요
- CUDA 개발 환경 보유

### 설치 요구사항

```bash
# CUDA 버전 확인
nvcc --version

# cuDNN 확인
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR
```

| 구성 요소 | 권장 버전 |
|------|---------|
| CUDA | 11.x 또는 12.x |
| cuDNN | 8.x |
| 드라이버 | 470+ |

### 컴파일

```bash
cd KataGo/cpp
mkdir build && cd build

cmake .. -DUSE_BACKEND=CUDA \
         -DCUDNN_INCLUDE_DIR=/usr/local/cuda/include \
         -DCUDNN_LIBRARY=/usr/local/cuda/lib64/libcudnn.so

make -j$(nproc)
```

### 성능 특징

- **Tensor Cores**: FP16 가속 지원 (RTX 시리즈)
- **배치 추론**: GPU 활용률 최상
- **메모리 관리**: VRAM 사용량 세밀하게 제어 가능

---

## OpenCL 백엔드

### 적용 시나리오

- AMD GPU
- Intel 내장 그래픽
- NVIDIA GPU (CUDA 환경 없이)
- 크로스 플랫폼 배포

### 설치 요구사항

```bash
# Linux - OpenCL 개발 패키지 설치
sudo apt install ocl-icd-opencl-dev

# 사용 가능한 OpenCL 장치 확인
clinfo
```

### 컴파일

```bash
cmake .. -DUSE_BACKEND=OPENCL
make -j$(nproc)
```

### 드라이버 선택

| GPU 유형 | 권장 드라이버 |
|---------|---------|
| AMD | ROCm 또는 AMDGPU-PRO |
| Intel | intel-opencl-icd |
| NVIDIA | nvidia-opencl-icd |

### 성능 튜닝

```ini
# config.cfg
openclDeviceToUse = 0          # GPU 번호
openclUseFP16 = auto           # 반정밀도 (지원 시)
openclUseFP16Storage = true    # FP16 저장
```

---

## Metal 백엔드

### 적용 시나리오

- Apple Silicon (M1/M2/M3)
- macOS 시스템

### 컴파일

```bash
cmake .. -DUSE_BACKEND=METAL
make -j$(sysctl -n hw.ncpu)
```

### Apple Silicon 최적화

Apple Silicon의 통합 메모리 아키텍처는 특별한 장점이 있습니다:

```ini
# Apple Silicon 권장 설정
numNNServerThreadsPerModel = 1
nnMaxBatchSize = 16
numSearchThreads = 4
```

### 성능 비교

| 칩 | 상대 성능 |
|------|---------|
| M1 | ~RTX 2060 |
| M1 Pro/Max | ~RTX 3060 |
| M2 | ~RTX 2070 |
| M3 Pro/Max | ~RTX 3070 |

---

## Eigen 백엔드 (순수 CPU)

### 적용 시나리오

- GPU 환경 없음
- 빠른 테스트
- 낮은 강도 사용

### 컴파일

```bash
sudo apt install libeigen3-dev
cmake .. -DUSE_BACKEND=EIGEN
make -j$(nproc)
```

### 예상 성능

```
CPU 싱글 코어: ~10-30 playouts/sec
CPU 멀티 코어: ~50-150 playouts/sec
GPU (중급): ~1000-3000 playouts/sec
```

---

## 성능 튜닝 파라미터

### 핵심 파라미터

```ini
# config.cfg

# === 신경망 설정 ===
# GPU 번호 (다중 GPU 시 사용)
nnDeviceIdxs = 0

# 모델당 추론 스레드 수
numNNServerThreadsPerModel = 2

# 최대 배치 크기
nnMaxBatchSize = 16

# 캐시 크기 (2^N 위치)
nnCacheSizePowerOfTwo = 20

# === 탐색 설정 ===
# 탐색 스레드 수
numSearchThreads = 8

# 수당 최대 방문 횟수
maxVisits = 800
```

### 파라미터 튜닝 가이드

#### nnMaxBatchSize

```
너무 작으면: GPU 활용률 낮음, 추론 지연 높음
너무 크면: VRAM 부족, 대기 시간 김

권장값:
- 4GB VRAM: 8-12
- 8GB VRAM: 16-24
- 16GB+ VRAM: 32-64
```

#### numSearchThreads

```
너무 적으면: GPU를 충분히 활용 못함
너무 많으면: CPU 병목, 메모리 압박

권장값:
- CPU 코어 수의 1-2배
- nnMaxBatchSize와 비슷하게
```

#### numNNServerThreadsPerModel

```
CUDA: 1-2
OpenCL: 1-2
Eigen: CPU 코어 수
```

### 메모리 튜닝

```ini
# VRAM 사용량 줄이기
nnMaxBatchSize = 8
nnCacheSizePowerOfTwo = 18

# VRAM 사용량 늘리기 (성능 향상)
nnMaxBatchSize = 32
nnCacheSizePowerOfTwo = 22
```

---

## 다중 GPU 설정

### 단일 머신 다중 GPU

```ini
# GPU 0과 GPU 1 사용
nnDeviceIdxs = 0,1

# GPU당 스레드 수
numNNServerThreadsPerModel = 2
```

### 로드 밸런싱

```ini
# GPU 성능에 따라 가중치 배분
# GPU 0이 더 강력하면 더 많은 작업 배분
nnDeviceIdxs = 0,0,1
```

---

## 벤치마크 테스트

### 벤치마크 실행

```bash
katago benchmark -model model.bin.gz -config config.cfg
```

### 출력 해석

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

### 일반적인 성능 데이터

| GPU | 모델 | Playouts/초 |
|-----|------|------------|
| RTX 3060 | b18c384 | ~2500 |
| RTX 3080 | b18c384 | ~4500 |
| RTX 4090 | b18c384 | ~8000 |
| M1 Pro | b18c384 | ~1500 |
| M2 Max | b18c384 | ~2200 |

---

## TensorRT 가속

### 적용 시나리오

- NVIDIA GPU
- 극한 성능 추구
- 긴 초기화 시간 수용 가능

### 활성화 방법

```bash
# 컴파일 시 활성화
cmake .. -DUSE_BACKEND=CUDA -DUSE_TENSORRT=ON

# 또는 사전 컴파일 버전 사용
katago-tensorrt
```

### 성능 향상

```
표준 CUDA: 100%
TensorRT FP32: +20-30%
TensorRT FP16: +50-80% (RTX 시리즈)
TensorRT INT8: +100-150% (교정 필요)
```

### 주의사항

- 첫 시작 시 TensorRT 엔진 컴파일 필요 (수 분)
- 다른 GPU는 재컴파일 필요
- FP16/INT8은 정밀도가 약간 떨어질 수 있음

---

## 일반적인 문제

### GPU가 감지되지 않음

```bash
# GPU 상태 확인
nvidia-smi  # NVIDIA
rocm-smi    # AMD
clinfo      # OpenCL

# KataGo에서 사용 가능한 GPU 나열
katago gpuinfo
```

### VRAM 부족

```ini
# 더 작은 모델 사용
# b18c384 → b10c128

# 배치 크기 줄이기
nnMaxBatchSize = 4

# 캐시 줄이기
nnCacheSizePowerOfTwo = 16
```

### 예상보다 낮은 성능

1. 올바른 백엔드 사용 확인 (CUDA > OpenCL > Eigen)
2. `numSearchThreads`가 충분한지 확인
3. GPU가 다른 프로그램에 점유되어 있지 않은지 확인
4. `benchmark` 명령으로 성능 확인

---

## 성능 최적화 체크리스트

- [ ] 올바른 백엔드 선택 (CUDA/OpenCL/Metal)
- [ ] 최신 GPU 드라이버 설치
- [ ] VRAM에 맞게 `nnMaxBatchSize` 조정
- [ ] CPU에 맞게 `numSearchThreads` 조정
- [ ] `benchmark` 실행하여 성능 확인
- [ ] GPU 사용률 모니터링 (80% 이상이어야 함)

---

## 추가 읽기

- [MCTS 구현 세부사항](../mcts-implementation) — 배치 추론의 필요성
- [모델 양자화와 배포](../quantization-deploy) — 추가 성능 최적화
- [설치 가이드](../../hands-on/setup) — 각 플랫폼 설치 단계
