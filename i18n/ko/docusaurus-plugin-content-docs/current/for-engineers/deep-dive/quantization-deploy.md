---
sidebar_position: 8
title: 모델 양자화와 배포
description: KataGo 모델의 양자화 기술, 내보내기 형식 및 각 플랫폼 배포 방안
---

# 모델 양자화와 배포

이 문서는 KataGo 모델을 양자화하여 리소스 요구사항을 줄이는 방법과 다양한 플랫폼에서의 배포 방안을 소개합니다.

---

## 양자화 기술 개요

### 왜 양자화가 필요한가?

| 정밀도 | 크기 | 속도 | 정밀도 손실 |
|------|------|------|---------|
| FP32 | 100% | 기준 | 0% |
| FP16 | 50% | +50% | ~0% |
| INT8 | 25% | +100% | \<1% |

### 양자화 유형

```
훈련 후 양자화 (PTQ)
├── 간단하고 빠름
├── 재학습 불필요
└── 정밀도 손실 가능성

양자화 인식 훈련 (QAT)
├── 더 높은 정밀도
├── 재학습 필요
└── 비교적 복잡
```

---

## FP16 반정밀도

### 개념

32비트 부동소수점을 16비트로 변환:

```python
# FP32 → FP16 변환
model_fp16 = model.half()

# 추론
with torch.cuda.amp.autocast():
    output = model_fp16(input.half())
```

### KataGo 설정

```ini
# config.cfg
useFP16 = true           # FP16 추론 활성화
useFP16Storage = true    # FP16 중간 결과 저장
```

### 성능 영향

| GPU 시리즈 | FP16 가속 |
|---------|----------|
| GTX 10xx | 없음 (Tensor Core 없음) |
| RTX 20xx | +30-50% |
| RTX 30xx | +50-80% |
| RTX 40xx | +80-100% |

---

## INT8 양자화

### 양자화 프로세스

```python
import torch.quantization as quant

# 1. 모델 준비
model.eval()
model.qconfig = quant.get_default_qconfig('fbgemm')

# 2. 양자화 준비
model_prepared = quant.prepare(model)

# 3. 교정 (대표 데이터 사용)
with torch.no_grad():
    for data in calibration_loader:
        model_prepared(data)

# 4. 양자화 모델로 변환
model_quantized = quant.convert(model_prepared)
```

### 교정 데이터

```python
def create_calibration_dataset(num_samples=1000):
    """교정 데이터셋 생성"""
    samples = []

    # 실제 대국에서 샘플링
    for game in random_games(num_samples):
        position = random_position(game)
        features = encode_state(position)
        samples.append(features)

    return samples
```

### 주의사항

- INT8 양자화는 교정 데이터 필요
- 일부 레이어는 양자화에 적합하지 않을 수 있음
- 정밀도 손실 테스트 필요

---

## TensorRT 배포

### 변환 프로세스

```python
import tensorrt as trt

def convert_to_tensorrt(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # ONNX 모델 파싱
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())

    # 최적화 옵션 설정
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB

    # FP16 활성화
    config.set_flag(trt.BuilderFlag.FP16)

    # 엔진 빌드
    engine = builder.build_engine(network, config)

    # 저장
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
```

### TensorRT 엔진 사용

```python
def inference_with_tensorrt(engine_path, input_data):
    # 엔진 로드
    with open(engine_path, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # 메모리 할당
    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(output_size)

    # 입력 복사
    cuda.memcpy_htod(d_input, input_data)

    # 추론 실행
    context.execute_v2([int(d_input), int(d_output)])

    # 출력 가져오기
    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output, d_output)

    return output
```

---

## ONNX 내보내기

### PyTorch → ONNX

```python
import torch.onnx

def export_to_onnx(model, output_path):
    model.eval()

    # 예제 입력 생성
    dummy_input = torch.randn(1, 22, 19, 19)

    # 내보내기
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['policy', 'value', 'ownership'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'policy': {0: 'batch_size'},
            'value': {0: 'batch_size'},
            'ownership': {0: 'batch_size'}
        },
        opset_version=13
    )
```

### ONNX 모델 검증

```python
import onnx
import onnxruntime as ort

# 모델 구조 검증
model = onnx.load("model.onnx")
onnx.checker.check_model(model)

# 추론 테스트
session = ort.InferenceSession("model.onnx")
output = session.run(None, {'input': input_data})
```

---

## 각 플랫폼 배포

### 서버 배포

```yaml
# docker-compose.yml
version: '3'
services:
  katago:
    image: katago/katago:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./models:/models
      - ./config:/config
    command: >
      katago analysis
      -model /models/kata-b18c384.bin.gz
      -config /config/analysis.cfg
```

### 데스크톱 애플리케이션 통합

```python
# KataGo를 Python 애플리케이션에 임베드
import subprocess
import json

class KataGoProcess:
    def __init__(self, katago_path, model_path):
        self.process = subprocess.Popen(
            [katago_path, 'analysis', '-model', model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )

    def analyze(self, moves):
        query = {
            'id': 'query1',
            'moves': moves,
            'rules': 'chinese',
            'komi': 7.5,
            'boardXSize': 19,
            'boardYSize': 19
        }
        self.process.stdin.write(json.dumps(query) + '\n')
        self.process.stdin.flush()

        response = self.process.stdout.readline()
        return json.loads(response)
```

### 모바일 기기 배포

#### iOS (Core ML)

```python
import coremltools as ct

# Core ML로 변환
mlmodel = ct.convert(
    model,
    inputs=[ct.TensorType(shape=(1, 22, 19, 19))],
    minimum_deployment_target=ct.target.iOS15
)

mlmodel.save("KataGo.mlmodel")
```

#### Android (TensorFlow Lite)

```python
import tensorflow as tf

# TFLite로 변환
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open('katago.tflite', 'wb') as f:
    f.write(tflite_model)
```

### 임베디드 시스템

#### Raspberry Pi

```bash
# Eigen 백엔드 사용 (순수 CPU)
./katago gtp -model kata-b10c128.bin.gz -config rpi.cfg
```

```ini
# rpi.cfg - Raspberry Pi 최적화 설정
numSearchThreads = 4
maxVisits = 100
nnMaxBatchSize = 1
```

#### NVIDIA Jetson

```bash
# CUDA 백엔드 사용
./katago gtp -model kata-b18c384.bin.gz -config jetson.cfg
```

---

## 성능 비교

### 다양한 배포 방식의 성능

| 배포 방식 | 하드웨어 | Playouts/초 |
|---------|------|------------|
| CUDA FP32 | RTX 3080 | ~3000 |
| CUDA FP16 | RTX 3080 | ~5000 |
| TensorRT FP16 | RTX 3080 | ~6500 |
| OpenCL | M1 Pro | ~1500 |
| Core ML | M1 Pro | ~1800 |
| TFLite | Pixel 7 | ~50 |
| Eigen | RPi 4 | ~15 |

### 모델 크기 비교

| 형식 | b18c384 크기 |
|------|-------------|
| 원본 (.bin.gz) | ~140 MB |
| ONNX FP32 | ~280 MB |
| ONNX FP16 | ~140 MB |
| TensorRT FP16 | ~100 MB |
| TFLite FP16 | ~140 MB |

---

## 배포 체크리스트

- [ ] 적합한 양자화 정밀도 선택
- [ ] 교정 데이터 준비 (INT8)
- [ ] 대상 형식으로 내보내기
- [ ] 정밀도 손실이 허용 가능한지 검증
- [ ] 대상 플랫폼 성능 테스트
- [ ] 메모리 사용량 최적화
- [ ] 자동화 배포 프로세스 구축

---

## 추가 읽기

- [GPU 백엔드와 최적화](../gpu-optimization) — 기본 성능 최적화
- [평가 및 벤치마크 테스트](../evaluation) — 배포 후 성능 검증
- [프로젝트에 통합하기](../../hands-on/integration) — API 통합 예제
