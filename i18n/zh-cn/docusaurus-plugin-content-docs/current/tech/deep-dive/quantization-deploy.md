---
sidebar_position: 8
title: 模型量化与部署
description: KataGo 模型的量化技术、导出格式与各平台部署方案
---

# 模型量化与部署

本文介绍如何将 KataGo 模型量化以减少资源需求，以及在各种平台上的部署方案。

---

## 量化技术总览

### 为什么需要量化？

| 精度 | 大小 | 速度 | 精度损失 |
|------|------|------|---------|
| FP32 | 100% | 基准 | 0% |
| FP16 | 50% | +50% | ~0% |
| INT8 | 25% | +100% | \<1% |

### 量化类型

```
训练后量化（PTQ）
├── 简单快速
├── 不需要重新训练
└── 可能有精度损失

量化感知训练（QAT）
├── 精度更高
├── 需要重新训练
└── 较为复杂
```

---

## FP16 半精度

### 概念

将 32 位浮点数转换为 16 位：

```python
# FP32 → FP16 转换
model_fp16 = model.half()

# 推理
with torch.cuda.amp.autocast():
    output = model_fp16(input.half())
```

### KataGo 配置

```ini
# config.cfg
useFP16 = true           # 启用 FP16 推理
useFP16Storage = true    # FP16 存储中间结果
```

### 性能影响

| GPU 系列 | FP16 加速 |
|---------|----------|
| GTX 10xx | 无（无 Tensor Core） |
| RTX 20xx | +30-50% |
| RTX 30xx | +50-80% |
| RTX 40xx | +80-100% |

---

## INT8 量化

### 量化流程

```python
import torch.quantization as quant

# 1. 准备模型
model.eval()
model.qconfig = quant.get_default_qconfig('fbgemm')

# 2. 准备量化
model_prepared = quant.prepare(model)

# 3. 校准（使用代表性数据）
with torch.no_grad():
    for data in calibration_loader:
        model_prepared(data)

# 4. 转换为量化模型
model_quantized = quant.convert(model_prepared)
```

### 校准数据

```python
def create_calibration_dataset(num_samples=1000):
    """创建校准数据集"""
    samples = []

    # 从实际对局中取样
    for game in random_games(num_samples):
        position = random_position(game)
        features = encode_state(position)
        samples.append(features)

    return samples
```

### 注意事项

- INT8 量化需要校准数据
- 某些层可能不适合量化
- 需要测试精度损失

---

## TensorRT 部署

### 转换流程

```python
import tensorrt as trt

def convert_to_tensorrt(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # 解析 ONNX 模型
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())

    # 设置优化选项
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB

    # 启用 FP16
    config.set_flag(trt.BuilderFlag.FP16)

    # 构建引擎
    engine = builder.build_engine(network, config)

    # 保存
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
```

### 使用 TensorRT 引擎

```python
def inference_with_tensorrt(engine_path, input_data):
    # 加载引擎
    with open(engine_path, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # 分配显存
    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(output_size)

    # 复制输入
    cuda.memcpy_htod(d_input, input_data)

    # 执行推理
    context.execute_v2([int(d_input), int(d_output)])

    # 获取输出
    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output, d_output)

    return output
```

---

## ONNX 导出

### PyTorch → ONNX

```python
import torch.onnx

def export_to_onnx(model, output_path):
    model.eval()

    # 创建示例输入
    dummy_input = torch.randn(1, 22, 19, 19)

    # 导出
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

### 验证 ONNX 模型

```python
import onnx
import onnxruntime as ort

# 验证模型结构
model = onnx.load("model.onnx")
onnx.checker.check_model(model)

# 测试推理
session = ort.InferenceSession("model.onnx")
output = session.run(None, {'input': input_data})
```

---

## 各平台部署

### 服务器部署

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

### 桌面应用集成

```python
# 嵌入 KataGo 到 Python 应用
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

### 移动设备部署

#### iOS（Core ML）

```python
import coremltools as ct

# 转换为 Core ML
mlmodel = ct.convert(
    model,
    inputs=[ct.TensorType(shape=(1, 22, 19, 19))],
    minimum_deployment_target=ct.target.iOS15
)

mlmodel.save("KataGo.mlmodel")
```

#### Android（TensorFlow Lite）

```python
import tensorflow as tf

# 转换为 TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open('katago.tflite', 'wb') as f:
    f.write(tflite_model)
```

### 嵌入式系统

#### Raspberry Pi

```bash
# 使用 Eigen 后端（纯 CPU）
./katago gtp -model kata-b10c128.bin.gz -config rpi.cfg
```

```ini
# rpi.cfg - Raspberry Pi 优化配置
numSearchThreads = 4
maxVisits = 100
nnMaxBatchSize = 1
```

#### NVIDIA Jetson

```bash
# 使用 CUDA 后端
./katago gtp -model kata-b18c384.bin.gz -config jetson.cfg
```

---

## 性能比较

### 不同部署方式的性能

| 部署方式 | 硬件 | Playouts/秒 |
|---------|------|------------|
| CUDA FP32 | RTX 3080 | ~3000 |
| CUDA FP16 | RTX 3080 | ~5000 |
| TensorRT FP16 | RTX 3080 | ~6500 |
| OpenCL | M1 Pro | ~1500 |
| Core ML | M1 Pro | ~1800 |
| TFLite | Pixel 7 | ~50 |
| Eigen | RPi 4 | ~15 |

### 模型大小比较

| 格式 | b18c384 大小 |
|------|-------------|
| 原始 (.bin.gz) | ~140 MB |
| ONNX FP32 | ~280 MB |
| ONNX FP16 | ~140 MB |
| TensorRT FP16 | ~100 MB |
| TFLite FP16 | ~140 MB |

---

## 部署检查清单

- [ ] 选择适合的量化精度
- [ ] 准备校准数据（INT8）
- [ ] 导出为目标格式
- [ ] 验证精度损失可接受
- [ ] 测试目标平台性能
- [ ] 优化内存用量
- [ ] 建立自动化部署流程

---

## 延伸阅读

- [GPU 后端与优化](../gpu-optimization) — 基础性能优化
- [评估与基准测试](../evaluation) — 验证部署后的性能
- [集成到你的项目](../../hands-on/integration) — API 集成示例
