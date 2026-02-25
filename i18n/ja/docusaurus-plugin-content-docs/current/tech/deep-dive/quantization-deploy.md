---
sidebar_position: 8
title: モデル量子化とデプロイメント
description: KataGoモデルの量子化技術、エクスポートフォーマット、各プラットフォームへのデプロイソリューション
---

# モデル量子化とデプロイメント

本記事では、KataGoモデルの量子化によるリソース削減方法と、各種プラットフォームへのデプロイソリューションを紹介します。

---

## 量子化技術の概要

### なぜ量子化が必要か？

| 精度 | サイズ | 速度 | 精度損失 |
|------|------|------|---------|
| FP32 | 100% | 基準 | 0% |
| FP16 | 50% | +50% | ~0% |
| INT8 | 25% | +100% | \<1% |

### 量子化の種類

```
訓練後量子化（PTQ）
├── 簡単で高速
├── 再訓練不要
└── 精度損失の可能性あり

量子化認識訓練（QAT）
├── より高い精度
├── 再訓練が必要
└── より複雑
```

---

## FP16半精度

### 概念

32ビット浮動小数点を16ビットに変換：

```python
# FP32 → FP16変換
model_fp16 = model.half()

# 推論
with torch.cuda.amp.autocast():
    output = model_fp16(input.half())
```

### KataGo設定

```ini
# config.cfg
useFP16 = true           # FP16推論を有効化
useFP16Storage = true    # FP16で中間結果を保存
```

### パフォーマンスへの影響

| GPUシリーズ | FP16アクセラレーション |
|---------|----------|
| GTX 10xx | なし（Tensor Coreなし） |
| RTX 20xx | +30-50% |
| RTX 30xx | +50-80% |
| RTX 40xx | +80-100% |

---

## INT8量子化

### 量子化フロー

```python
import torch.quantization as quant

# 1. モデルを準備
model.eval()
model.qconfig = quant.get_default_qconfig('fbgemm')

# 2. 量子化を準備
model_prepared = quant.prepare(model)

# 3. キャリブレーション（代表的なデータを使用）
with torch.no_grad():
    for data in calibration_loader:
        model_prepared(data)

# 4. 量子化モデルに変換
model_quantized = quant.convert(model_prepared)
```

### キャリブレーションデータ

```python
def create_calibration_dataset(num_samples=1000):
    """キャリブレーションデータセットを作成"""
    samples = []

    # 実際の対局からサンプリング
    for game in random_games(num_samples):
        position = random_position(game)
        features = encode_state(position)
        samples.append(features)

    return samples
```

### 注意事項

- INT8量子化にはキャリブレーションデータが必要
- 一部のレイヤーは量子化に適さない場合がある
- 精度損失のテストが必要

---

## TensorRTデプロイメント

### 変換フロー

```python
import tensorrt as trt

def convert_to_tensorrt(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # ONNXモデルを解析
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())

    # 最適化オプションを設定
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB

    # FP16を有効化
    config.set_flag(trt.BuilderFlag.FP16)

    # エンジンを構築
    engine = builder.build_engine(network, config)

    # 保存
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
```

### TensorRTエンジンの使用

```python
def inference_with_tensorrt(engine_path, input_data):
    # エンジンをロード
    with open(engine_path, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # メモリを割り当て
    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(output_size)

    # 入力をコピー
    cuda.memcpy_htod(d_input, input_data)

    # 推論を実行
    context.execute_v2([int(d_input), int(d_output)])

    # 出力を取得
    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output, d_output)

    return output
```

---

## ONNXエクスポート

### PyTorch → ONNX

```python
import torch.onnx

def export_to_onnx(model, output_path):
    model.eval()

    # サンプル入力を作成
    dummy_input = torch.randn(1, 22, 19, 19)

    # エクスポート
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

### ONNXモデルの検証

```python
import onnx
import onnxruntime as ort

# モデル構造を検証
model = onnx.load("model.onnx")
onnx.checker.check_model(model)

# 推論をテスト
session = ort.InferenceSession("model.onnx")
output = session.run(None, {'input': input_data})
```

---

## 各プラットフォームへのデプロイ

### サーバーデプロイ

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

### デスクトップアプリ統合

```python
# KataGoをPythonアプリに埋め込む
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

### モバイルデプロイ

#### iOS（Core ML）

```python
import coremltools as ct

# Core MLに変換
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

# TFLiteに変換
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open('katago.tflite', 'wb') as f:
    f.write(tflite_model)
```

### 組み込みシステム

#### Raspberry Pi

```bash
# Eigenバックエンドを使用（CPU専用）
./katago gtp -model kata-b10c128.bin.gz -config rpi.cfg
```

```ini
# rpi.cfg - Raspberry Pi最適化設定
numSearchThreads = 4
maxVisits = 100
nnMaxBatchSize = 1
```

#### NVIDIA Jetson

```bash
# CUDAバックエンドを使用
./katago gtp -model kata-b18c384.bin.gz -config jetson.cfg
```

---

## パフォーマンス比較

### 各デプロイ方式のパフォーマンス

| デプロイ方式 | ハードウェア | Playouts/秒 |
|---------|------|------------|
| CUDA FP32 | RTX 3080 | ~3000 |
| CUDA FP16 | RTX 3080 | ~5000 |
| TensorRT FP16 | RTX 3080 | ~6500 |
| OpenCL | M1 Pro | ~1500 |
| Core ML | M1 Pro | ~1800 |
| TFLite | Pixel 7 | ~50 |
| Eigen | RPi 4 | ~15 |

### モデルサイズ比較

| フォーマット | b18c384サイズ |
|------|-------------|
| オリジナル (.bin.gz) | ~140 MB |
| ONNX FP32 | ~280 MB |
| ONNX FP16 | ~140 MB |
| TensorRT FP16 | ~100 MB |
| TFLite FP16 | ~140 MB |

---

## デプロイチェックリスト

- [ ] 適切な量子化精度を選択
- [ ] キャリブレーションデータを準備（INT8）
- [ ] ターゲットフォーマットにエクスポート
- [ ] 精度損失が許容範囲内か検証
- [ ] ターゲットプラットフォームでパフォーマンスをテスト
- [ ] メモリ使用量を最適化
- [ ] 自動化されたデプロイフローを構築

---

## 関連記事

- [GPUバックエンドと最適化](../gpu-optimization) — 基本的なパフォーマンス最適化
- [評価とベンチマーク](../evaluation) — デプロイ後のパフォーマンス検証
- [プロジェクトへの統合](../../hands-on/integration) — API統合例
