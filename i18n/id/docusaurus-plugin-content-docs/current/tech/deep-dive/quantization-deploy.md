---
sidebar_position: 8
title: Kuantisasi Model dan Deployment
description: Teknik kuantisasi model KataGo, format ekspor, dan solusi deployment multi-platform
---

# Kuantisasi Model dan Deployment

Artikel ini memperkenalkan cara mengkuantisasi model KataGo untuk mengurangi kebutuhan sumber daya, serta solusi deployment di berbagai platform.

---

## Gambaran Teknik Kuantisasi

### Mengapa Perlu Kuantisasi?

| Presisi | Ukuran | Kecepatan | Kehilangan Presisi |
|---------|--------|-----------|-------------------|
| FP32 | 100% | Baseline | 0% |
| FP16 | 50% | +50% | ~0% |
| INT8 | 25% | +100% | \<1% |

### Tipe Kuantisasi

```
Post-Training Quantization (PTQ)
├── Sederhana dan cepat
├── Tidak perlu pelatihan ulang
└── Mungkin ada kehilangan presisi

Quantization-Aware Training (QAT)
├── Presisi lebih tinggi
├── Perlu pelatihan ulang
└── Lebih kompleks
```

---

## FP16 Half Precision

### Konsep

Mengkonversi floating point 32-bit ke 16-bit:

```python
# Konversi FP32 → FP16
model_fp16 = model.half()

# Inferensi
with torch.cuda.amp.autocast():
    output = model_fp16(input.half())
```

### Konfigurasi KataGo

```ini
# config.cfg
useFP16 = true           # Aktifkan inferensi FP16
useFP16Storage = true    # Penyimpanan hasil antara FP16
```

### Dampak Performa

| Seri GPU | Akselerasi FP16 |
|----------|-----------------|
| GTX 10xx | Tidak ada (tanpa Tensor Core) |
| RTX 20xx | +30-50% |
| RTX 30xx | +50-80% |
| RTX 40xx | +80-100% |

---

## Kuantisasi INT8

### Alur Kuantisasi

```python
import torch.quantization as quant

# 1. Siapkan model
model.eval()
model.qconfig = quant.get_default_qconfig('fbgemm')

# 2. Persiapan kuantisasi
model_prepared = quant.prepare(model)

# 3. Kalibrasi (menggunakan data representatif)
with torch.no_grad():
    for data in calibration_loader:
        model_prepared(data)

# 4. Konversi ke model terkuantisasi
model_quantized = quant.convert(model_prepared)
```

### Data Kalibrasi

```python
def create_calibration_dataset(num_samples=1000):
    """Buat dataset kalibrasi"""
    samples = []

    # Sampel dari pertandingan aktual
    for game in random_games(num_samples):
        position = random_position(game)
        features = encode_state(position)
        samples.append(features)

    return samples
```

### Catatan

- Kuantisasi INT8 membutuhkan data kalibrasi
- Beberapa layer mungkin tidak cocok untuk kuantisasi
- Perlu menguji kehilangan presisi

---

## Deployment TensorRT

### Alur Konversi

```python
import tensorrt as trt

def convert_to_tensorrt(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # Parse model ONNX
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())

    # Atur opsi optimasi
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB

    # Aktifkan FP16
    config.set_flag(trt.BuilderFlag.FP16)

    # Bangun engine
    engine = builder.build_engine(network, config)

    # Simpan
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
```

### Menggunakan Engine TensorRT

```python
def inference_with_tensorrt(engine_path, input_data):
    # Muat engine
    with open(engine_path, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Alokasi memori
    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(output_size)

    # Salin input
    cuda.memcpy_htod(d_input, input_data)

    # Eksekusi inferensi
    context.execute_v2([int(d_input), int(d_output)])

    # Ambil output
    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output, d_output)

    return output
```

---

## Ekspor ONNX

### PyTorch → ONNX

```python
import torch.onnx

def export_to_onnx(model, output_path):
    model.eval()

    # Buat contoh input
    dummy_input = torch.randn(1, 22, 19, 19)

    # Ekspor
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

### Validasi Model ONNX

```python
import onnx
import onnxruntime as ort

# Validasi struktur model
model = onnx.load("model.onnx")
onnx.checker.check_model(model)

# Uji inferensi
session = ort.InferenceSession("model.onnx")
output = session.run(None, {'input': input_data})
```

---

## Deployment Multi-Platform

### Deployment Server

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

### Integrasi Aplikasi Desktop

```python
# Embed KataGo ke aplikasi Python
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

### Deployment Perangkat Mobile

#### iOS (Core ML)

```python
import coremltools as ct

# Konversi ke Core ML
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

# Konversi ke TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open('katago.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Sistem Embedded

#### Raspberry Pi

```bash
# Gunakan backend Eigen (CPU saja)
./katago gtp -model kata-b10c128.bin.gz -config rpi.cfg
```

```ini
# rpi.cfg - Konfigurasi optimasi Raspberry Pi
numSearchThreads = 4
maxVisits = 100
nnMaxBatchSize = 1
```

#### NVIDIA Jetson

```bash
# Gunakan backend CUDA
./katago gtp -model kata-b18c384.bin.gz -config jetson.cfg
```

---

## Perbandingan Performa

### Performa Berbagai Metode Deployment

| Metode Deployment | Hardware | Playouts/detik |
|-------------------|----------|----------------|
| CUDA FP32 | RTX 3080 | ~3000 |
| CUDA FP16 | RTX 3080 | ~5000 |
| TensorRT FP16 | RTX 3080 | ~6500 |
| OpenCL | M1 Pro | ~1500 |
| Core ML | M1 Pro | ~1800 |
| TFLite | Pixel 7 | ~50 |
| Eigen | RPi 4 | ~15 |

### Perbandingan Ukuran Model

| Format | Ukuran b18c384 |
|--------|----------------|
| Asli (.bin.gz) | ~140 MB |
| ONNX FP32 | ~280 MB |
| ONNX FP16 | ~140 MB |
| TensorRT FP16 | ~100 MB |
| TFLite FP16 | ~140 MB |

---

## Checklist Deployment

- [ ] Pilih presisi kuantisasi yang sesuai
- [ ] Siapkan data kalibrasi (INT8)
- [ ] Ekspor ke format target
- [ ] Verifikasi kehilangan presisi dapat diterima
- [ ] Uji performa platform target
- [ ] Optimasi penggunaan memori
- [ ] Buat alur deployment otomatis

---

## Bacaan Lanjutan

- [Backend GPU dan Optimasi](../gpu-optimization) — Optimasi performa dasar
- [Evaluasi dan Benchmark](../evaluation) — Verifikasi performa setelah deployment
- [Integrasi ke Proyek Anda](../../hands-on/integration) — Contoh integrasi API
