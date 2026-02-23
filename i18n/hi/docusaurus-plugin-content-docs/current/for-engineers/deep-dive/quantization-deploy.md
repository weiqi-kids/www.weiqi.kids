---
sidebar_position: 8
title: मॉडल क्वांटाइज़ेशन और डिप्लॉयमेंट
description: KataGo मॉडल की क्वांटाइज़ेशन तकनीकें, एक्सपोर्ट प्रारूप और विभिन्न प्लेटफॉर्म डिप्लॉयमेंट समाधान
---

# मॉडल क्वांटाइज़ेशन और डिप्लॉयमेंट

यह लेख KataGo मॉडल को क्वांटाइज़ करके संसाधन आवश्यकताओं को कम करने और विभिन्न प्लेटफॉर्मों पर डिप्लॉयमेंट समाधान का परिचय देता है।

---

## क्वांटाइज़ेशन तकनीक अवलोकन

### क्वांटाइज़ेशन क्यों आवश्यक है?

| प्रिसिजन | आकार | गति | सटीकता हानि |
|------|------|------|---------|
| FP32 | 100% | आधार | 0% |
| FP16 | 50% | +50% | ~0% |
| INT8 | 25% | +100% | \<1% |

### क्वांटाइज़ेशन प्रकार

```
पोस्ट-ट्रेनिंग क्वांटाइज़ेशन (PTQ)
├── सरल और तेज
├── रीट्रेनिंग आवश्यक नहीं
└── सटीकता हानि संभव

क्वांटाइज़ेशन-अवेयर ट्रेनिंग (QAT)
├── उच्च सटीकता
├── रीट्रेनिंग आवश्यक
└── अपेक्षाकृत जटिल
```

---

## FP16 हाफ प्रिसिजन

### अवधारणा

32-बिट फ्लोटिंग पॉइंट को 16-बिट में कन्वर्ट करें:

```python
# FP32 → FP16 कन्वर्जन
model_fp16 = model.half()

# इन्फरेंस
with torch.cuda.amp.autocast():
    output = model_fp16(input.half())
```

### KataGo सेटिंग

```ini
# config.cfg
useFP16 = true           # FP16 इन्फरेंस सक्षम
useFP16Storage = true    # FP16 इंटरमीडिएट परिणाम स्टोरेज
```

### प्रदर्शन प्रभाव

| GPU सीरीज़ | FP16 त्वरण |
|---------|----------|
| GTX 10xx | नहीं (Tensor Core नहीं) |
| RTX 20xx | +30-50% |
| RTX 30xx | +50-80% |
| RTX 40xx | +80-100% |

---

## INT8 क्वांटाइज़ेशन

### क्वांटाइज़ेशन प्रक्रिया

```python
import torch.quantization as quant

# 1. मॉडल तैयार करें
model.eval()
model.qconfig = quant.get_default_qconfig('fbgemm')

# 2. क्वांटाइज़ेशन तैयारी
model_prepared = quant.prepare(model)

# 3. कैलिब्रेशन (प्रतिनिधि डेटा के साथ)
with torch.no_grad():
    for data in calibration_loader:
        model_prepared(data)

# 4. क्वांटाइज़्ड मॉडल में कन्वर्ट करें
model_quantized = quant.convert(model_prepared)
```

### कैलिब्रेशन डेटा

```python
def create_calibration_dataset(num_samples=1000):
    """कैलिब्रेशन डेटासेट बनाएं"""
    samples = []

    # वास्तविक गेम से सैंपल लें
    for game in random_games(num_samples):
        position = random_position(game)
        features = encode_state(position)
        samples.append(features)

    return samples
```

### नोट्स

- INT8 क्वांटाइज़ेशन को कैलिब्रेशन डेटा चाहिए
- कुछ लेयर क्वांटाइज़ेशन के लिए उपयुक्त नहीं हो सकती
- सटीकता हानि का परीक्षण आवश्यक

---

## TensorRT डिप्लॉयमेंट

### कन्वर्जन प्रक्रिया

```python
import tensorrt as trt

def convert_to_tensorrt(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # ONNX मॉडल पार्स करें
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())

    # ऑप्टिमाइज़ेशन विकल्प सेट करें
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB

    # FP16 सक्षम करें
    config.set_flag(trt.BuilderFlag.FP16)

    # इंजन बिल्ड करें
    engine = builder.build_engine(network, config)

    # सेव करें
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
```

### TensorRT इंजन उपयोग

```python
def inference_with_tensorrt(engine_path, input_data):
    # इंजन लोड करें
    with open(engine_path, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # मेमोरी आवंटित करें
    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(output_size)

    # इनपुट कॉपी करें
    cuda.memcpy_htod(d_input, input_data)

    # इन्फरेंस निष्पादित करें
    context.execute_v2([int(d_input), int(d_output)])

    # आउटपुट प्राप्त करें
    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output, d_output)

    return output
```

---

## ONNX एक्सपोर्ट

### PyTorch → ONNX

```python
import torch.onnx

def export_to_onnx(model, output_path):
    model.eval()

    # सैंपल इनपुट बनाएं
    dummy_input = torch.randn(1, 22, 19, 19)

    # एक्सपोर्ट करें
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

### ONNX मॉडल सत्यापन

```python
import onnx
import onnxruntime as ort

# मॉडल संरचना सत्यापित करें
model = onnx.load("model.onnx")
onnx.checker.check_model(model)

# इन्फरेंस परीक्षण
session = ort.InferenceSession("model.onnx")
output = session.run(None, {'input': input_data})
```

---

## विभिन्न प्लेटफॉर्म डिप्लॉयमेंट

### सर्वर डिप्लॉयमेंट

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

### डेस्कटॉप एप्लिकेशन इंटीग्रेशन

```python
# KataGo को Python एप्लिकेशन में एम्बेड करें
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

### मोबाइल डिवाइस डिप्लॉयमेंट

#### iOS (Core ML)

```python
import coremltools as ct

# Core ML में कन्वर्ट करें
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

# TFLite में कन्वर्ट करें
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open('katago.tflite', 'wb') as f:
    f.write(tflite_model)
```

### एम्बेडेड सिस्टम

#### Raspberry Pi

```bash
# Eigen बैकएंड उपयोग करें (शुद्ध CPU)
./katago gtp -model kata-b10c128.bin.gz -config rpi.cfg
```

```ini
# rpi.cfg - Raspberry Pi ऑप्टिमाइज़्ड सेटिंग
numSearchThreads = 4
maxVisits = 100
nnMaxBatchSize = 1
```

#### NVIDIA Jetson

```bash
# CUDA बैकएंड उपयोग करें
./katago gtp -model kata-b18c384.bin.gz -config jetson.cfg
```

---

## प्रदर्शन तुलना

### विभिन्न डिप्लॉयमेंट विधियों का प्रदर्शन

| डिप्लॉयमेंट विधि | हार्डवेयर | Playouts/सेकंड |
|---------|------|------------|
| CUDA FP32 | RTX 3080 | ~3000 |
| CUDA FP16 | RTX 3080 | ~5000 |
| TensorRT FP16 | RTX 3080 | ~6500 |
| OpenCL | M1 Pro | ~1500 |
| Core ML | M1 Pro | ~1800 |
| TFLite | Pixel 7 | ~50 |
| Eigen | RPi 4 | ~15 |

### मॉडल आकार तुलना

| प्रारूप | b18c384 आकार |
|------|-------------|
| मूल (.bin.gz) | ~140 MB |
| ONNX FP32 | ~280 MB |
| ONNX FP16 | ~140 MB |
| TensorRT FP16 | ~100 MB |
| TFLite FP16 | ~140 MB |

---

## डिप्लॉयमेंट चेकलिस्ट

- [ ] उपयुक्त क्वांटाइज़ेशन प्रिसिजन चुनें
- [ ] कैलिब्रेशन डेटा तैयार करें (INT8)
- [ ] लक्ष्य प्रारूप में एक्सपोर्ट करें
- [ ] सत्यापित करें सटीकता हानि स्वीकार्य है
- [ ] लक्ष्य प्लेटफॉर्म प्रदर्शन परीक्षण
- [ ] मेमोरी उपयोग ऑप्टिमाइज़ करें
- [ ] ऑटोमेटेड डिप्लॉयमेंट प्रक्रिया स्थापित करें

---

## आगे पढ़ें

- [GPU बैकएंड और अनुकूलन](../gpu-optimization) — मूल प्रदर्शन अनुकूलन
- [मूल्यांकन और बेंचमार्क टेस्टिंग](../evaluation) — डिप्लॉयमेंट के बाद प्रदर्शन सत्यापन
- [अपने प्रोजेक्ट में इंटीग्रेट करें](../../hands-on/integration) — API इंटीग्रेशन उदाहरण
