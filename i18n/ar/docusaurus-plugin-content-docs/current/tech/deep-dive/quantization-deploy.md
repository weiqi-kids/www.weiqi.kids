---
sidebar_position: 8
title: تكميم النموذج والنشر
description: تقنيات تكميم نماذج KataGo، تنسيقات التصدير وحلول النشر لمختلف المنصات
---

# تكميم النموذج والنشر

يقدم هذا المقال كيفية تكميم نماذج KataGo لتقليل متطلبات الموارد، وحلول النشر على مختلف المنصات.

---

## نظرة عامة على تقنيات التكميم

### لماذا نحتاج التكميم؟

| الدقة | الحجم | السرعة | فقدان الدقة |
|-------|-------|--------|-------------|
| FP32 | 100% | أساس | 0% |
| FP16 | 50% | +50% | ~0% |
| INT8 | 25% | +100% | \<1% |

### أنواع التكميم

```
التكميم بعد التدريب (PTQ)
├── بسيط وسريع
├── لا يتطلب إعادة التدريب
└── قد يكون هناك فقدان في الدقة

التدريب الواعي بالتكميم (QAT)
├── دقة أعلى
├── يتطلب إعادة التدريب
└── أكثر تعقيداً
```

---

## FP16 نصف الدقة

### المفهوم

تحويل أرقام الفاصلة العائمة من 32 بت إلى 16 بت:

```python
# تحويل FP32 → FP16
model_fp16 = model.half()

# الاستدلال
with torch.cuda.amp.autocast():
    output = model_fp16(input.half())
```

### إعدادات KataGo

```ini
# config.cfg
useFP16 = true           # تفعيل استدلال FP16
useFP16Storage = true    # تخزين النتائج الوسيطة بـ FP16
```

### تأثير الأداء

| سلسلة GPU | تسريع FP16 |
|-----------|-----------|
| GTX 10xx | لا يوجد (بدون Tensor Core) |
| RTX 20xx | +30-50% |
| RTX 30xx | +50-80% |
| RTX 40xx | +80-100% |

---

## تكميم INT8

### عملية التكميم

```python
import torch.quantization as quant

# 1. إعداد النموذج
model.eval()
model.qconfig = quant.get_default_qconfig('fbgemm')

# 2. إعداد التكميم
model_prepared = quant.prepare(model)

# 3. المعايرة (باستخدام بيانات تمثيلية)
with torch.no_grad():
    for data in calibration_loader:
        model_prepared(data)

# 4. التحويل إلى نموذج مكمم
model_quantized = quant.convert(model_prepared)
```

### بيانات المعايرة

```python
def create_calibration_dataset(num_samples=1000):
    """إنشاء مجموعة بيانات المعايرة"""
    samples = []

    # أخذ عينات من مباريات فعلية
    for game in random_games(num_samples):
        position = random_position(game)
        features = encode_state(position)
        samples.append(features)

    return samples
```

### ملاحظات

- تكميم INT8 يتطلب بيانات معايرة
- بعض الطبقات قد لا تكون مناسبة للتكميم
- يجب اختبار فقدان الدقة

---

## نشر TensorRT

### عملية التحويل

```python
import tensorrt as trt

def convert_to_tensorrt(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # تحليل نموذج ONNX
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())

    # إعداد خيارات التحسين
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB

    # تفعيل FP16
    config.set_flag(trt.BuilderFlag.FP16)

    # بناء المحرك
    engine = builder.build_engine(network, config)

    # الحفظ
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
```

### استخدام محرك TensorRT

```python
def inference_with_tensorrt(engine_path, input_data):
    # تحميل المحرك
    with open(engine_path, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # تخصيص الذاكرة
    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(output_size)

    # نسخ الإدخال
    cuda.memcpy_htod(d_input, input_data)

    # تنفيذ الاستدلال
    context.execute_v2([int(d_input), int(d_output)])

    # الحصول على الإخراج
    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output, d_output)

    return output
```

---

## تصدير ONNX

### PyTorch → ONNX

```python
import torch.onnx

def export_to_onnx(model, output_path):
    model.eval()

    # إنشاء إدخال عينة
    dummy_input = torch.randn(1, 22, 19, 19)

    # التصدير
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

### التحقق من نموذج ONNX

```python
import onnx
import onnxruntime as ort

# التحقق من هيكل النموذج
model = onnx.load("model.onnx")
onnx.checker.check_model(model)

# اختبار الاستدلال
session = ort.InferenceSession("model.onnx")
output = session.run(None, {'input': input_data})
```

---

## النشر على مختلف المنصات

### نشر الخوادم

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

### تكامل تطبيقات سطح المكتب

```python
# تضمين KataGo في تطبيق Python
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

### نشر الأجهزة المحمولة

#### iOS (Core ML)

```python
import coremltools as ct

# التحويل إلى Core ML
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

# التحويل إلى TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open('katago.tflite', 'wb') as f:
    f.write(tflite_model)
```

### الأنظمة المدمجة

#### Raspberry Pi

```bash
# استخدام واجهة Eigen (CPU فقط)
./katago gtp -model kata-b10c128.bin.gz -config rpi.cfg
```

```ini
# rpi.cfg - إعدادات Raspberry Pi المحسنة
numSearchThreads = 4
maxVisits = 100
nnMaxBatchSize = 1
```

#### NVIDIA Jetson

```bash
# استخدام واجهة CUDA
./katago gtp -model kata-b18c384.bin.gz -config jetson.cfg
```

---

## مقارنة الأداء

### أداء طرق النشر المختلفة

| طريقة النشر | الأجهزة | Playouts/ثانية |
|------------|---------|----------------|
| CUDA FP32 | RTX 3080 | ~3000 |
| CUDA FP16 | RTX 3080 | ~5000 |
| TensorRT FP16 | RTX 3080 | ~6500 |
| OpenCL | M1 Pro | ~1500 |
| Core ML | M1 Pro | ~1800 |
| TFLite | Pixel 7 | ~50 |
| Eigen | RPi 4 | ~15 |

### مقارنة حجم النماذج

| التنسيق | حجم b18c384 |
|---------|-------------|
| أصلي (.bin.gz) | ~140 MB |
| ONNX FP32 | ~280 MB |
| ONNX FP16 | ~140 MB |
| TensorRT FP16 | ~100 MB |
| TFLite FP16 | ~140 MB |

---

## قائمة فحص النشر

- [ ] اختيار دقة التكميم المناسبة
- [ ] إعداد بيانات المعايرة (INT8)
- [ ] التصدير إلى التنسيق المستهدف
- [ ] التحقق من أن فقدان الدقة مقبول
- [ ] اختبار أداء المنصة المستهدفة
- [ ] تحسين استخدام الذاكرة
- [ ] بناء عملية نشر آلية

---

## قراءات إضافية

- [واجهات GPU والتحسين](../gpu-optimization) — تحسين الأداء الأساسي
- [التقييم والاختبار المعياري](../evaluation) — التحقق من الأداء بعد النشر
- [التكامل مع مشروعك](../../hands-on/integration) — أمثلة تكامل API
