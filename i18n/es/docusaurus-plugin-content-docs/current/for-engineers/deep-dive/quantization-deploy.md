---
sidebar_position: 8
title: Cuantización y despliegue de modelos
description: Técnicas de cuantización de modelos KataGo, formatos de exportación y soluciones de despliegue multiplataforma
---

# Cuantización y despliegue de modelos

Este artículo presenta cómo cuantizar modelos KataGo para reducir los requisitos de recursos, así como soluciones de despliegue en diversas plataformas.

---

## Visión general de técnicas de cuantización

### ¿Por qué se necesita cuantización?

| Precisión | Tamaño | Velocidad | Pérdida de precisión |
|-----------|--------|-----------|---------------------|
| FP32 | 100% | Base | 0% |
| FP16 | 50% | +50% | ~0% |
| INT8 | 25% | +100% | \<1% |

### Tipos de cuantización

```
Cuantización post-entrenamiento (PTQ)
├── Simple y rápido
├── No requiere re-entrenamiento
└── Puede haber pérdida de precisión

Entrenamiento consciente de cuantización (QAT)
├── Mayor precisión
├── Requiere re-entrenamiento
└── Más complejo
```

---

## FP16 Media precisión

### Concepto

Convertir números de punto flotante de 32 bits a 16 bits:

```python
# Conversión FP32 → FP16
model_fp16 = model.half()

# Inferencia
with torch.cuda.amp.autocast():
    output = model_fp16(input.half())
```

### Configuración KataGo

```ini
# config.cfg
useFP16 = true           # Habilitar inferencia FP16
useFP16Storage = true    # Almacenamiento de resultados intermedios en FP16
```

### Impacto en el rendimiento

| Serie GPU | Aceleración FP16 |
|-----------|------------------|
| GTX 10xx | Ninguna (sin Tensor Core) |
| RTX 20xx | +30-50% |
| RTX 30xx | +50-80% |
| RTX 40xx | +80-100% |

---

## Cuantización INT8

### Proceso de cuantización

```python
import torch.quantization as quant

# 1. Preparar modelo
model.eval()
model.qconfig = quant.get_default_qconfig('fbgemm')

# 2. Preparar cuantización
model_prepared = quant.prepare(model)

# 3. Calibrar (usar datos representativos)
with torch.no_grad():
    for data in calibration_loader:
        model_prepared(data)

# 4. Convertir a modelo cuantizado
model_quantized = quant.convert(model_prepared)
```

### Datos de calibración

```python
def create_calibration_dataset(num_samples=1000):
    """Crear dataset de calibración"""
    samples = []

    # Muestrear de partidas reales
    for game in random_games(num_samples):
        position = random_position(game)
        features = encode_state(position)
        samples.append(features)

    return samples
```

### Notas

- La cuantización INT8 requiere datos de calibración
- Algunas capas pueden no ser adecuadas para cuantización
- Es necesario probar la pérdida de precisión

---

## Despliegue TensorRT

### Proceso de conversión

```python
import tensorrt as trt

def convert_to_tensorrt(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # Parsear modelo ONNX
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())

    # Configurar opciones de optimización
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB

    # Habilitar FP16
    config.set_flag(trt.BuilderFlag.FP16)

    # Construir motor
    engine = builder.build_engine(network, config)

    # Guardar
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
```

### Usar motor TensorRT

```python
def inference_with_tensorrt(engine_path, input_data):
    # Cargar motor
    with open(engine_path, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Asignar memoria
    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(output_size)

    # Copiar entrada
    cuda.memcpy_htod(d_input, input_data)

    # Ejecutar inferencia
    context.execute_v2([int(d_input), int(d_output)])

    # Obtener salida
    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output, d_output)

    return output
```

---

## Exportación ONNX

### PyTorch → ONNX

```python
import torch.onnx

def export_to_onnx(model, output_path):
    model.eval()

    # Crear entrada de ejemplo
    dummy_input = torch.randn(1, 22, 19, 19)

    # Exportar
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

### Validar modelo ONNX

```python
import onnx
import onnxruntime as ort

# Validar estructura del modelo
model = onnx.load("model.onnx")
onnx.checker.check_model(model)

# Probar inferencia
session = ort.InferenceSession("model.onnx")
output = session.run(None, {'input': input_data})
```

---

## Despliegue multiplataforma

### Despliegue en servidor

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

### Integración en aplicación de escritorio

```python
# Integrar KataGo en aplicación Python
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

### Despliegue en dispositivos móviles

#### iOS (Core ML)

```python
import coremltools as ct

# Convertir a Core ML
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

# Convertir a TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open('katago.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Sistemas embebidos

#### Raspberry Pi

```bash
# Usar backend Eigen (solo CPU)
./katago gtp -model kata-b10c128.bin.gz -config rpi.cfg
```

```ini
# rpi.cfg - Configuración optimizada para Raspberry Pi
numSearchThreads = 4
maxVisits = 100
nnMaxBatchSize = 1
```

#### NVIDIA Jetson

```bash
# Usar backend CUDA
./katago gtp -model kata-b18c384.bin.gz -config jetson.cfg
```

---

## Comparación de rendimiento

### Rendimiento de diferentes métodos de despliegue

| Método de despliegue | Hardware | Playouts/seg |
|---------------------|----------|--------------|
| CUDA FP32 | RTX 3080 | ~3000 |
| CUDA FP16 | RTX 3080 | ~5000 |
| TensorRT FP16 | RTX 3080 | ~6500 |
| OpenCL | M1 Pro | ~1500 |
| Core ML | M1 Pro | ~1800 |
| TFLite | Pixel 7 | ~50 |
| Eigen | RPi 4 | ~15 |

### Comparación de tamaño de modelo

| Formato | Tamaño b18c384 |
|---------|----------------|
| Original (.bin.gz) | ~140 MB |
| ONNX FP32 | ~280 MB |
| ONNX FP16 | ~140 MB |
| TensorRT FP16 | ~100 MB |
| TFLite FP16 | ~140 MB |

---

## Lista de verificación de despliegue

- [ ] Elegir la precisión de cuantización adecuada
- [ ] Preparar datos de calibración (INT8)
- [ ] Exportar al formato objetivo
- [ ] Verificar que la pérdida de precisión es aceptable
- [ ] Probar el rendimiento en la plataforma objetivo
- [ ] Optimizar el uso de memoria
- [ ] Establecer flujo de despliegue automatizado

---

## Lectura adicional

- [Backend GPU y optimización](../gpu-optimization) — Optimización básica de rendimiento
- [Evaluación y benchmarking](../evaluation) — Verificar rendimiento post-despliegue
- [Integración en tu proyecto](../../hands-on/integration) — Ejemplos de integración de API
