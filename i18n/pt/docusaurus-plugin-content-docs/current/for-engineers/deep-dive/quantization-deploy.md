---
sidebar_position: 8
title: Quantização e Implantação de Modelos
description: Técnicas de quantização de modelos KataGo, formatos de exportação e soluções de implantação para várias plataformas
---

# Quantização e Implantação de Modelos

Este artigo apresenta como quantizar modelos KataGo para reduzir requisitos de recursos, além de soluções de implantação para várias plataformas.

---

## Visão Geral das Técnicas de Quantização

### Por que Quantizar?

| Precisão | Tamanho | Velocidade | Perda de Precisão |
|----------|---------|------------|-------------------|
| FP32 | 100% | Base | 0% |
| FP16 | 50% | +50% | ~0% |
| INT8 | 25% | +100% | \<1% |

### Tipos de Quantização

```
Quantização Pós-Treinamento (PTQ)
├── Simples e rápida
├── Não requer retreinamento
└── Pode ter perda de precisão

Treinamento com Consciência de Quantização (QAT)
├── Maior precisão
├── Requer retreinamento
└── Mais complexo
```

---

## FP16 Meia Precisão

### Conceito

Converter números de ponto flutuante de 32 bits para 16 bits:

```python
# Conversão FP32 → FP16
model_fp16 = model.half()

# Inferência
with torch.cuda.amp.autocast():
    output = model_fp16(input.half())
```

### Configuração do KataGo

```ini
# config.cfg
useFP16 = true           # Habilitar inferência FP16
useFP16Storage = true    # Armazenar resultados intermediários em FP16
```

### Impacto no Desempenho

| Série de GPU | Aceleração FP16 |
|--------------|-----------------|
| GTX 10xx | Nenhuma (sem Tensor Core) |
| RTX 20xx | +30-50% |
| RTX 30xx | +50-80% |
| RTX 40xx | +80-100% |

---

## Quantização INT8

### Fluxo de Quantização

```python
import torch.quantization as quant

# 1. Preparar modelo
model.eval()
model.qconfig = quant.get_default_qconfig('fbgemm')

# 2. Preparar quantização
model_prepared = quant.prepare(model)

# 3. Calibração (usando dados representativos)
with torch.no_grad():
    for data in calibration_loader:
        model_prepared(data)

# 4. Converter para modelo quantizado
model_quantized = quant.convert(model_prepared)
```

### Dados de Calibração

```python
def create_calibration_dataset(num_samples=1000):
    """Cria conjunto de dados de calibração"""
    samples = []

    # Amostrar de partidas reais
    for game in random_games(num_samples):
        position = random_position(game)
        features = encode_state(position)
        samples.append(features)

    return samples
```

### Observações

- Quantização INT8 requer dados de calibração
- Algumas camadas podem não ser adequadas para quantização
- Necessário testar perda de precisão

---

## Implantação com TensorRT

### Fluxo de Conversão

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

    # Configurar opções de otimização
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB

    # Habilitar FP16
    config.set_flag(trt.BuilderFlag.FP16)

    # Construir engine
    engine = builder.build_engine(network, config)

    # Salvar
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
```

### Usando Engine TensorRT

```python
def inference_with_tensorrt(engine_path, input_data):
    # Carregar engine
    with open(engine_path, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Alocar memória
    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(output_size)

    # Copiar entrada
    cuda.memcpy_htod(d_input, input_data)

    # Executar inferência
    context.execute_v2([int(d_input), int(d_output)])

    # Obter saída
    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output, d_output)

    return output
```

---

## Exportação ONNX

### PyTorch → ONNX

```python
import torch.onnx

def export_to_onnx(model, output_path):
    model.eval()

    # Criar entrada de exemplo
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

### Validar Modelo ONNX

```python
import onnx
import onnxruntime as ort

# Validar estrutura do modelo
model = onnx.load("model.onnx")
onnx.checker.check_model(model)

# Testar inferência
session = ort.InferenceSession("model.onnx")
output = session.run(None, {'input': input_data})
```

---

## Implantação em Várias Plataformas

### Implantação em Servidor

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

### Integração com Aplicativo Desktop

```python
# Embutir KataGo em aplicação Python
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

### Implantação em Dispositivos Móveis

#### iOS (Core ML)

```python
import coremltools as ct

# Converter para Core ML
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

# Converter para TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open('katago.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Sistemas Embarcados

#### Raspberry Pi

```bash
# Usar backend Eigen (CPU pura)
./katago gtp -model kata-b10c128.bin.gz -config rpi.cfg
```

```ini
# rpi.cfg - Configuração otimizada para Raspberry Pi
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

## Comparação de Desempenho

### Desempenho de Diferentes Métodos de Implantação

| Método de Implantação | Hardware | Playouts/seg |
|----------------------|----------|--------------|
| CUDA FP32 | RTX 3080 | ~3000 |
| CUDA FP16 | RTX 3080 | ~5000 |
| TensorRT FP16 | RTX 3080 | ~6500 |
| OpenCL | M1 Pro | ~1500 |
| Core ML | M1 Pro | ~1800 |
| TFLite | Pixel 7 | ~50 |
| Eigen | RPi 4 | ~15 |

### Comparação de Tamanho de Modelo

| Formato | Tamanho b18c384 |
|---------|-----------------|
| Original (.bin.gz) | ~140 MB |
| ONNX FP32 | ~280 MB |
| ONNX FP16 | ~140 MB |
| TensorRT FP16 | ~100 MB |
| TFLite FP16 | ~140 MB |

---

## Checklist de Implantação

- [ ] Escolher precisão de quantização adequada
- [ ] Preparar dados de calibração (INT8)
- [ ] Exportar para formato de destino
- [ ] Validar que perda de precisão é aceitável
- [ ] Testar desempenho na plataforma alvo
- [ ] Otimizar uso de memória
- [ ] Criar fluxo de implantação automatizado

---

## Leitura Adicional

- [Backend GPU e Otimização](../gpu-optimization) — Otimização básica de desempenho
- [Avaliação e Benchmarks](../evaluation) — Verificar desempenho após implantação
- [Integração no Seu Projeto](../../hands-on/integration) — Exemplos de integração via API
