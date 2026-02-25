---
sidebar_position: 3
title: Análisis del mecanismo de entrenamiento de KataGo
description: Comprender en profundidad el flujo de entrenamiento de auto-juego y las técnicas centrales de KataGo
---

# Análisis del mecanismo de entrenamiento de KataGo

Este artículo analiza en profundidad el mecanismo de entrenamiento de KataGo, ayudándote a comprender el principio de funcionamiento del entrenamiento por auto-juego.

---

## Visión general del entrenamiento

### Ciclo de entrenamiento

```
Modelo inicial → Auto-juego → Recopilar datos → Actualizar entrenamiento → Modelo más fuerte → Repetir
```

**Correspondencia con animaciones**:
- E5 Auto-juego ↔ Convergencia al punto fijo
- E6 Curva de fuerza ↔ Crecimiento en curva S
- H1 MDP ↔ Cadena de Markov

### Requisitos de hardware

| Escala del modelo | Memoria GPU | Tiempo de entrenamiento |
|-------------------|-------------|-------------------------|
| b6c96 | 4 GB | Varias horas |
| b10c128 | 8 GB | 1-2 días |
| b18c384 | 16 GB | 1-2 semanas |
| b40c256 | 24 GB+ | Varias semanas |

---

## Configuración del entorno

### Instalar dependencias

```bash
# Entorno Python
conda create -n katago python=3.10
conda activate katago

# PyTorch (versión CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Otras dependencias
pip install numpy h5py tqdm tensorboard
```

### Obtener el código de entrenamiento

```bash
git clone https://github.com/lightvector/KataGo.git
cd KataGo/python
```

---

## Configuración de entrenamiento

### Estructura del archivo de configuración

```yaml
# configs/train_config.yaml

# Arquitectura del modelo
model:
  num_blocks: 10          # Número de bloques residuales
  trunk_channels: 128     # Número de canales del tronco
  policy_channels: 32     # Canales de la cabeza Policy
  value_channels: 32      # Canales de la cabeza Value

# Parámetros de entrenamiento
training:
  batch_size: 256
  learning_rate: 0.001
  lr_schedule: "cosine"
  weight_decay: 0.0001
  epochs: 100

# Parámetros de auto-juego
selfplay:
  num_games_per_iteration: 1000
  max_visits: 600
  temperature: 1.0
  temperature_drop_move: 20

# Configuración de datos
data:
  max_history_games: 500000
  shuffle_buffer_size: 100000
```

### Referencia de escalas de modelo

| Nombre | num_blocks | trunk_channels | Parámetros |
|--------|-----------|----------------|------------|
| b6c96 | 6 | 96 | ~1M |
| b10c128 | 10 | 128 | ~3M |
| b18c384 | 18 | 384 | ~20M |
| b40c256 | 40 | 256 | ~45M |

**Correspondencia con animaciones**:
- F2 Tamaño de red vs fuerza: Escalado de capacidad
- F6 Leyes de escalado neuronal: Relación log-log

---

## Flujo de entrenamiento

### Paso 1: Inicializar el modelo

```python
# init_model.py
import torch
from model import KataGoModel

config = {
    'num_blocks': 10,
    'trunk_channels': 128,
    'input_features': 22,
    'policy_size': 362,  # 361 + pass
}

model = KataGoModel(config)
torch.save(model.state_dict(), 'model_init.pt')
print(f"Parámetros del modelo: {sum(p.numel() for p in model.parameters()):,}")
```

### Paso 2: Generar datos con auto-juego

```bash
# Compilar el motor C++
cd ../cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=CUDA
make -j$(nproc)

# Ejecutar auto-juego
./katago selfplay \
  -model ../python/model_init.pt \
  -output-dir ../python/selfplay_data \
  -config selfplay.cfg \
  -num-games 1000
```

Configuración de auto-juego (selfplay.cfg):

```ini
maxVisits = 600
numSearchThreads = 4

# Configuración de temperatura (aumentar exploración)
chosenMoveTemperature = 1.0
chosenMoveTemperatureEarly = 1.0
chosenMoveTemperatureHalflife = 20

# Ruido de Dirichlet (aumentar diversidad)
rootNoiseEnabled = true
rootDirichletNoiseTotalConcentration = 10.83
rootDirichletNoiseWeight = 0.25
```

**Correspondencia con animaciones**:
- C3 Exploración vs explotación: Parámetro de temperatura
- E10 Ruido de Dirichlet: Exploración del nodo raíz

### Paso 3: Entrenar la red neuronal

```python
# train.py
import torch
from torch.utils.data import DataLoader
from model import KataGoModel
from dataset import SelfPlayDataset

# Cargar datos
dataset = SelfPlayDataset('selfplay_data/')
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# Cargar modelo
model = KataGoModel(config)
model.load_state_dict(torch.load('model_init.pt'))
model = model.cuda()

# Optimizador
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

# Programador de tasa de aprendizaje
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=0.00001
)

# Ciclo de entrenamiento
for epoch in range(100):
    model.train()
    total_loss = 0

    for batch in dataloader:
        inputs = batch['inputs'].cuda()
        policy_target = batch['policy'].cuda()
        value_target = batch['value'].cuda()
        ownership_target = batch['ownership'].cuda()

        # Propagación hacia adelante
        policy_pred, value_pred, ownership_pred = model(inputs)

        # Calcular pérdida
        policy_loss = torch.nn.functional.cross_entropy(
            policy_pred, policy_target
        )
        value_loss = torch.nn.functional.mse_loss(
            value_pred, value_target
        )
        ownership_loss = torch.nn.functional.mse_loss(
            ownership_pred, ownership_target
        )

        loss = policy_loss + value_loss + 0.5 * ownership_loss

        # Retropropagación
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")

    # Guardar checkpoint
    torch.save(model.state_dict(), f'model_epoch{epoch}.pt')
```

**Correspondencia con animaciones**:
- D5 Descenso de gradiente: optimizer.step()
- K2 Momentum: Optimizador Adam
- K4 Decaimiento de tasa de aprendizaje: CosineAnnealingLR
- K5 Recorte de gradiente: clip_grad_norm_

### Paso 4: Evaluar e iterar

```bash
# Evaluar nuevo modelo vs modelo anterior
./katago match \
  -model1 model_epoch99.pt \
  -model2 model_init.pt \
  -num-games 100 \
  -output match_results.txt
```

Si el nuevo modelo tiene una tasa de victoria > 55%, reemplaza al modelo anterior y pasa a la siguiente iteración.

---

## Detalle de funciones de pérdida

### Policy Loss

```python
# Pérdida de entropía cruzada
policy_loss = -sum(target * log(pred))
```

Objetivo: Hacer que la distribución de probabilidad predicha se acerque al resultado de búsqueda MCTS.

**Correspondencia con animaciones**:
- J1 Entropía de política: Entropía cruzada
- J2 Divergencia KL: Distancia entre distribuciones

### Value Loss

```python
# Error cuadrático medio
value_loss = (pred - actual_result)^2
```

Objetivo: Predecir el resultado final del juego (victoria/derrota/empate).

### Ownership Loss

```python
# Predicción de propiedad por punto
ownership_loss = mean((pred - actual_ownership)^2)
```

Objetivo: Predecir la propiedad final de cada posición.

---

## Técnicas avanzadas

### 1. Aumento de datos

Aprovechar la simetría del tablero:

```python
def augment_data(board, policy, ownership):
    """Aumento de datos para las 8 transformaciones del grupo D4"""
    augmented = []

    for rotation in range(4):
        for flip in [False, True]:
            # Rotación y volteo
            aug_board = transform(board, rotation, flip)
            aug_policy = transform(policy, rotation, flip)
            aug_ownership = transform(ownership, rotation, flip)
            augmented.append((aug_board, aug_policy, aug_ownership))

    return augmented
```

**Correspondencia con animaciones**:
- A9 Simetría del tablero: Grupo D4
- L4 Aumento de datos: Uso de simetría

### 2. Aprendizaje por currículum

De simple a complejo:

```python
# Primero entrenar con menos búsquedas
schedule = [
    (100, 10000),   # 100 visitas, 10000 partidas
    (200, 20000),   # 200 visitas, 20000 partidas
    (400, 50000),   # 400 visitas, 50000 partidas
    (600, 100000),  # 600 visitas, 100000 partidas
]
```

**Correspondencia con animaciones**:
- E12 Currículum de entrenamiento: Aprendizaje por currículum

### 3. Entrenamiento de precisión mixta

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    policy_pred, value_pred, ownership_pred = model(inputs)
    loss = compute_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 4. Entrenamiento multi-GPU

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Inicializar distribuido
dist.init_process_group(backend='nccl')

# Envolver el modelo
model = DistributedDataParallel(model)
```

---

## Monitoreo y depuración

### Monitoreo con TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/training')

# Registrar pérdida
writer.add_scalar('Loss/policy', policy_loss, step)
writer.add_scalar('Loss/value', value_loss, step)
writer.add_scalar('Loss/total', total_loss, step)

# Registrar tasa de aprendizaje
writer.add_scalar('LR', scheduler.get_last_lr()[0], step)
```

```bash
tensorboard --logdir runs
```

### Problemas comunes

| Problema | Posible causa | Solución |
|----------|---------------|----------|
| La pérdida no baja | Tasa de aprendizaje muy baja/alta | Ajustar tasa de aprendizaje |
| La pérdida oscila | Tamaño de lote muy pequeño | Aumentar tamaño de lote |
| Sobreajuste | Datos insuficientes | Generar más datos de auto-juego |
| La fuerza no aumenta | Muy pocas búsquedas | Aumentar maxVisits |

**Correspondencia con animaciones**:
- L1 Sobreajuste: Sobre-adaptación
- L2 Regularización: weight_decay
- D6 Efecto de tasa de aprendizaje: Ajuste de parámetros

---

## Sugerencias para experimentos a pequeña escala

Si solo quieres experimentar, se recomienda:

1. **Usar tablero 9×9**: Reduce significativamente el cálculo
2. **Usar modelo pequeño**: b6c96 es suficiente para experimentar
3. **Reducir número de búsquedas**: 100-200 visitas
4. **Ajustar modelo pre-entrenado**: Más rápido que empezar desde cero

```bash
# Configuración para tablero 9×9
boardSize = 9
maxVisits = 100
```

---

## Lectura adicional

- [Guía del código fuente](../source-code) — Entender la estructura del código
- [Participar en la comunidad de código abierto](../contributing) — Unirse al entrenamiento distribuido
- [Innovaciones clave de KataGo](../../how-it-works/katago-innovations) — El secreto de la eficiencia 50x
