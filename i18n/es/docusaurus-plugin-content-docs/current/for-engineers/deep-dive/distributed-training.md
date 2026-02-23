---
sidebar_position: 7
title: Arquitectura de entrenamiento distribuido
description: Arquitectura del sistema de entrenamiento distribuido de KataGo, Self-play Worker y flujo de publicación de modelos
---

# Arquitectura de entrenamiento distribuido

Este artículo presenta la arquitectura del sistema de entrenamiento distribuido de KataGo, explicando cómo mejorar continuamente los modelos a través del poder de cómputo de la comunidad global.

---

## Visión general de la arquitectura del sistema

```
┌─────────────────────────────────────────────────────────────┐
│           Sistema de entrenamiento distribuido KataGo       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│   │  Worker 1   │    │  Worker 2   │    │  Worker N   │   │
│   │  (Self-play)│    │  (Self-play)│    │  (Self-play)│   │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘   │
│          │                  │                  │           │
│          └────────────┬─────┴──────────────────┘           │
│                       │                                     │
│                       ▼                                     │
│            ┌─────────────────────┐                         │
│            │  Servidor de        │                         │
│            │  entrenamiento      │                         │
│            │  (Data Collection)  │                         │
│            └──────────┬──────────┘                         │
│                       │                                     │
│                       ▼                                     │
│            ┌─────────────────────┐                         │
│            │  Proceso de         │                         │
│            │  entrenamiento      │                         │
│            │  (Model Training)   │                         │
│            └──────────┬──────────┘                         │
│                       │                                     │
│                       ▼                                     │
│            ┌─────────────────────┐                         │
│            │  Publicación de     │                         │
│            │  nuevo modelo       │                         │
│            │  (Model Release)    │                         │
│            └─────────────────────┘                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Self-play Worker

### Flujo de trabajo

Cada Worker ejecuta el siguiente ciclo:

```python
def self_play_worker():
    while True:
        # 1. Descargar el modelo más reciente
        model = download_latest_model()

        # 2. Ejecutar auto-juego
        games = []
        for _ in range(batch_size):
            game = play_game(model)
            games.append(game)

        # 3. Subir datos de partidas
        upload_games(games)

        # 4. Verificar nuevo modelo
        if new_model_available():
            model = download_latest_model()
```

### Generación de partidas

```python
def play_game(model):
    """Ejecutar una partida de auto-juego"""
    game = Game()
    positions = []

    while not game.is_terminal():
        # Búsqueda MCTS
        mcts = MCTS(model, num_simulations=800)
        policy = mcts.get_policy(game.state)

        # Agregar ruido de Dirichlet (aumentar exploración)
        if game.move_count < 30:
            policy = add_dirichlet_noise(policy)

        # Seleccionar acción según policy
        if game.move_count < 30:
            # Primeros 30 movimientos con muestreo de temperatura
            action = sample_with_temperature(policy, temp=1.0)
        else:
            # Después selección voraz
            action = np.argmax(policy)

        # Registrar datos de entrenamiento
        positions.append({
            'state': game.state.copy(),
            'policy': policy,
            'player': game.current_player
        })

        game.play(action)

    # Marcar victoria/derrota
    winner = game.get_winner()
    for pos in positions:
        pos['value'] = 1.0 if pos['player'] == winner else -1.0

    return positions
```

### Formato de datos

```json
{
  "version": 1,
  "rules": "chinese",
  "komi": 7.5,
  "board_size": 19,
  "positions": [
    {
      "move_number": 0,
      "board": "...",
      "policy": [0.01, 0.02, ...],
      "value": 1.0,
      "score": 2.5
    }
  ]
}
```

---

## Servidor de recolección de datos

### Funciones

1. **Recibir datos de partidas**: Recopilar partidas de los Workers
2. **Validación de datos**: Verificar formato, filtrar anomalías
3. **Almacenamiento de datos**: Escribir en el dataset de entrenamiento
4. **Monitoreo estadístico**: Rastrear cantidad de partidas, estado de Workers

### Validación de datos

```python
def validate_game(game_data):
    """Validar datos de partida"""
    checks = [
        len(game_data['positions']) > 10,  # Mínimo de movimientos
        len(game_data['positions']) < 500,  # Máximo de movimientos
        all(is_valid_policy(p['policy']) for p in game_data['positions']),
        game_data['rules'] in SUPPORTED_RULES,
    ]
    return all(checks)
```

### Estructura de almacenamiento de datos

```
training_data/
├── run_001/
│   ├── games_00001.npz
│   ├── games_00002.npz
│   └── ...
├── run_002/
│   └── ...
└── current/
    └── latest_games.npz
```

---

## Proceso de entrenamiento

### Ciclo de entrenamiento

```python
def training_loop():
    model = load_model()
    optimizer = Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        # Cargar los datos de partidas más recientes
        dataset = load_recent_games(num_games=100000)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        for batch in dataloader:
            states = batch['states']
            target_policies = batch['policies']
            target_values = batch['values']

            # Propagación hacia adelante
            pred_policies, pred_values = model(states)

            # Calcular pérdida
            policy_loss = cross_entropy(pred_policies, target_policies)
            value_loss = mse_loss(pred_values, target_values)
            loss = policy_loss + value_loss

            # Retropropagación
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluar periódicamente
        if epoch % 100 == 0:
            evaluate_model(model)
```

### Función de pérdida

KataGo usa múltiples términos de pérdida:

```python
def compute_loss(predictions, targets):
    # Pérdida de Policy (entropía cruzada)
    policy_loss = F.cross_entropy(
        predictions['policy'],
        targets['policy']
    )

    # Pérdida de Value (MSE)
    value_loss = F.mse_loss(
        predictions['value'],
        targets['value']
    )

    # Pérdida de Score (MSE)
    score_loss = F.mse_loss(
        predictions['score'],
        targets['score']
    )

    # Pérdida de Ownership (MSE)
    ownership_loss = F.mse_loss(
        predictions['ownership'],
        targets['ownership']
    )

    # Suma ponderada
    total_loss = (
        1.0 * policy_loss +
        1.0 * value_loss +
        0.5 * score_loss +
        0.5 * ownership_loss
    )

    return total_loss
```

---

## Evaluación y publicación de modelos

### Evaluación Elo

El nuevo modelo debe jugar contra el modelo anterior para evaluar su fuerza:

```python
def evaluate_new_model(new_model, baseline_model, num_games=400):
    """Evaluar el Elo del nuevo modelo"""
    wins = 0
    losses = 0
    draws = 0

    for _ in range(num_games // 2):
        # Nuevo modelo juega con Negro
        result = play_game(new_model, baseline_model)
        if result == 'black_wins':
            wins += 1
        elif result == 'white_wins':
            losses += 1
        else:
            draws += 1

        # Nuevo modelo juega con Blanco
        result = play_game(baseline_model, new_model)
        if result == 'white_wins':
            wins += 1
        elif result == 'black_wins':
            losses += 1
        else:
            draws += 1

    # Calcular diferencia de Elo
    win_rate = (wins + 0.5 * draws) / num_games
    elo_diff = 400 * math.log10(win_rate / (1 - win_rate))

    return elo_diff
```

### Condiciones de publicación

```python
def should_release_model(new_model, current_best):
    """Decidir si publicar el nuevo modelo"""
    elo_diff = evaluate_new_model(new_model, current_best)

    # Condición: Mejora de Elo supera el umbral
    if elo_diff > 20:
        return True

    # O: Alcanza cierto número de pasos de entrenamiento
    if training_steps % 10000 == 0:
        return True

    return False
```

### Nomenclatura de versiones de modelo

```
kata1-b18c384nbt-s{steps}-d{data}.bin.gz

Ejemplo:
kata1-b18c384nbt-s9996604416-d4316597426.bin.gz
├── kata1: Serie de entrenamiento
├── b18c384nbt: Arquitectura (18 bloques residuales, 384 canales)
├── s9996604416: Pasos de entrenamiento
└── d4316597426: Cantidad de datos de entrenamiento
```

---

## Guía de participación en KataGo Training

### Requisitos del sistema

| Elemento | Requisito mínimo | Requisito recomendado |
|----------|------------------|----------------------|
| GPU | GTX 1060 | RTX 3060+ |
| VRAM | 4 GB | 8 GB+ |
| Red | 10 Mbps | 50 Mbps+ |
| Tiempo de ejecución | Continuo | 24/7 |

### Instalar Worker

```bash
# Descargar Worker
wget https://katagotraining.org/download/worker

# Configurar
./katago contribute -config contribute.cfg

# Comenzar a contribuir
./katago contribute
```

### Archivo de configuración

```ini
# contribute.cfg

# Configuración del servidor
serverUrl = https://katagotraining.org/

# Nombre de usuario (para estadísticas)
username = your_username

# Configuración de GPU
numNNServerThreadsPerModel = 1
nnMaxBatchSize = 16

# Configuración de partidas
gamesPerBatch = 25
```

### Monitorear contribuciones

```bash
# Ver estadísticas
https://katagotraining.org/contributions/

# Log local
tail -f katago_contribute.log
```

---

## Estadísticas de entrenamiento

### Hitos de entrenamiento de KataGo

| Fecha | Partidas | Elo |
|-------|----------|-----|
| 2019.06 | 10M | Inicial |
| 2020.01 | 100M | +500 |
| 2021.01 | 500M | +800 |
| 2022.01 | 1B | +1000 |
| 2024.01 | 5B+ | +1200 |

### Contribuidores de la comunidad

- Cientos de contribuidores globales
- Miles de años-GPU de poder de cómputo acumulado
- Funcionando 24/7 continuamente

---

## Temas avanzados

### Aprendizaje por currículum (Curriculum Learning)

Aumentar gradualmente la dificultad del entrenamiento:

```python
def get_training_config(training_step):
    if training_step < 100000:
        return {'board_size': 9, 'visits': 200}
    elif training_step < 500000:
        return {'board_size': 13, 'visits': 400}
    else:
        return {'board_size': 19, 'visits': 800}
```

### Aumento de datos

Usar la simetría del tablero para aumentar la cantidad de datos:

```python
def augment_position(state, policy):
    """8 transformaciones simétricas"""
    augmented = []

    for rotation in [0, 90, 180, 270]:
        for flip in [False, True]:
            aug_state = transform(state, rotation, flip)
            aug_policy = transform_policy(policy, rotation, flip)
            augmented.append((aug_state, aug_policy))

    return augmented
```

---

## Lectura adicional

- [Análisis del mecanismo de entrenamiento de KataGo](../training) — Detalles del proceso de entrenamiento
- [Participar en la comunidad de código abierto](../contributing) — Cómo contribuir código
- [Evaluación y benchmarking](../evaluation) — Métodos de evaluación de modelos
