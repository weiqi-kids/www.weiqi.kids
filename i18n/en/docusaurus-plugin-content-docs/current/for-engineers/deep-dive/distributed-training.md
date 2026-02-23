---
sidebar_position: 7
title: Distributed Training Architecture
description: KataGo distributed training system architecture, Self-play Worker, and model release process
---

# Distributed Training Architecture

This article introduces KataGo's distributed training system architecture, explaining how the global community's computing power continuously improves the model.

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│              KataGo Distributed Training System              │
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
│            │   Training Server   │                         │
│            │  (Data Collection)  │                         │
│            └──────────┬──────────┘                         │
│                       │                                     │
│                       ▼                                     │
│            ┌─────────────────────┐                         │
│            │  Training Process   │                         │
│            │  (Model Training)   │                         │
│            └──────────┬──────────┘                         │
│                       │                                     │
│                       ▼                                     │
│            ┌─────────────────────┐                         │
│            │   New Model Release │                         │
│            │  (Model Release)    │                         │
│            └─────────────────────┘                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Self-play Worker

### Workflow

Each Worker executes the following loop:

```python
def self_play_worker():
    while True:
        # 1. Download latest model
        model = download_latest_model()

        # 2. Execute self-play
        games = []
        for _ in range(batch_size):
            game = play_game(model)
            games.append(game)

        # 3. Upload game data
        upload_games(games)

        # 4. Check for new model
        if new_model_available():
            model = download_latest_model()
```

### Game Generation

```python
def play_game(model):
    """Execute one self-play game"""
    game = Game()
    positions = []

    while not game.is_terminal():
        # MCTS search
        mcts = MCTS(model, num_simulations=800)
        policy = mcts.get_policy(game.state)

        # Add Dirichlet noise (increase exploration)
        if game.move_count < 30:
            policy = add_dirichlet_noise(policy)

        # Select action based on policy
        if game.move_count < 30:
            # First 30 moves use temperature sampling
            action = sample_with_temperature(policy, temp=1.0)
        else:
            # Later moves greedy selection
            action = np.argmax(policy)

        # Record training data
        positions.append({
            'state': game.state.copy(),
            'policy': policy,
            'player': game.current_player
        })

        game.play(action)

    # Mark winner
    winner = game.get_winner()
    for pos in positions:
        pos['value'] = 1.0 if pos['player'] == winner else -1.0

    return positions
```

### Data Format

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

## Data Collection Server

### Functions

1. **Receive game data**: Collect games from Workers
2. **Data validation**: Check format, filter anomalies
3. **Data storage**: Write to training dataset
4. **Statistics monitoring**: Track game counts, Worker status

### Data Validation

```python
def validate_game(game_data):
    """Validate game data"""
    checks = [
        len(game_data['positions']) > 10,  # Minimum moves
        len(game_data['positions']) < 500,  # Maximum moves
        all(is_valid_policy(p['policy']) for p in game_data['positions']),
        game_data['rules'] in SUPPORTED_RULES,
    ]
    return all(checks)
```

### Data Storage Structure

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

## Training Process

### Training Loop

```python
def training_loop():
    model = load_model()
    optimizer = Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        # Load latest game data
        dataset = load_recent_games(num_games=100000)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        for batch in dataloader:
            states = batch['states']
            target_policies = batch['policies']
            target_values = batch['values']

            # Forward pass
            pred_policies, pred_values = model(states)

            # Compute loss
            policy_loss = cross_entropy(pred_policies, target_policies)
            value_loss = mse_loss(pred_values, target_values)
            loss = policy_loss + value_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Periodic evaluation
        if epoch % 100 == 0:
            evaluate_model(model)
```

### Loss Functions

KataGo uses multiple loss terms:

```python
def compute_loss(predictions, targets):
    # Policy loss (cross-entropy)
    policy_loss = F.cross_entropy(
        predictions['policy'],
        targets['policy']
    )

    # Value loss (MSE)
    value_loss = F.mse_loss(
        predictions['value'],
        targets['value']
    )

    # Score loss (MSE)
    score_loss = F.mse_loss(
        predictions['score'],
        targets['score']
    )

    # Ownership loss (MSE)
    ownership_loss = F.mse_loss(
        predictions['ownership'],
        targets['ownership']
    )

    # Weighted sum
    total_loss = (
        1.0 * policy_loss +
        1.0 * value_loss +
        0.5 * score_loss +
        0.5 * ownership_loss
    )

    return total_loss
```

---

## Model Evaluation & Release

### Elo Evaluation

New models need to play against old models to evaluate strength:

```python
def evaluate_new_model(new_model, baseline_model, num_games=400):
    """Evaluate new model's Elo"""
    wins = 0
    losses = 0
    draws = 0

    for _ in range(num_games // 2):
        # New model plays Black
        result = play_game(new_model, baseline_model)
        if result == 'black_wins':
            wins += 1
        elif result == 'white_wins':
            losses += 1
        else:
            draws += 1

        # New model plays White
        result = play_game(baseline_model, new_model)
        if result == 'white_wins':
            wins += 1
        elif result == 'black_wins':
            losses += 1
        else:
            draws += 1

    # Calculate Elo difference
    win_rate = (wins + 0.5 * draws) / num_games
    elo_diff = 400 * math.log10(win_rate / (1 - win_rate))

    return elo_diff
```

### Release Conditions

```python
def should_release_model(new_model, current_best):
    """Decide whether to release new model"""
    elo_diff = evaluate_new_model(new_model, current_best)

    # Condition: Elo improvement exceeds threshold
    if elo_diff > 20:
        return True

    # Or: Reached certain training steps
    if training_steps % 10000 == 0:
        return True

    return False
```

### Model Version Naming

```
kata1-b18c384nbt-s{steps}-d{data}.bin.gz

Example:
kata1-b18c384nbt-s9996604416-d4316597426.bin.gz
├── kata1: Training run
├── b18c384nbt: Architecture (18 residual blocks, 384 channels)
├── s9996604416: Training steps
└── d4316597426: Training data volume
```

---

## KataGo Training Participation Guide

### System Requirements

| Item | Minimum | Recommended |
|------|---------|-------------|
| GPU | GTX 1060 | RTX 3060+ |
| VRAM | 4 GB | 8 GB+ |
| Network | 10 Mbps | 50 Mbps+ |
| Runtime | Continuous | 24/7 |

### Install Worker

```bash
# Download Worker
wget https://katagotraining.org/download/worker

# Configure
./katago contribute -config contribute.cfg

# Start contributing
./katago contribute
```

### Configuration File

```ini
# contribute.cfg

# Server settings
serverUrl = https://katagotraining.org/

# Username (for statistics)
username = your_username

# GPU settings
numNNServerThreadsPerModel = 1
nnMaxBatchSize = 16

# Game settings
gamesPerBatch = 25
```

### Monitor Contributions

```bash
# View statistics
https://katagotraining.org/contributions/

# Local logs
tail -f katago_contribute.log
```

---

## Training Statistics

### KataGo Training Milestones

| Time | Games | Elo |
|------|-------|-----|
| 2019.06 | 10M | Initial |
| 2020.01 | 100M | +500 |
| 2021.01 | 500M | +800 |
| 2022.01 | 1B | +1000 |
| 2024.01 | 5B+ | +1200 |

### Community Contributors

- Hundreds of global contributors
- Thousands of GPU-years of compute accumulated
- Running 24/7 continuously

---

## Advanced Topics

### Curriculum Learning

Gradually increase training difficulty:

```python
def get_training_config(training_step):
    if training_step < 100000:
        return {'board_size': 9, 'visits': 200}
    elif training_step < 500000:
        return {'board_size': 13, 'visits': 400}
    else:
        return {'board_size': 19, 'visits': 800}
```

### Data Augmentation

Leverage board symmetry to increase data volume:

```python
def augment_position(state, policy):
    """8 symmetry transformations"""
    augmented = []

    for rotation in [0, 90, 180, 270]:
        for flip in [False, True]:
            aug_state = transform(state, rotation, flip)
            aug_policy = transform_policy(policy, rotation, flip)
            augmented.append((aug_state, aug_policy))

    return augmented
```

---

## Further Reading

- [KataGo Training Mechanism](../training) — Training process details
- [Contributing to Open Source](../contributing) — How to contribute code
- [Evaluation & Benchmarking](../evaluation) — Model evaluation methods
