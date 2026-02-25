---
sidebar_position: 7
title: Arsitektur Pelatihan Terdistribusi
description: Arsitektur sistem pelatihan terdistribusi KataGo, Self-play Worker, dan alur rilis model
---

# Arsitektur Pelatihan Terdistribusi

Artikel ini memperkenalkan arsitektur sistem pelatihan terdistribusi KataGo, menjelaskan bagaimana model terus ditingkatkan melalui daya komputasi komunitas global.

---

## Gambaran Arsitektur Sistem

```mermaid
flowchart TB
    subgraph Workers["Self-play Workers"]
        W1["Worker 1<br/>(Self-play)"]
        W2["Worker 2<br/>(Self-play)"]
        WN["Worker N<br/>(Self-play)"]
    end

    W1 --> Server
    W2 --> Server
    WN --> Server

    Server["Server Pelatihan<br/>(Data Collection)"]
    Server --> Training["Proses Pelatihan<br/>(Model Training)"]
    Training --> Release["Rilis Model Baru<br/>(Model Release)"]
```

---

## Self-play Worker

### Alur Kerja

Setiap Worker menjalankan loop berikut:

```python
def self_play_worker():
    while True:
        # 1. Unduh model terbaru
        model = download_latest_model()

        # 2. Jalankan self-play
        games = []
        for _ in range(batch_size):
            game = play_game(model)
            games.append(game)

        # 3. Upload data pertandingan
        upload_games(games)

        # 4. Periksa model baru
        if new_model_available():
            model = download_latest_model()
```

### Pembuatan Pertandingan

```python
def play_game(model):
    """Jalankan satu pertandingan self-play"""
    game = Game()
    positions = []

    while not game.is_terminal():
        # Pencarian MCTS
        mcts = MCTS(model, num_simulations=800)
        policy = mcts.get_policy(game.state)

        # Tambahkan noise Dirichlet (meningkatkan eksplorasi)
        if game.move_count < 30:
            policy = add_dirichlet_noise(policy)

        # Pilih aksi berdasarkan policy
        if game.move_count < 30:
            # 30 langkah pertama gunakan sampling temperatur
            action = sample_with_temperature(policy, temp=1.0)
        else:
            # Setelahnya pilihan greedy
            action = np.argmax(policy)

        # Catat data pelatihan
        positions.append({
            'state': game.state.copy(),
            'policy': policy,
            'player': game.current_player
        })

        game.play(action)

    # Tandai menang/kalah
    winner = game.get_winner()
    for pos in positions:
        pos['value'] = 1.0 if pos['player'] == winner else -1.0

    return positions
```

### Format Data

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

## Server Pengumpulan Data

### Fungsi

1. **Menerima data pertandingan**: Kumpulkan pertandingan dari Workers
2. **Validasi data**: Periksa format, filter anomali
3. **Penyimpanan data**: Tulis ke dataset pelatihan
4. **Monitoring statistik**: Lacak jumlah pertandingan, status Worker

### Validasi Data

```python
def validate_game(game_data):
    """Validasi data pertandingan"""
    checks = [
        len(game_data['positions']) > 10,  # Minimal langkah
        len(game_data['positions']) < 500,  # Maksimal langkah
        all(is_valid_policy(p['policy']) for p in game_data['positions']),
        game_data['rules'] in SUPPORTED_RULES,
    ]
    return all(checks)
```

### Struktur Penyimpanan Data

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

## Alur Pelatihan

### Loop Pelatihan

```python
def training_loop():
    model = load_model()
    optimizer = Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        # Muat data pertandingan terbaru
        dataset = load_recent_games(num_games=100000)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        for batch in dataloader:
            states = batch['states']
            target_policies = batch['policies']
            target_values = batch['values']

            # Forward pass
            pred_policies, pred_values = model(states)

            # Hitung loss
            policy_loss = cross_entropy(pred_policies, target_policies)
            value_loss = mse_loss(pred_values, target_values)
            loss = policy_loss + value_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluasi berkala
        if epoch % 100 == 0:
            evaluate_model(model)
```

### Fungsi Loss

KataGo menggunakan beberapa komponen loss:

```python
def compute_loss(predictions, targets):
    # Policy loss (cross entropy)
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

## Evaluasi dan Rilis Model

### Evaluasi Elo

Model baru perlu bertanding melawan model lama untuk mengevaluasi kekuatan:

```python
def evaluate_new_model(new_model, baseline_model, num_games=400):
    """Evaluasi Elo model baru"""
    wins = 0
    losses = 0
    draws = 0

    for _ in range(num_games // 2):
        # Model baru main Hitam
        result = play_game(new_model, baseline_model)
        if result == 'black_wins':
            wins += 1
        elif result == 'white_wins':
            losses += 1
        else:
            draws += 1

        # Model baru main Putih
        result = play_game(baseline_model, new_model)
        if result == 'white_wins':
            wins += 1
        elif result == 'black_wins':
            losses += 1
        else:
            draws += 1

    # Hitung selisih Elo
    win_rate = (wins + 0.5 * draws) / num_games
    elo_diff = 400 * math.log10(win_rate / (1 - win_rate))

    return elo_diff
```

### Kondisi Rilis

```python
def should_release_model(new_model, current_best):
    """Tentukan apakah merilis model baru"""
    elo_diff = evaluate_new_model(new_model, current_best)

    # Kondisi: Peningkatan Elo melebihi threshold
    if elo_diff > 20:
        return True

    # Atau: Mencapai jumlah langkah pelatihan tertentu
    if training_steps % 10000 == 0:
        return True

    return False
```

### Penamaan Versi Model

```
kata1-b18c384nbt-s{steps}-d{data}.bin.gz

Contoh:
kata1-b18c384nbt-s9996604416-d4316597426.bin.gz
├── kata1: Seri pelatihan
├── b18c384nbt: Arsitektur (18 blok residual, 384 channel)
├── s9996604416: Langkah pelatihan
└── d4316597426: Jumlah data pelatihan
```

---

## Panduan Partisipasi KataGo Training

### Kebutuhan Sistem

| Item | Kebutuhan Minimum | Kebutuhan yang Disarankan |
|------|-------------------|---------------------------|
| GPU | GTX 1060 | RTX 3060+ |
| VRAM | 4 GB | 8 GB+ |
| Jaringan | 10 Mbps | 50 Mbps+ |
| Waktu operasi | Berjalan terus | 24/7 |

### Instalasi Worker

```bash
# Unduh Worker
wget https://katagotraining.org/download/worker

# Konfigurasi
./katago contribute -config contribute.cfg

# Mulai berkontribusi
./katago contribute
```

### File Konfigurasi

```ini
# contribute.cfg

# Pengaturan server
serverUrl = https://katagotraining.org/

# Username (untuk statistik)
username = your_username

# Pengaturan GPU
numNNServerThreadsPerModel = 1
nnMaxBatchSize = 16

# Pengaturan pertandingan
gamesPerBatch = 25
```

### Monitoring Kontribusi

```bash
# Lihat statistik
https://katagotraining.org/contributions/

# Log lokal
tail -f katago_contribute.log
```

---

## Statistik Pelatihan

### Milestone Pelatihan KataGo

| Waktu | Jumlah Pertandingan | Elo |
|-------|---------------------|-----|
| 2019.06 | 10M | Awal |
| 2020.01 | 100M | +500 |
| 2021.01 | 500M | +800 |
| 2022.01 | 1B | +1000 |
| 2024.01 | 5B+ | +1200 |

### Kontributor Komunitas

- Ratusan kontributor global
- Total ribuan GPU-tahun daya komputasi
- Berjalan terus 24/7

---

## Topik Lanjutan

### Curriculum Learning

Tingkatkan kesulitan pelatihan secara bertahap:

```python
def get_training_config(training_step):
    if training_step < 100000:
        return {'board_size': 9, 'visits': 200}
    elif training_step < 500000:
        return {'board_size': 13, 'visits': 400}
    else:
        return {'board_size': 19, 'visits': 800}
```

### Augmentasi Data

Gunakan simetri papan untuk meningkatkan jumlah data:

```python
def augment_position(state, policy):
    """8 transformasi simetri"""
    augmented = []

    for rotation in [0, 90, 180, 270]:
        for flip in [False, True]:
            aug_state = transform(state, rotation, flip)
            aug_policy = transform_policy(policy, rotation, flip)
            augmented.append((aug_state, aug_policy))

    return augmented
```

---

## Bacaan Lanjutan

- [Analisis Mekanisme Pelatihan KataGo](../training) — Detail alur pelatihan
- [Berkontribusi ke Komunitas Open Source](../contributing) — Cara berkontribusi kode
- [Evaluasi dan Benchmark](../evaluation) — Metode evaluasi model
