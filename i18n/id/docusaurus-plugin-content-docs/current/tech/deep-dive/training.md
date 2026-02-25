---
sidebar_position: 3
title: Analisis Mekanisme Pelatihan KataGo
description: Memahami secara mendalam alur pelatihan self-play dan teknik inti KataGo
---

# Analisis Mekanisme Pelatihan KataGo

Artikel ini menganalisis secara mendalam mekanisme pelatihan KataGo, membantu Anda memahami prinsip kerja pelatihan self-play.

---

## Gambaran Pelatihan

### Siklus Pelatihan

```
Model Awal → Self-play → Kumpulkan Data → Update Pelatihan → Model Lebih Kuat → Ulangi
```

**Padanan Animasi**:
- E5 Self-play ↔ Konvergensi titik tetap
- E6 Kurva kekuatan ↔ Pertumbuhan kurva S
- H1 MDP ↔ Rantai Markov

### Kebutuhan Hardware

| Skala Model | Memori GPU | Waktu Pelatihan |
|-------------|------------|-----------------|
| b6c96 | 4 GB | Beberapa jam |
| b10c128 | 8 GB | 1-2 hari |
| b18c384 | 16 GB | 1-2 minggu |
| b40c256 | 24 GB+ | Beberapa minggu |

---

## Pengaturan Lingkungan

### Instalasi Dependensi

```bash
# Lingkungan Python
conda create -n katago python=3.10
conda activate katago

# PyTorch (versi CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Dependensi lainnya
pip install numpy h5py tqdm tensorboard
```

### Mendapatkan Kode Pelatihan

```bash
git clone https://github.com/lightvector/KataGo.git
cd KataGo/python
```

---

## Konfigurasi Pelatihan

### Struktur File Konfigurasi

```yaml
# configs/train_config.yaml

# Arsitektur model
model:
  num_blocks: 10          # Jumlah blok residual
  trunk_channels: 128     # Jumlah channel trunk
  policy_channels: 32     # Jumlah channel Policy head
  value_channels: 32      # Jumlah channel Value head

# Parameter pelatihan
training:
  batch_size: 256
  learning_rate: 0.001
  lr_schedule: "cosine"
  weight_decay: 0.0001
  epochs: 100

# Parameter self-play
selfplay:
  num_games_per_iteration: 1000
  max_visits: 600
  temperature: 1.0
  temperature_drop_move: 20

# Konfigurasi data
data:
  max_history_games: 500000
  shuffle_buffer_size: 100000
```

### Referensi Skala Model

| Nama | num_blocks | trunk_channels | Jumlah Parameter |
|------|------------|----------------|------------------|
| b6c96 | 6 | 96 | ~1M |
| b10c128 | 10 | 128 | ~3M |
| b18c384 | 18 | 384 | ~20M |
| b40c256 | 40 | 256 | ~45M |

**Padanan Animasi**:
- F2 Ukuran network vs kekuatan: Penskalaan kapasitas
- F6 Hukum penskalaan neural: Hubungan log-log

---

## Alur Pelatihan

### Langkah 1: Inisialisasi Model

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
print(f"Jumlah parameter model: {sum(p.numel() for p in model.parameters()):,}")
```

### Langkah 2: Self-play untuk Menghasilkan Data

```bash
# Kompilasi engine C++
cd ../cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=CUDA
make -j$(nproc)

# Jalankan self-play
./katago selfplay \
  -model ../python/model_init.pt \
  -output-dir ../python/selfplay_data \
  -config selfplay.cfg \
  -num-games 1000
```

Konfigurasi self-play (selfplay.cfg):

```ini
maxVisits = 600
numSearchThreads = 4

# Pengaturan temperatur (meningkatkan eksplorasi)
chosenMoveTemperature = 1.0
chosenMoveTemperatureEarly = 1.0
chosenMoveTemperatureHalflife = 20

# Noise Dirichlet (meningkatkan keragaman)
rootNoiseEnabled = true
rootDirichletNoiseTotalConcentration = 10.83
rootDirichletNoiseWeight = 0.25
```

**Padanan Animasi**:
- C3 Eksplorasi vs eksploitasi: Parameter temperatur
- E10 Noise Dirichlet: Eksplorasi root node

### Langkah 3: Melatih Neural Network

```python
# train.py
import torch
from torch.utils.data import DataLoader
from model import KataGoModel
from dataset import SelfPlayDataset

# Muat data
dataset = SelfPlayDataset('selfplay_data/')
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# Muat model
model = KataGoModel(config)
model.load_state_dict(torch.load('model_init.pt'))
model = model.cuda()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

# Jadwal learning rate
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=0.00001
)

# Loop pelatihan
for epoch in range(100):
    model.train()
    total_loss = 0

    for batch in dataloader:
        inputs = batch['inputs'].cuda()
        policy_target = batch['policy'].cuda()
        value_target = batch['value'].cuda()
        ownership_target = batch['ownership'].cuda()

        # Forward pass
        policy_pred, value_pred, ownership_pred = model(inputs)

        # Hitung loss
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

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")

    # Simpan checkpoint
    torch.save(model.state_dict(), f'model_epoch{epoch}.pt')
```

**Padanan Animasi**:
- D5 Gradient descent: optimizer.step()
- K2 Momentum: Optimizer Adam
- K4 Decay learning rate: CosineAnnealingLR
- K5 Gradient clipping: clip_grad_norm_

### Langkah 4: Evaluasi dan Iterasi

```bash
# Evaluasi model baru vs model lama
./katago match \
  -model1 model_epoch99.pt \
  -model2 model_init.pt \
  -num-games 100 \
  -output match_results.txt
```

Jika winrate model baru > 55%, ganti model lama dan masuk ke iterasi berikutnya.

---

## Penjelasan Detail Fungsi Loss

### Policy Loss

```python
# Cross entropy loss
policy_loss = -sum(target * log(pred))
```

Tujuan: Membuat distribusi probabilitas yang diprediksi mendekati hasil pencarian MCTS.

**Padanan Animasi**:
- J1 Entropi kebijakan: Cross entropy
- J2 KL divergence: Jarak distribusi

### Value Loss

```python
# Mean squared error
value_loss = (pred - actual_result)^2
```

Tujuan: Memprediksi hasil akhir pertandingan (menang/kalah/seri).

### Ownership Loss

```python
# Prediksi kepemilikan setiap titik
ownership_loss = mean((pred - actual_ownership)^2)
```

Tujuan: Memprediksi kepemilikan akhir setiap posisi.

---

## Teknik Lanjutan

### 1. Augmentasi Data

Memanfaatkan simetri papan:

```python
def augment_data(board, policy, ownership):
    """Augmentasi data untuk 8 transformasi grup D4"""
    augmented = []

    for rotation in range(4):
        for flip in [False, True]:
            # Rotasi dan flip
            aug_board = transform(board, rotation, flip)
            aug_policy = transform(policy, rotation, flip)
            aug_ownership = transform(ownership, rotation, flip)
            augmented.append((aug_board, aug_policy, aug_ownership))

    return augmented
```

**Padanan Animasi**:
- A9 Simetri papan: Grup D4
- L4 Augmentasi data: Pemanfaatan simetri

### 2. Curriculum Learning

Dari sederhana ke kompleks:

```python
# Latih dulu dengan jumlah pencarian lebih sedikit
schedule = [
    (100, 10000),   # 100 visits, 10000 games
    (200, 20000),   # 200 visits, 20000 games
    (400, 50000),   # 400 visits, 50000 games
    (600, 100000),  # 600 visits, 100000 games
]
```

**Padanan Animasi**:
- E12 Kurikulum pelatihan: Curriculum learning

### 3. Pelatihan Mixed Precision

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

### 4. Pelatihan Multi-GPU

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Inisialisasi terdistribusi
dist.init_process_group(backend='nccl')

# Bungkus model
model = DistributedDataParallel(model)
```

---

## Monitoring dan Debugging

### Monitoring TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/training')

# Catat loss
writer.add_scalar('Loss/policy', policy_loss, step)
writer.add_scalar('Loss/value', value_loss, step)
writer.add_scalar('Loss/total', total_loss, step)

# Catat learning rate
writer.add_scalar('LR', scheduler.get_last_lr()[0], step)
```

```bash
tensorboard --logdir runs
```

### Masalah Umum

| Masalah | Kemungkinan Penyebab | Solusi |
|---------|---------------------|--------|
| Loss tidak turun | Learning rate terlalu rendah/tinggi | Sesuaikan learning rate |
| Loss berfluktuasi | Batch size terlalu kecil | Tingkatkan batch size |
| Overfitting | Data tidak cukup | Hasilkan lebih banyak data self-play |
| Kekuatan tidak meningkat | Jumlah pencarian terlalu sedikit | Tingkatkan maxVisits |

**Padanan Animasi**:
- L1 Overfitting: Over-adaptasi
- L2 Regularisasi: weight_decay
- D6 Efek learning rate: Tuning parameter

---

## Saran Eksperimen Skala Kecil

Jika Anda hanya ingin bereksperimen, disarankan:

1. **Gunakan papan 9×9**: Mengurangi komputasi secara signifikan
2. **Gunakan model kecil**: b6c96 cukup untuk eksperimen
3. **Kurangi jumlah pencarian**: 100-200 visits
4. **Fine-tune dari model pre-trained**: Lebih cepat daripada dari nol

```bash
# Konfigurasi papan 9×9
boardSize = 9
maxVisits = 100
```

---

## Bacaan Lanjutan

- [Panduan Source Code](../source-code) — Memahami struktur kode
- [Berkontribusi ke Komunitas Open Source](../contributing) — Bergabung dengan pelatihan terdistribusi
- [Inovasi Kunci KataGo](../../how-it-works/katago-innovations) — Rahasia efisiensi 50x
