---
sidebar_position: 3
title: Análise do Mecanismo de Treinamento do KataGo
description: Compreensão profunda do fluxo de treinamento por auto-jogo do KataGo e suas tecnologias principais
---

# Análise do Mecanismo de Treinamento do KataGo

Este artigo analisa em profundidade o mecanismo de treinamento do KataGo, ajudando você a entender o princípio de funcionamento do treinamento por auto-jogo.

---

## Visão Geral do Treinamento

### Ciclo de Treinamento

```
Modelo inicial → Auto-jogo → Coleta de dados → Atualização de treinamento → Modelo mais forte → Repetir
```

**Correspondência com animações**:
- E5 Auto-jogo ↔ Convergência de ponto fixo
- E6 Curva de força ↔ Crescimento em curva S
- H1 MDP ↔ Cadeia de Markov

### Requisitos de Hardware

| Escala do Modelo | Memória GPU | Tempo de Treinamento |
|------------------|-------------|---------------------|
| b6c96 | 4 GB | Algumas horas |
| b10c128 | 8 GB | 1-2 dias |
| b18c384 | 16 GB | 1-2 semanas |
| b40c256 | 24 GB+ | Várias semanas |

---

## Configuração do Ambiente

### Instalação de Dependências

```bash
# Ambiente Python
conda create -n katago python=3.10
conda activate katago

# PyTorch (versão CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Outras dependências
pip install numpy h5py tqdm tensorboard
```

### Obter Código de Treinamento

```bash
git clone https://github.com/lightvector/KataGo.git
cd KataGo/python
```

---

## Configuração de Treinamento

### Estrutura do Arquivo de Configuração

```yaml
# configs/train_config.yaml

# Arquitetura do modelo
model:
  num_blocks: 10          # Número de blocos residuais
  trunk_channels: 128     # Canais do trunk
  policy_channels: 32     # Canais do Policy head
  value_channels: 32      # Canais do Value head

# Parâmetros de treinamento
training:
  batch_size: 256
  learning_rate: 0.001
  lr_schedule: "cosine"
  weight_decay: 0.0001
  epochs: 100

# Parâmetros de auto-jogo
selfplay:
  num_games_per_iteration: 1000
  max_visits: 600
  temperature: 1.0
  temperature_drop_move: 20

# Configuração de dados
data:
  max_history_games: 500000
  shuffle_buffer_size: 100000
```

### Comparação de Escalas de Modelo

| Nome | num_blocks | trunk_channels | Parâmetros |
|------|-----------|----------------|------------|
| b6c96 | 6 | 96 | ~1M |
| b10c128 | 10 | 128 | ~3M |
| b18c384 | 18 | 384 | ~20M |
| b40c256 | 40 | 256 | ~45M |

**Correspondência com animações**:
- F2 Tamanho da rede vs Força de jogo: Escalabilidade de capacidade
- F6 Leis de escalabilidade neural: Relação log-log

---

## Fluxo de Treinamento

### Passo 1: Inicializar Modelo

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
print(f"Parâmetros do modelo: {sum(p.numel() for p in model.parameters()):,}")
```

### Passo 2: Auto-jogo para Gerar Dados

```bash
# Compilar engine C++
cd ../cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=CUDA
make -j$(nproc)

# Executar auto-jogo
./katago selfplay \
  -model ../python/model_init.pt \
  -output-dir ../python/selfplay_data \
  -config selfplay.cfg \
  -num-games 1000
```

Configuração de auto-jogo (selfplay.cfg):

```ini
maxVisits = 600
numSearchThreads = 4

# Configuração de temperatura (aumenta exploração)
chosenMoveTemperature = 1.0
chosenMoveTemperatureEarly = 1.0
chosenMoveTemperatureHalflife = 20

# Ruído de Dirichlet (aumenta diversidade)
rootNoiseEnabled = true
rootDirichletNoiseTotalConcentration = 10.83
rootDirichletNoiseWeight = 0.25
```

**Correspondência com animações**:
- C3 Exploração vs Aproveitamento: Parâmetro de temperatura
- E10 Ruído de Dirichlet: Exploração no nó raiz

### Passo 3: Treinar a Rede Neural

```python
# train.py
import torch
from torch.utils.data import DataLoader
from model import KataGoModel
from dataset import SelfPlayDataset

# Carregar dados
dataset = SelfPlayDataset('selfplay_data/')
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# Carregar modelo
model = KataGoModel(config)
model.load_state_dict(torch.load('model_init.pt'))
model = model.cuda()

# Otimizador
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

# Agendador de taxa de aprendizado
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=0.00001
)

# Loop de treinamento
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

        # Calcular perda
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

    # Salvar checkpoint
    torch.save(model.state_dict(), f'model_epoch{epoch}.pt')
```

**Correspondência com animações**:
- D5 Descida de gradiente: optimizer.step()
- K2 Momentum: Otimizador Adam
- K4 Decaimento da taxa de aprendizado: CosineAnnealingLR
- K5 Clipping de gradiente: clip_grad_norm_

### Passo 4: Avaliar e Iterar

```bash
# Avaliar novo modelo vs modelo antigo
./katago match \
  -model1 model_epoch99.pt \
  -model2 model_init.pt \
  -num-games 100 \
  -output match_results.txt
```

Se a taxa de vitória do novo modelo > 55%, ele substitui o modelo antigo e entra na próxima iteração.

---

## Funções de Perda Detalhadas

### Policy Loss

```python
# Perda de entropia cruzada
policy_loss = -sum(target * log(pred))
```

Objetivo: Fazer a distribuição de probabilidade prevista se aproximar do resultado da busca MCTS.

**Correspondência com animações**:
- J1 Entropia de política: Entropia cruzada
- J2 Divergência KL: Distância entre distribuições

### Value Loss

```python
# Erro quadrático médio
value_loss = (pred - actual_result)^2
```

Objetivo: Prever o resultado final do jogo (vitória/derrota/empate).

### Ownership Loss

```python
# Previsão de pertencimento de cada ponto
ownership_loss = mean((pred - actual_ownership)^2)
```

Objetivo: Prever o pertencimento final de cada posição.

---

## Técnicas Avançadas

### 1. Data Augmentation

Aproveitando a simetria do tabuleiro:

```python
def augment_data(board, policy, ownership):
    """Aumentação de dados para as 8 transformações do grupo D4"""
    augmented = []

    for rotation in range(4):
        for flip in [False, True]:
            # Rotação e espelhamento
            aug_board = transform(board, rotation, flip)
            aug_policy = transform(policy, rotation, flip)
            aug_ownership = transform(ownership, rotation, flip)
            augmented.append((aug_board, aug_policy, aug_ownership))

    return augmented
```

**Correspondência com animações**:
- A9 Simetria do tabuleiro: Grupo D4
- L4 Data augmentation: Aproveitamento de simetria

### 2. Curriculum Learning

Do simples ao complexo:

```python
# Primeiro treina com menos simulações
schedule = [
    (100, 10000),   # 100 visitas, 10000 jogos
    (200, 20000),   # 200 visitas, 20000 jogos
    (400, 50000),   # 400 visitas, 50000 jogos
    (600, 100000),  # 600 visitas, 100000 jogos
]
```

**Correspondência com animações**:
- E12 Currículo de treinamento: Curriculum learning

### 3. Treinamento de Precisão Mista

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

### 4. Treinamento Multi-GPU

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Inicializar distribuído
dist.init_process_group(backend='nccl')

# Encapsular modelo
model = DistributedDataParallel(model)
```

---

## Monitoramento e Depuração

### Monitoramento com TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/training')

# Registrar perdas
writer.add_scalar('Loss/policy', policy_loss, step)
writer.add_scalar('Loss/value', value_loss, step)
writer.add_scalar('Loss/total', total_loss, step)

# Registrar taxa de aprendizado
writer.add_scalar('LR', scheduler.get_last_lr()[0], step)
```

```bash
tensorboard --logdir runs
```

### Problemas Comuns

| Problema | Causa Possível | Solução |
|----------|----------------|---------|
| Perda não diminui | Taxa de aprendizado muito baixa/alta | Ajustar taxa de aprendizado |
| Perda oscila | Tamanho de lote muito pequeno | Aumentar tamanho do lote |
| Overfitting | Dados insuficientes | Gerar mais dados de auto-jogo |
| Força não aumenta | Poucas simulações | Aumentar maxVisits |

**Correspondência com animações**:
- L1 Overfitting: Super-adaptação
- L2 Regularização: weight_decay
- D6 Efeito da taxa de aprendizado: Ajuste de parâmetros

---

## Sugestões para Experimentos de Pequena Escala

Se você quer apenas experimentar, recomendamos:

1. **Usar tabuleiro 9×9**: Reduz drasticamente a computação
2. **Usar modelo pequeno**: b6c96 é suficiente para experimentos
3. **Reduzir número de simulações**: 100-200 visitas
4. **Fazer fine-tuning de modelo pré-treinado**: Mais rápido que treinar do zero

```bash
# Configuração para tabuleiro 9×9
boardSize = 9
maxVisits = 100
```

---

## Leitura Adicional

- [Guia do Código-fonte](../source-code) — Entender a estrutura do código
- [Participando da Comunidade Open Source](../contributing) — Participar do treinamento distribuído
- [Inovações do KataGo](../../how-it-works/katago-innovations) — O segredo da eficiência 50x
