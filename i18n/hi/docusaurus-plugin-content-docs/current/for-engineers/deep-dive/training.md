---
sidebar_position: 3
title: KataGo प्रशिक्षण तंत्र विश्लेषण
description: KataGo की सेल्फ-प्ले प्रशिक्षण प्रक्रिया और मूल तकनीकों की गहन समझ
---

# KataGo प्रशिक्षण तंत्र विश्लेषण

यह लेख KataGo के प्रशिक्षण तंत्र की गहन व्याख्या करता है, आपको सेल्फ-प्ले प्रशिक्षण के कार्य सिद्धांत को समझने में मदद करता है।

---

## प्रशिक्षण अवलोकन

### प्रशिक्षण लूप

```
प्रारंभिक मॉडल → सेल्फ-प्ले → डेटा संग्रह → प्रशिक्षण अपडेट → मजबूत मॉडल → दोहराएं
```

**एनिमेशन संबंध**:
- E5 सेल्फ-प्ले ↔ फिक्स्ड पॉइंट कन्वर्जेंस
- E6 खेल शक्ति वक्र ↔ S-कर्व ग्रोथ
- H1 MDP ↔ मार्कोव चेन

### हार्डवेयर आवश्यकताएँ

| मॉडल स्केल | GPU मेमोरी | प्रशिक्षण समय |
|---------|-----------|---------|
| b6c96 | 4 GB | कुछ घंटे |
| b10c128 | 8 GB | 1-2 दिन |
| b18c384 | 16 GB | 1-2 सप्ताह |
| b40c256 | 24 GB+ | कई सप्ताह |

---

## वातावरण सेटअप

### डिपेंडेंसी इंस्टॉल करें

```bash
# Python वातावरण
conda create -n katago python=3.10
conda activate katago

# PyTorch (CUDA संस्करण)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# अन्य डिपेंडेंसी
pip install numpy h5py tqdm tensorboard
```

### प्रशिक्षण कोड प्राप्त करें

```bash
git clone https://github.com/lightvector/KataGo.git
cd KataGo/python
```

---

## प्रशिक्षण कॉन्फ़िगरेशन

### कॉन्फ़िगरेशन फ़ाइल संरचना

```yaml
# configs/train_config.yaml

# मॉडल आर्किटेक्चर
model:
  num_blocks: 10          # रेसिड्यूअल ब्लॉक संख्या
  trunk_channels: 128     # मुख्य ट्रंक चैनल
  policy_channels: 32     # Policy हेड चैनल
  value_channels: 32      # Value हेड चैनल

# प्रशिक्षण पैरामीटर
training:
  batch_size: 256
  learning_rate: 0.001
  lr_schedule: "cosine"
  weight_decay: 0.0001
  epochs: 100

# सेल्फ-प्ले पैरामीटर
selfplay:
  num_games_per_iteration: 1000
  max_visits: 600
  temperature: 1.0
  temperature_drop_move: 20

# डेटा कॉन्फ़िगरेशन
data:
  max_history_games: 500000
  shuffle_buffer_size: 100000
```

### मॉडल स्केल तुलना

| नाम | num_blocks | trunk_channels | पैरामीटर |
|------|-----------|----------------|--------|
| b6c96 | 6 | 96 | ~1M |
| b10c128 | 10 | 128 | ~3M |
| b18c384 | 18 | 384 | ~20M |
| b40c256 | 40 | 256 | ~45M |

**एनिमेशन संबंध**:
- F2 नेटवर्क आकार vs खेल शक्ति: क्षमता स्केलिंग
- F6 न्यूरल स्केलिंग नियम: डबल लॉगरिदमिक संबंध

---

## प्रशिक्षण प्रक्रिया

### चरण 1: मॉडल इनिशियलाइज़ करें

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
print(f"मॉडल पैरामीटर: {sum(p.numel() for p in model.parameters()):,}")
```

### चरण 2: सेल्फ-प्ले से डेटा उत्पन्न करें

```bash
# C++ इंजन कंपाइल करें
cd ../cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=CUDA
make -j$(nproc)

# सेल्फ-प्ले निष्पादित करें
./katago selfplay \
  -model ../python/model_init.pt \
  -output-dir ../python/selfplay_data \
  -config selfplay.cfg \
  -num-games 1000
```

सेल्फ-प्ले कॉन्फ़िगरेशन (selfplay.cfg):

```ini
maxVisits = 600
numSearchThreads = 4

# तापमान सेटिंग (अन्वेषण बढ़ाएं)
chosenMoveTemperature = 1.0
chosenMoveTemperatureEarly = 1.0
chosenMoveTemperatureHalflife = 20

# Dirichlet नॉइज़ (विविधता बढ़ाएं)
rootNoiseEnabled = true
rootDirichletNoiseTotalConcentration = 10.83
rootDirichletNoiseWeight = 0.25
```

**एनिमेशन संबंध**:
- C3 अन्वेषण vs उपयोग: तापमान पैरामीटर
- E10 Dirichlet नॉइज़: रूट नोड अन्वेषण

### चरण 3: न्यूरल नेटवर्क प्रशिक्षित करें

```python
# train.py
import torch
from torch.utils.data import DataLoader
from model import KataGoModel
from dataset import SelfPlayDataset

# डेटा लोड करें
dataset = SelfPlayDataset('selfplay_data/')
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# मॉडल लोड करें
model = KataGoModel(config)
model.load_state_dict(torch.load('model_init.pt'))
model = model.cuda()

# ऑप्टिमाइज़र
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

# लर्निंग रेट शेड्यूल
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=0.00001
)

# प्रशिक्षण लूप
for epoch in range(100):
    model.train()
    total_loss = 0

    for batch in dataloader:
        inputs = batch['inputs'].cuda()
        policy_target = batch['policy'].cuda()
        value_target = batch['value'].cuda()
        ownership_target = batch['ownership'].cuda()

        # फॉरवर्ड प्रोपेगेशन
        policy_pred, value_pred, ownership_pred = model(inputs)

        # लॉस गणना करें
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

        # बैकप्रोपेगेशन
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")

    # चेकपॉइंट सेव करें
    torch.save(model.state_dict(), f'model_epoch{epoch}.pt')
```

**एनिमेशन संबंध**:
- D5 ग्रेडिएंट डिसेंट: optimizer.step()
- K2 मोमेंटम: Adam ऑप्टिमाइज़र
- K4 लर्निंग रेट डिके: CosineAnnealingLR
- K5 ग्रेडिएंट क्लिपिंग: clip_grad_norm_

### चरण 4: मूल्यांकन और इटरेशन

```bash
# नए मॉडल vs पुराने मॉडल का मूल्यांकन
./katago match \
  -model1 model_epoch99.pt \
  -model2 model_init.pt \
  -num-games 100 \
  -output match_results.txt
```

यदि नए मॉडल की जीत दर > 55%, तो पुराने मॉडल को बदलें, अगले इटरेशन में प्रवेश करें।

---

## लॉस फंक्शन विस्तृत व्याख्या

### Policy Loss

```python
# क्रॉस-एंट्रॉपी लॉस
policy_loss = -sum(target * log(pred))
```

लक्ष्य: पूर्वानुमानित प्रायिकता वितरण को MCTS सर्च परिणाम के करीब लाना।

**एनिमेशन संबंध**:
- J1 स्ट्रैटेजी एंट्रॉपी: क्रॉस-एंट्रॉपी
- J2 KL डाइवर्जेंस: वितरण दूरी

### Value Loss

```python
# मीन स्क्वेयर्ड एरर
value_loss = (pred - actual_result)^2
```

लक्ष्य: खेल के अंतिम परिणाम (जीत/हार/ड्रॉ) का पूर्वानुमान।

### Ownership Loss

```python
# प्रत्येक बिंदु का स्वामित्व पूर्वानुमान
ownership_loss = mean((pred - actual_ownership)^2)
```

लक्ष्य: प्रत्येक स्थान के अंतिम स्वामित्व का पूर्वानुमान।

---

## उन्नत तकनीकें

### 1. डेटा ऑगमेंटेशन

बोर्ड की सममिति का उपयोग करें:

```python
def augment_data(board, policy, ownership):
    """D4 समूह के 8 परिवर्तनों के लिए डेटा ऑगमेंटेशन"""
    augmented = []

    for rotation in range(4):
        for flip in [False, True]:
            # रोटेशन और फ्लिप
            aug_board = transform(board, rotation, flip)
            aug_policy = transform(policy, rotation, flip)
            aug_ownership = transform(ownership, rotation, flip)
            augmented.append((aug_board, aug_policy, aug_ownership))

    return augmented
```

**एनिमेशन संबंध**:
- A9 बोर्ड सममिति: D4 समूह
- L4 डेटा ऑगमेंटेशन: सममिति उपयोग

### 2. करिकुलम लर्निंग

सरल से जटिल की ओर:

```python
# पहले कम सर्च विज़िट्स से प्रशिक्षित करें
schedule = [
    (100, 10000),   # 100 visits, 10000 games
    (200, 20000),   # 200 visits, 20000 games
    (400, 50000),   # 400 visits, 50000 games
    (600, 100000),  # 600 visits, 100000 games
]
```

**एनिमेशन संबंध**:
- E12 प्रशिक्षण करिकुलम: करिकुलम लर्निंग

### 3. मिक्स्ड प्रिसिजन ट्रेनिंग

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

### 4. मल्टी GPU ट्रेनिंग

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# डिस्ट्रिब्यूटेड इनिशियलाइज़ करें
dist.init_process_group(backend='nccl')

# मॉडल रैप करें
model = DistributedDataParallel(model)
```

---

## मॉनिटरिंग और डीबगिंग

### TensorBoard मॉनिटरिंग

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/training')

# लॉस रिकॉर्ड करें
writer.add_scalar('Loss/policy', policy_loss, step)
writer.add_scalar('Loss/value', value_loss, step)
writer.add_scalar('Loss/total', total_loss, step)

# लर्निंग रेट रिकॉर्ड करें
writer.add_scalar('LR', scheduler.get_last_lr()[0], step)
```

```bash
tensorboard --logdir runs
```

### आम समस्याएं

| समस्या | संभावित कारण | समाधान |
|------|---------|---------|
| लॉस नहीं घटता | लर्निंग रेट बहुत कम/अधिक | लर्निंग रेट समायोजित करें |
| लॉस ऑसिलेट करता है | बैच साइज़ बहुत छोटा | बैच साइज़ बढ़ाएं |
| ओवरफिटिंग | डेटा अपर्याप्त | अधिक सेल्फ-प्ले डेटा उत्पन्न करें |
| खेल शक्ति नहीं बढ़ती | सर्च विज़िट्स बहुत कम | maxVisits बढ़ाएं |

**एनिमेशन संबंध**:
- L1 ओवरफिटिंग: अति-अनुकूलन
- L2 रेगुलराइज़ेशन: weight_decay
- D6 लर्निंग रेट प्रभाव: ट्यूनिंग

---

## छोटे स्केल प्रयोग सुझाव

यदि आप केवल प्रयोग करना चाहते हैं, तो सुझाव:

1. **9×9 बोर्ड का उपयोग करें**: गणना काफी कम
2. **छोटे मॉडल का उपयोग करें**: b6c96 प्रयोग के लिए पर्याप्त
3. **सर्च विज़िट्स कम करें**: 100-200 visits
4. **प्री-ट्रेंड मॉडल फाइन-ट्यून करें**: शून्य से शुरू करने से तेज

```bash
# 9×9 बोर्ड कॉन्फ़िगरेशन
boardSize = 9
maxVisits = 100
```

---

## आगे पढ़ें

- [स्रोत कोड गाइड](../source-code) — कोड संरचना समझें
- [ओपन सोर्स समुदाय में भागीदारी](../contributing) — वितरित प्रशिक्षण में शामिल हों
- [KataGo की मुख्य नवाचार](../../how-it-works/katago-innovations) — 50 गुना दक्षता का रहस्य
