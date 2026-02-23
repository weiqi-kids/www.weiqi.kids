---
sidebar_position: 3
title: تحليل آلية تدريب KataGo
description: فهم معمق لعملية تدريب اللعب الذاتي في KataGo والتقنيات الأساسية
---

# تحليل آلية تدريب KataGo

يقدم هذا المقال تحليلاً معمقاً لآلية تدريب KataGo، لمساعدتك على فهم مبادئ عمل تدريب اللعب الذاتي.

---

## نظرة عامة على التدريب

### دورة التدريب

```
نموذج أولي → لعب ذاتي → جمع البيانات → تحديث التدريب → نموذج أقوى → تكرار
```

**المفاهيم المقابلة**:
- اللعب الذاتي ↔ تقارب النقطة الثابتة
- منحنى قوة اللعب ↔ نمو المنحنى S
- MDP ↔ سلسلة ماركوف

### متطلبات الأجهزة

| حجم النموذج | ذاكرة GPU | وقت التدريب |
|------------|----------|-------------|
| b6c96 | 4 GB | عدة ساعات |
| b10c128 | 8 GB | 1-2 يوم |
| b18c384 | 16 GB | 1-2 أسبوع |
| b40c256 | 24 GB+ | عدة أسابيع |

---

## إعداد البيئة

### تثبيت التبعيات

```bash
# بيئة Python
conda create -n katago python=3.10
conda activate katago

# PyTorch (إصدار CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# تبعيات أخرى
pip install numpy h5py tqdm tensorboard
```

### الحصول على كود التدريب

```bash
git clone https://github.com/lightvector/KataGo.git
cd KataGo/python
```

---

## إعدادات التدريب

### هيكل ملف الإعدادات

```yaml
# configs/train_config.yaml

# بنية النموذج
model:
  num_blocks: 10          # عدد الكتل المتبقية
  trunk_channels: 128     # عدد قنوات الجذع
  policy_channels: 32     # عدد قنوات رأس السياسة
  value_channels: 32      # عدد قنوات رأس القيمة

# معلمات التدريب
training:
  batch_size: 256
  learning_rate: 0.001
  lr_schedule: "cosine"
  weight_decay: 0.0001
  epochs: 100

# معلمات اللعب الذاتي
selfplay:
  num_games_per_iteration: 1000
  max_visits: 600
  temperature: 1.0
  temperature_drop_move: 20

# إعدادات البيانات
data:
  max_history_games: 500000
  shuffle_buffer_size: 100000
```

### جدول أحجام النماذج

| الاسم | num_blocks | trunk_channels | عدد المعلمات |
|-------|-----------|----------------|--------------|
| b6c96 | 6 | 96 | ~1M |
| b10c128 | 10 | 128 | ~3M |
| b18c384 | 18 | 384 | ~20M |
| b40c256 | 40 | 256 | ~45M |

**المفاهيم المقابلة**:
- حجم الشبكة مقابل قوة اللعب: قياس السعة
- قوانين القياس العصبية: العلاقة اللوغاريتمية المزدوجة

---

## عملية التدريب

### الخطوة 1: تهيئة النموذج

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
print(f"عدد معلمات النموذج: {sum(p.numel() for p in model.parameters()):,}")
```

### الخطوة 2: توليد بيانات اللعب الذاتي

```bash
# بناء محرك C++
cd ../cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=CUDA
make -j$(nproc)

# تنفيذ اللعب الذاتي
./katago selfplay \
  -model ../python/model_init.pt \
  -output-dir ../python/selfplay_data \
  -config selfplay.cfg \
  -num-games 1000
```

إعدادات اللعب الذاتي (selfplay.cfg):

```ini
maxVisits = 600
numSearchThreads = 4

# إعدادات الحرارة (زيادة الاستكشاف)
chosenMoveTemperature = 1.0
chosenMoveTemperatureEarly = 1.0
chosenMoveTemperatureHalflife = 20

# ضوضاء Dirichlet (زيادة التنوع)
rootNoiseEnabled = true
rootDirichletNoiseTotalConcentration = 10.83
rootDirichletNoiseWeight = 0.25
```

**المفاهيم المقابلة**:
- الاستكشاف مقابل الاستغلال: معلمة الحرارة
- ضوضاء Dirichlet: استكشاف عقدة الجذر

### الخطوة 3: تدريب الشبكة العصبية

```python
# train.py
import torch
from torch.utils.data import DataLoader
from model import KataGoModel
from dataset import SelfPlayDataset

# تحميل البيانات
dataset = SelfPlayDataset('selfplay_data/')
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# تحميل النموذج
model = KataGoModel(config)
model.load_state_dict(torch.load('model_init.pt'))
model = model.cuda()

# المحسّن
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

# جدولة معدل التعلم
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=0.00001
)

# دورة التدريب
for epoch in range(100):
    model.train()
    total_loss = 0

    for batch in dataloader:
        inputs = batch['inputs'].cuda()
        policy_target = batch['policy'].cuda()
        value_target = batch['value'].cuda()
        ownership_target = batch['ownership'].cuda()

        # التمرير الأمامي
        policy_pred, value_pred, ownership_pred = model(inputs)

        # حساب الخسارة
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

        # التمرير العكسي
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")

    # حفظ نقطة التحقق
    torch.save(model.state_dict(), f'model_epoch{epoch}.pt')
```

**المفاهيم المقابلة**:
- نزول التدرج: optimizer.step()
- الزخم: محسّن Adam
- تناقص معدل التعلم: CosineAnnealingLR
- قص التدرج: clip_grad_norm_

### الخطوة 4: التقييم والتكرار

```bash
# تقييم النموذج الجديد مقابل القديم
./katago match \
  -model1 model_epoch99.pt \
  -model2 model_init.pt \
  -num-games 100 \
  -output match_results.txt
```

إذا كان معدل فوز النموذج الجديد > 55%، فإنه يحل محل النموذج القديم ويدخل في الجولة التالية من التكرار.

---

## شرح تفصيلي لدوال الخسارة

### خسارة السياسة (Policy Loss)

```python
# خسارة الانتروبيا المتقاطعة
policy_loss = -sum(target * log(pred))
```

الهدف: جعل التوزيع الاحتمالي المتوقع قريباً من نتائج بحث MCTS.

**المفاهيم المقابلة**:
- انتروبيا السياسة: الانتروبيا المتقاطعة
- تباعد KL: مسافة التوزيع

### خسارة القيمة (Value Loss)

```python
# متوسط الخطأ التربيعي
value_loss = (pred - actual_result)^2
```

الهدف: التنبؤ بالنتيجة النهائية للمباراة (فوز/خسارة/تعادل).

### خسارة الملكية (Ownership Loss)

```python
# التنبؤ بملكية كل نقطة
ownership_loss = mean((pred - actual_ownership)^2)
```

الهدف: التنبؤ بالملكية النهائية لكل موقع.

---

## التقنيات المتقدمة

### 1. زيادة البيانات

استغلال تناظر اللوحة:

```python
def augment_data(board, policy, ownership):
    """زيادة البيانات لـ 8 تحولات من مجموعة D4"""
    augmented = []

    for rotation in range(4):
        for flip in [False, True]:
            # التدوير والانعكاس
            aug_board = transform(board, rotation, flip)
            aug_policy = transform(policy, rotation, flip)
            aug_ownership = transform(ownership, rotation, flip)
            augmented.append((aug_board, aug_policy, aug_ownership))

    return augmented
```

**المفاهيم المقابلة**:
- تناظر اللوحة: مجموعة D4
- زيادة البيانات: استغلال التناظر

### 2. التعلم المتدرج

من البسيط إلى المعقد:

```python
# التدريب أولاً بعدد زيارات أقل
schedule = [
    (100, 10000),   # 100 زيارة، 10000 مباراة
    (200, 20000),   # 200 زيارة، 20000 مباراة
    (400, 50000),   # 400 زيارة، 50000 مباراة
    (600, 100000),  # 600 زيارة، 100000 مباراة
]
```

**المفاهيم المقابلة**:
- منهج التدريب: التعلم المتدرج

### 3. التدريب بالدقة المختلطة

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

### 4. التدريب متعدد GPU

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# تهيئة التوزيع
dist.init_process_group(backend='nccl')

# تغليف النموذج
model = DistributedDataParallel(model)
```

---

## المراقبة والتصحيح

### مراقبة TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/training')

# تسجيل الخسارة
writer.add_scalar('Loss/policy', policy_loss, step)
writer.add_scalar('Loss/value', value_loss, step)
writer.add_scalar('Loss/total', total_loss, step)

# تسجيل معدل التعلم
writer.add_scalar('LR', scheduler.get_last_lr()[0], step)
```

```bash
tensorboard --logdir runs
```

### المشاكل الشائعة

| المشكلة | السبب المحتمل | الحل |
|--------|--------------|------|
| الخسارة لا تنخفض | معدل التعلم منخفض/مرتفع جداً | ضبط معدل التعلم |
| الخسارة متذبذبة | حجم الدفعة صغير جداً | زيادة حجم الدفعة |
| فرط التخصيص | بيانات غير كافية | توليد المزيد من بيانات اللعب الذاتي |
| قوة اللعب لا تتحسن | عدد الزيارات قليل جداً | زيادة maxVisits |

**المفاهيم المقابلة**:
- فرط التخصيص: التكيف المفرط
- التنظيم: weight_decay
- تأثير معدل التعلم: ضبط المعلمات

---

## اقتراحات للتجارب الصغيرة

إذا كنت تريد فقط التجربة، يُنصح بـ:

1. **استخدام لوحة 9×9**: تقليل كبير في الحسابات
2. **استخدام نموذج صغير**: b6c96 كافٍ للتجربة
3. **تقليل عدد الزيارات**: 100-200 زيارة
4. **ضبط نموذج مدرب مسبقاً**: أسرع من البدء من الصفر

```bash
# إعدادات لوحة 9×9
boardSize = 9
maxVisits = 100
```

---

## قراءات إضافية

- [دليل قراءة الكود المصدري](../source-code) — فهم هيكل الكود
- [المشاركة في مجتمع المصادر المفتوحة](../contributing) — الانضمام للتدريب الموزع
- [ابتكارات KataGo الرئيسية](../../how-it-works/katago-innovations) — سر الكفاءة 50 ضعفاً
