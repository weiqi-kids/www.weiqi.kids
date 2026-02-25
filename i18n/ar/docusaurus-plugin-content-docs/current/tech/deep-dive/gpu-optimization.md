---
sidebar_position: 6
title: واجهات GPU والتحسين
description: مقارنة واجهات CUDA وOpenCL وMetal في KataGo ودليل ضبط الأداء
---

# واجهات GPU والتحسين

يقدم هذا المقال واجهات GPU المختلفة التي يدعمها KataGo، والفروقات في الأداء، وكيفية ضبطها للحصول على أفضل أداء.

---

## نظرة عامة على الواجهات

يدعم KataGo أربع واجهات حسابية:

| الواجهة | دعم الأجهزة | الأداء | صعوبة التثبيت |
|---------|------------|--------|---------------|
| **CUDA** | NVIDIA GPU | الأفضل | متوسطة |
| **OpenCL** | NVIDIA/AMD/Intel GPU | جيد | سهلة |
| **Metal** | Apple Silicon | جيد | سهلة |
| **Eigen** | CPU فقط | أبطأ | الأسهل |

---

## واجهة CUDA

### حالات الاستخدام

- NVIDIA GPU (GTX 10 series وأحدث)
- الحاجة لأعلى أداء
- وجود بيئة تطوير CUDA

### متطلبات التثبيت

```bash
# التحقق من إصدار CUDA
nvcc --version

# التحقق من cuDNN
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR
```

| المكون | الإصدار المقترح |
|--------|-----------------|
| CUDA | 11.x أو 12.x |
| cuDNN | 8.x |
| برنامج التشغيل | 470+ |

### البناء

```bash
cd KataGo/cpp
mkdir build && cd build

cmake .. -DUSE_BACKEND=CUDA \
         -DCUDNN_INCLUDE_DIR=/usr/local/cuda/include \
         -DCUDNN_LIBRARY=/usr/local/cuda/lib64/libcudnn.so

make -j$(nproc)
```

### مميزات الأداء

- **Tensor Cores**: دعم تسريع FP16 (سلسلة RTX)
- **الاستدلال الدفعي**: أفضل استخدام لـ GPU
- **إدارة الذاكرة**: تحكم دقيق في VRAM

---

## واجهة OpenCL

### حالات الاستخدام

- AMD GPU
- Intel integrated graphics
- NVIDIA GPU (بدون بيئة CUDA)
- النشر عبر المنصات

### متطلبات التثبيت

```bash
# Linux - تثبيت حزمة تطوير OpenCL
sudo apt install ocl-icd-opencl-dev

# التحقق من أجهزة OpenCL المتاحة
clinfo
```

### البناء

```bash
cmake .. -DUSE_BACKEND=OPENCL
make -j$(nproc)
```

### اختيار برنامج التشغيل

| نوع GPU | برنامج التشغيل المقترح |
|---------|------------------------|
| AMD | ROCm أو AMDGPU-PRO |
| Intel | intel-opencl-icd |
| NVIDIA | nvidia-opencl-icd |

### ضبط الأداء

```ini
# config.cfg
openclDeviceToUse = 0          # رقم GPU
openclUseFP16 = auto           # نصف الدقة (عند الدعم)
openclUseFP16Storage = true    # تخزين FP16
```

---

## واجهة Metal

### حالات الاستخدام

- Apple Silicon (M1/M2/M3)
- نظام macOS

### البناء

```bash
cmake .. -DUSE_BACKEND=METAL
make -j$(sysctl -n hw.ncpu)
```

### تحسين Apple Silicon

بنية الذاكرة الموحدة في Apple Silicon لها مزايا خاصة:

```ini
# إعدادات Apple Silicon المقترحة
numNNServerThreadsPerModel = 1
nnMaxBatchSize = 16
numSearchThreads = 4
```

### مقارنة الأداء

| الشريحة | الأداء النسبي |
|---------|---------------|
| M1 | ~RTX 2060 |
| M1 Pro/Max | ~RTX 3060 |
| M2 | ~RTX 2070 |
| M3 Pro/Max | ~RTX 3070 |

---

## واجهة Eigen (CPU فقط)

### حالات الاستخدام

- بيئة بدون GPU
- الاختبار السريع
- الاستخدام الخفيف

### البناء

```bash
sudo apt install libeigen3-dev
cmake .. -DUSE_BACKEND=EIGEN
make -j$(nproc)
```

### توقعات الأداء

```
CPU نواة واحدة: ~10-30 playouts/sec
CPU متعدد النواة: ~50-150 playouts/sec
GPU (متوسط): ~1000-3000 playouts/sec
```

---

## معلمات ضبط الأداء

### المعلمات الأساسية

```ini
# config.cfg

# === إعدادات الشبكة العصبية ===
# رقم GPU (لأجهزة GPU متعددة)
nnDeviceIdxs = 0

# عدد خيوط الاستدلال لكل نموذج
numNNServerThreadsPerModel = 2

# الحد الأقصى لحجم الدفعة
nnMaxBatchSize = 16

# حجم الذاكرة المؤقتة (2^N مواقع)
nnCacheSizePowerOfTwo = 20

# === إعدادات البحث ===
# عدد خيوط البحث
numSearchThreads = 8

# الحد الأقصى للزيارات لكل حركة
maxVisits = 800
```

### دليل ضبط المعلمات

#### nnMaxBatchSize

```
صغير جداً: استخدام GPU منخفض، تأخر استدلال عالي
كبير جداً: VRAM غير كافي، وقت انتظار طويل

القيم المقترحة:
- 4GB VRAM: 8-12
- 8GB VRAM: 16-24
- 16GB+ VRAM: 32-64
```

#### numSearchThreads

```
قليل جداً: لا يمكن إشباع GPU
كثير جداً: عنق زجاجة CPU، ضغط الذاكرة

القيم المقترحة:
- 1-2 ضعف عدد نوى CPU
- قريب من nnMaxBatchSize
```

#### numNNServerThreadsPerModel

```
CUDA: 1-2
OpenCL: 1-2
Eigen: عدد نوى CPU
```

### ضبط الذاكرة

```ini
# تقليل استخدام VRAM
nnMaxBatchSize = 8
nnCacheSizePowerOfTwo = 18

# زيادة استخدام VRAM (تحسين الأداء)
nnMaxBatchSize = 32
nnCacheSizePowerOfTwo = 22
```

---

## إعداد GPU متعدد

### جهاز واحد بـ GPU متعدد

```ini
# استخدام GPU 0 و GPU 1
nnDeviceIdxs = 0,1

# عدد الخيوط لكل GPU
numNNServerThreadsPerModel = 2
```

### توازن الحمل

```ini
# توزيع الأوزان حسب أداء GPU
# GPU 0 أقوى، يحصل على عمل أكثر
nnDeviceIdxs = 0,0,1
```

---

## الاختبار المعياري

### تشغيل الاختبار المعياري

```bash
katago benchmark -model model.bin.gz -config config.cfg
```

### تفسير المخرجات

```
GPU 0: NVIDIA GeForce RTX 3080
Threads: 8, Batch Size: 16

Benchmark results:
- Neural net evals/second: 2847.3
- Playouts/second: 4521.8
- Time per move (1000 visits): 0.221 sec

Memory usage:
- Peak GPU memory: 2.1 GB
- Peak system memory: 1.3 GB
```

### بيانات الأداء الشائعة

| GPU | النموذج | Playouts/ثانية |
|-----|---------|----------------|
| RTX 3060 | b18c384 | ~2500 |
| RTX 3080 | b18c384 | ~4500 |
| RTX 4090 | b18c384 | ~8000 |
| M1 Pro | b18c384 | ~1500 |
| M2 Max | b18c384 | ~2200 |

---

## تسريع TensorRT

### حالات الاستخدام

- NVIDIA GPU
- السعي لأقصى أداء
- قبول وقت تهيئة أطول

### طريقة التفعيل

```bash
# التفعيل عند البناء
cmake .. -DUSE_BACKEND=CUDA -DUSE_TENSORRT=ON

# أو استخدام الإصدار المبني مسبقاً
katago-tensorrt
```

### تحسين الأداء

```
CUDA القياسي: 100%
TensorRT FP32: +20-30%
TensorRT FP16: +50-80% (سلسلة RTX)
TensorRT INT8: +100-150% (يتطلب معايرة)
```

### ملاحظات

- التشغيل الأول يتطلب بناء محرك TensorRT (عدة دقائق)
- GPU مختلف يتطلب إعادة البناء
- FP16/INT8 قد يقلل الدقة قليلاً

---

## المشاكل الشائعة

### GPU غير مكتشف

```bash
# التحقق من حالة GPU
nvidia-smi  # NVIDIA
rocm-smi    # AMD
clinfo      # OpenCL

# KataGo يعرض GPU المتاحة
katago gpuinfo
```

### VRAM غير كافية

```ini
# استخدام نموذج أصغر
# b18c384 → b10c128

# تقليل حجم الدفعة
nnMaxBatchSize = 4

# تقليل الذاكرة المؤقتة
nnCacheSizePowerOfTwo = 16
```

### الأداء أقل من المتوقع

1. تأكد من استخدام الواجهة الصحيحة (CUDA > OpenCL > Eigen)
2. تحقق من أن `numSearchThreads` كافٍ
3. تأكد من أن GPU غير مشغول ببرامج أخرى
4. استخدم أمر `benchmark` للتأكد من الأداء

---

## قائمة فحص تحسين الأداء

- [ ] اختيار الواجهة الصحيحة (CUDA/OpenCL/Metal)
- [ ] تثبيت أحدث برامج تشغيل GPU
- [ ] ضبط `nnMaxBatchSize` ليتناسب مع VRAM
- [ ] ضبط `numSearchThreads` ليتناسب مع CPU
- [ ] تشغيل `benchmark` للتأكد من الأداء
- [ ] مراقبة استخدام GPU (يجب > 80%)

---

## قراءات إضافية

- [تفاصيل تنفيذ MCTS](../mcts-implementation) — مصدر الحاجة للاستدلال الدفعي
- [تكميم النموذج والنشر](../quantization-deploy) — مزيد من تحسين الأداء
- [دليل التثبيت الكامل](../../hands-on/setup) — خطوات التثبيت لمختلف المنصات
