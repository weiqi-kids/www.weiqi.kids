---
sidebar_position: 4
title: المشاركة في مجتمع المصادر المفتوحة
description: الانضمام لمجتمع KataGo مفتوح المصدر، المساهمة بقوة الحوسبة أو الكود
---

# المشاركة في مجتمع المصادر المفتوحة

KataGo هو مشروع مفتوح المصدر نشط، وهناك طرق متعددة للمشاركة والمساهمة.

---

## نظرة عامة على طرق المساهمة

| الطريقة | الصعوبة | المتطلبات |
|---------|---------|-----------|
| **المساهمة بقوة الحوسبة** | منخفضة | حاسوب مع GPU |
| **الإبلاغ عن المشاكل** | منخفضة | حساب GitHub |
| **تحسين التوثيق** | متوسطة | معرفة المحتوى التقني |
| **المساهمة بالكود** | عالية | مهارات تطوير C++/Python |

---

## المساهمة بقوة الحوسبة: التدريب الموزع

### مقدمة عن KataGo Training

KataGo Training هي شبكة تدريب موزعة عالمية:

- المتطوعون يساهمون بقوة GPU لتنفيذ اللعب الذاتي
- بيانات اللعب الذاتي تُرفع للخادم المركزي
- الخادم يدرب نماذج جديدة بشكل دوري
- النماذج الجديدة توزع على المتطوعين لمواصلة اللعب

الموقع الرسمي: https://katagotraining.org/

### خطوات المشاركة

#### 1. إنشاء حساب

اذهب إلى https://katagotraining.org/ وسجل حساباً.

#### 2. تحميل KataGo

```bash
# تحميل أحدث إصدار
wget https://github.com/lightvector/KataGo/releases/download/v1.15.3/katago-v1.15.3-cuda11.1-linux-x64.zip
unzip katago-v1.15.3-cuda11.1-linux-x64.zip
```

#### 3. إعداد وضع المساهمة

```bash
# التشغيل الأول سيرشدك خلال الإعداد
./katago contribute -username YOUR_USERNAME -password YOUR_PASSWORD
```

سيقوم النظام تلقائياً بـ:
- تحميل أحدث نموذج
- تنفيذ اللعب الذاتي
- رفع بيانات اللعب

#### 4. التشغيل في الخلفية

```bash
# استخدام screen أو tmux للتشغيل في الخلفية
screen -S katago
./katago contribute -username YOUR_USERNAME -password YOUR_PASSWORD
# Ctrl+A, D للخروج من screen
```

### إحصائيات المساهمة

يمكنك مشاهدة ما يلي في https://katagotraining.org/contributions/:
- ترتيب مساهماتك
- إجمالي عدد المباريات المساهمة
- أحدث النماذج المدربة

---

## الإبلاغ عن المشاكل

### أين يمكن الإبلاغ

- **GitHub Issues**: https://github.com/lightvector/KataGo/issues
- **Discord**: https://discord.gg/bqkZAz3

### تقرير المشكلة الجيد يتضمن

1. **إصدار KataGo**: `katago version`
2. **نظام التشغيل**: Windows/Linux/macOS
3. **الأجهزة**: طراز GPU، الذاكرة
4. **رسالة الخطأ الكاملة**: نسخ السجل الكامل
5. **خطوات إعادة الإنتاج**: كيفية تفعيل هذه المشكلة

### مثال

```markdown
## وصف المشكلة
يظهر خطأ نفاد الذاكرة عند تشغيل benchmark

## البيئة
- إصدار KataGo: 1.15.3
- نظام التشغيل: Ubuntu 22.04
- GPU: RTX 3060 12GB
- النموذج: kata-b40c256.bin.gz

## رسالة الخطأ
```
CUDA error: out of memory
```

## خطوات إعادة الإنتاج
1. تشغيل `katago benchmark -model kata-b40c256.bin.gz`
2. الانتظار حوالي 30 ثانية
3. ظهور الخطأ
```

---

## تحسين التوثيق

### مواقع التوثيق

- **README**: `README.md`
- **توثيق GTP**: `docs/GTP_Extensions.md`
- **توثيق Analysis**: `docs/Analysis_Engine.md`
- **توثيق التدريب**: `python/README.md`

### عملية المساهمة

1. Fork المشروع
2. إنشاء فرع جديد
3. تعديل التوثيق
4. تقديم Pull Request

```bash
git clone https://github.com/YOUR_USERNAME/KataGo.git
cd KataGo
git checkout -b improve-docs
# تعديل الملفات
git add .
git commit -m "Improve documentation for Analysis Engine"
git push origin improve-docs
# إنشاء Pull Request على GitHub
```

---

## المساهمة بالكود

### إعداد بيئة التطوير

```bash
# نسخ المشروع
git clone https://github.com/lightvector/KataGo.git
cd KataGo

# البناء (وضع Debug)
cd cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=OPENCL -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# تشغيل الاختبارات
./katago runtests
```

### أسلوب الكود

يستخدم KataGo أسلوب الكود التالي:

**C++**:
- مسافتان للإزاحة
- الأقواس المنحنية في نفس السطر
- أسماء المتغيرات بـ camelCase
- أسماء الفئات بـ PascalCase

```cpp
class ExampleClass {
public:
  void exampleMethod() {
    int localVariable = 0;
    if(condition) {
      doSomething();
    }
  }
};
```

**Python**:
- اتباع PEP 8
- 4 مسافات للإزاحة

### مجالات المساهمة

| المجال | موقع الملفات | المهارات المطلوبة |
|--------|-------------|------------------|
| المحرك الأساسي | `cpp/` | C++, CUDA/OpenCL |
| برنامج التدريب | `python/` | Python, PyTorch |
| بروتوكول GTP | `cpp/command/gtp.cpp` | C++ |
| Analysis API | `cpp/command/analysis.cpp` | C++, JSON |
| الاختبارات | `cpp/tests/` | C++ |

### عملية Pull Request

1. **إنشاء Issue**: ناقش التغيير الذي تريده أولاً
2. **Fork & Clone**: إنشاء فرعك الخاص
3. **التطوير والاختبار**: تأكد من نجاح جميع الاختبارات
4. **تقديم PR**: وصف مفصل لمحتوى التغيير
5. **Code Review**: الرد على ملاحظات المشرفين
6. **الدمج**: المشرفون يدمجون كودك

### مثال PR

```markdown
## وصف التغيير
إضافة دعم لقواعد نيوزيلندا

## محتوى التغيير
- إضافة قاعدة NEW_ZEALAND في rules.cpp
- تحديث أمر GTP لدعم `kata-set-rules nz`
- إضافة اختبارات وحدة

## نتائج الاختبار
- جميع الاختبارات الحالية ناجحة
- الاختبارات الجديدة ناجحة

## Issue ذو صلة
Fixes #123
```

---

## موارد المجتمع

### الروابط الرسمية

| المورد | الرابط |
|--------|--------|
| GitHub | https://github.com/lightvector/KataGo |
| Discord | https://discord.gg/bqkZAz3 |
| شبكة التدريب | https://katagotraining.org/ |

### منتديات النقاش

- **Discord**: نقاش فوري، أسئلة تقنية
- **GitHub Discussions**: نقاشات طويلة، اقتراحات ميزات
- **Reddit r/baduk**: نقاش عام عن الذكاء الاصطناعي للغو

### المشاريع ذات الصلة

| المشروع | الوصف | الرابط |
|---------|-------|--------|
| KaTrain | أداة تعليم وتحليل | github.com/sanderland/katrain |
| Lizzie | واجهة تحليل | github.com/featurecat/lizzie |
| Sabaki | محرر سجلات المباريات | sabaki.yichuanshen.de |
| BadukAI | تحليل عبر الإنترنت | baduk.ai |

---

## التقدير والمكافآت

### قائمة المساهمين

جميع المساهمين يُذكرون في:
- صفحة GitHub Contributors
- لوحة ترتيب المساهمين في KataGo Training

### فوائد التعلم

فوائد المشاركة في المشاريع مفتوحة المصدر:
- تعلم بنية أنظمة AI على مستوى صناعي
- التواصل مع مطورين من حول العالم
- تراكم سجل المساهمات مفتوحة المصدر
- فهم معمق لتقنية الذكاء الاصطناعي للغو

---

## قراءات إضافية

- [دليل قراءة الكود المصدري](../source-code) — فهم هيكل الكود
- [تحليل آلية تدريب KataGo](../training) — تجارب التدريب المحلي
- [فهم الذكاء الاصطناعي للغو في مقال واحد](../../how-it-works/) — المبادئ التقنية
