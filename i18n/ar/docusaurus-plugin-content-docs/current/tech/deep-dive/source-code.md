---
sidebar_position: 2
title: دليل قراءة الكود المصدري لـ KataGo
description: هيكل كود KataGo، الوحدات الأساسية وتصميم البنية
---

# دليل قراءة الكود المصدري لـ KataGo

يساعدك هذا المقال على فهم هيكل كود KataGo، وهو مناسب للمهندسين الذين يرغبون في البحث المعمق أو المساهمة بالكود.

---

## الحصول على الكود المصدري

```bash
git clone https://github.com/lightvector/KataGo.git
cd KataGo
```

---

## هيكل الدليل

```
KataGo/
├── cpp/                    # محرك C++ الأساسي
│   ├── main.cpp            # نقطة دخول البرنامج الرئيسي
│   ├── command/            # معالجة الأوامر
│   ├── core/               # أدوات أساسية
│   ├── game/               # قواعد الغو
│   ├── search/             # بحث MCTS
│   ├── neuralnet/          # استدلال الشبكة العصبية
│   ├── dataio/             # إدخال/إخراج البيانات
│   └── tests/              # اختبارات الوحدة
│
├── python/                 # كود تدريب Python
│   ├── train.py            # البرنامج الرئيسي للتدريب
│   ├── model.py            # تعريف بنية الشبكة
│   ├── data/               # معالجة البيانات
│   └── configs/            # إعدادات التدريب
│
└── docs/                   # التوثيق
```

---

## تحليل الوحدات الأساسية

### 1. game/ — قواعد الغو

التنفيذ الكامل لقواعد الغو.

#### board.h / board.cpp

```cpp
// تمثيل حالة اللوحة
class Board {
public:
    static constexpr int MAX_BOARD_SIZE = 19;

    // حالة اللوحة
    Color colors[MAX_ARR_SIZE];  // لون كل موقع
    Chain chains[MAX_ARR_SIZE];  // معلومات السلاسل

    // العمليات الأساسية
    bool playMove(Loc loc, Player pla);  // لعب حركة
    bool isLegal(Loc loc, Player pla);   // التحقق من القانونية
    void calculateArea(Color* area);      // حساب المنطقة
};
```

**المفاهيم المقابلة**:
- نموذج الشبكة: هيكل بيانات اللوحة
- المنطقة المتصلة: تمثيل السلسلة (Chain)
- حساب الحريات: تتبع liberty

#### rules.h / rules.cpp

```cpp
// دعم قواعد متعددة
struct Rules {
    enum KoRule { SIMPLE_KO, POSITIONAL_KO, SITUATIONAL_KO };
    enum ScoringRule { TERRITORY_SCORING, AREA_SCORING };
    enum TaxRule { NO_TAX, TAX_SEKI, TAX_ALL };

    KoRule koRule;
    ScoringRule scoringRule;
    TaxRule taxRule;
    float komi;

    // تحويل أسماء القواعد
    static Rules parseRules(const std::string& name);
};
```

القواعد المدعومة:
- `chinese`: القواعد الصينية (عد الأحجار)
- `japanese`: القواعد اليابانية (عد المنطقة)
- `korean`: القواعد الكورية
- `aga`: القواعد الأمريكية
- `tromp-taylor`: قواعد Tromp-Taylor

---

### 2. search/ — بحث MCTS

تنفيذ بحث شجرة مونت كارلو.

#### search.h / search.cpp

```cpp
class Search {
public:
    // البحث الأساسي
    void runWholeSearch(Player pla);

    // خطوات MCTS
    void selectNode();           // اختيار العقدة
    void expandNode();           // توسيع العقدة
    void evaluateNode();         // تقييم الشبكة العصبية
    void backpropValue();        // تحديث الرجوع

    // الحصول على النتائج
    Loc getChosenMove();
    std::vector<MoveInfo> getSortedMoveInfos();
};
```

**المفاهيم المقابلة**:
- الخطوات الأربع لـ MCTS: تتوافق مع select → expand → evaluate → backprop
- صيغة PUCT: مطبقة في `selectNode()`

#### searchparams.h

```cpp
struct SearchParams {
    // التحكم في البحث
    int64_t maxVisits;          // الحد الأقصى للزيارات
    double maxTime;             // الحد الأقصى للوقت

    // معلمات PUCT
    double cpuctExploration;    // ثابت الاستكشاف
    double cpuctBase;

    // الخسارة الافتراضية
    int virtualLoss;

    // ضوضاء عقدة الجذر
    double rootNoiseEnabled;
    double rootDirichletAlpha;
};
```

---

### 3. neuralnet/ — استدلال الشبكة العصبية

محرك استدلال الشبكة العصبية.

#### nninputs.h / nninputs.cpp

```cpp
// ميزات إدخال الشبكة العصبية
class NNInputs {
public:
    // مستويات الميزات
    static constexpr int NUM_FEATURES = 22;

    // ملء الميزات
    static void fillFeatures(
        const Board& board,
        const BoardHistory& hist,
        float* features
    );
};
```

ميزات الإدخال تشمل:
- مواقع الأحجار السوداء، مواقع الأحجار البيضاء
- عدد الحريات (1، 2، 3+)
- خطوات التاريخ
- ترميز القواعد

**المفاهيم المقابلة**:
- تكديس التاريخ: إدخال متعدد الإطارات
- قناع الحركات القانونية: تصفية الحركات المحظورة

#### nneval.h / nneval.cpp

```cpp
// نتائج تقييم الشبكة العصبية
struct NNOutput {
    // إخراج السياسة (362 موقع، يشمل pass)
    float policyProbs[NNPos::MAX_NN_POLICY_SIZE];

    // إخراج القيمة
    float winProb;       // معدل الفوز
    float lossProb;      // معدل الخسارة
    float noResultProb;  // معدل التعادل

    // الإخراج المساعد
    float scoreMean;     // توقع النقاط
    float scoreStdev;    // الانحراف المعياري للنقاط
    float lead;          // النقاط المتقدمة

    // توقع المنطقة
    float ownership[NNPos::MAX_BOARD_AREA];
};
```

**المفاهيم المقابلة**:
- شبكة السياسة: policyProbs
- شبكة القيمة: winProb, scoreMean
- الشبكة برأسين: تصميم إخراج متعدد

---

### 4. command/ — معالجة الأوامر

تنفيذ أوضاع التشغيل المختلفة.

#### gtp.cpp

تنفيذ وضع GTP (Go Text Protocol):

```cpp
void MainCmds::gtp(const std::vector<std::string>& args) {
    // تحليل وتنفيذ الأوامر
    while(true) {
        std::string line;
        std::getline(std::cin, line);

        if(line == "name") {
            respond("KataGo");
        }
        else if(line.find("play") == 0) {
            // معالجة أمر اللعب
        }
        else if(line.find("genmove") == 0) {
            // تنفيذ البحث وإرجاع أفضل حركة
        }
        // ... أوامر أخرى
    }
}
```

#### analysis.cpp

تنفيذ محرك التحليل:

```cpp
void MainCmds::analysis(const std::vector<std::string>& args) {
    while(true) {
        // قراءة طلب JSON
        std::string line;
        std::getline(std::cin, line);
        json query = json::parse(line);

        // إنشاء حالة اللوحة
        Board board = setupBoard(query);

        // تنفيذ التحليل
        Search search(...);
        search.runWholeSearch();

        // إخراج استجابة JSON
        json response = formatResponse(search);
        std::cout << response.dump() << std::endl;
    }
}
```

---

## كود تدريب Python

### model.py — بنية الشبكة

```python
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        # الالتفاف الأولي
        self.initial_conv = nn.Conv2d(
            in_channels=config.input_features,
            out_channels=config.trunk_channels,
            kernel_size=3, padding=1
        )

        # البرج المتبقي
        self.trunk = nn.ModuleList([
            ResidualBlock(config.trunk_channels)
            for _ in range(config.num_blocks)
        ])

        # رؤوس الإخراج
        self.policy_head = PolicyHead(config)
        self.value_head = ValueHead(config)
        self.ownership_head = OwnershipHead(config)

    def forward(self, x):
        # الالتفاف الأولي
        x = self.initial_conv(x)

        # البرج المتبقي
        for block in self.trunk:
            x = block(x)

        # إخراج متعدد الرؤوس
        policy = self.policy_head(x)
        value = self.value_head(x)
        ownership = self.ownership_head(x)

        return policy, value, ownership
```

**المفاهيم المقابلة**:
- عملية الالتفاف: Conv2d
- الاتصال المتبقي: ResidualBlock
- البرج المتبقي: هيكل trunk

### train.py — دورة التدريب

```python
def train_step(model, optimizer, batch):
    # التمرير الأمامي
    policy_pred, value_pred, ownership_pred = model(batch.inputs)

    # حساب الخسارة
    policy_loss = cross_entropy(policy_pred, batch.policy_target)
    value_loss = mse_loss(value_pred, batch.value_target)
    ownership_loss = mse_loss(ownership_pred, batch.ownership_target)

    total_loss = policy_loss + value_loss + ownership_loss

    # التمرير العكسي
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()
```

**المفاهيم المقابلة**:
- التمرير الأمامي: model(batch.inputs)
- التمرير العكسي: total_loss.backward()
- Adam: optimizer.step()

---

## تنفيذ الخوارزميات الرئيسية

### صيغة اختيار PUCT

```cpp
// search.cpp
double Search::getPUCTScore(const SearchNode* node, int moveIdx) {
    double Q = node->getChildValue(moveIdx);
    double P = node->getChildPolicy(moveIdx);
    double N_parent = node->visits;
    double N_child = node->getChildVisits(moveIdx);

    double exploration = params.cpuctExploration;
    double cpuct = exploration * sqrt(N_parent) / (1.0 + N_child);

    return Q + cpuct * P;
}
```

### الخسارة الافتراضية

```cpp
// تجنب اختيار نفس العقدة من خيوط متعددة
void Search::applyVirtualLoss(SearchNode* node) {
    node->virtualLoss += params.virtualLoss;
}

void Search::removeVirtualLoss(SearchNode* node) {
    node->virtualLoss -= params.virtualLoss;
}
```

**المفاهيم المقابلة**:
- الخسارة الافتراضية: تقنية البحث المتوازي

---

## البناء والتصحيح

### البناء (وضع Debug)

```bash
cd cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=OPENCL -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

### اختبارات الوحدة

```bash
./katago runtests
```

### نصائح التصحيح

```cpp
// تفعيل السجلات المفصلة
#define SEARCH_DEBUG 1

// إضافة نقاط توقف في البحث
if(node->visits > 1000) {
    // تعيين نقطة توقف لفحص حالة البحث
}
```

---

## قراءات إضافية

- [تحليل آلية تدريب KataGo](../training) — عملية التدريب الكاملة
- [المشاركة في مجتمع المصادر المفتوحة](../contributing) — دليل المساهمة
- [مرجع المفاهيم السريع](/docs/animations/) — مقارنة 109 مفاهيم
