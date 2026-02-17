---
sidebar_position: 2
title: الأوامر الشائعة
---

# أوامر KataGo الشائعة

تقدم هذه المقالة وضعي التشغيل الرئيسيين لـ KataGo: بروتوكول GTP وAnalysis Engine، مع شرح مفصل للأوامر الشائعة.

## مقدمة عن بروتوكول GTP

GTP (Go Text Protocol) هو بروتوكول اتصال قياسي بين برامج Go. معظم واجهات Go الرسومية (مثل Sabaki، Lizzie) تستخدم GTP للتواصل مع محركات AI.

### بدء وضع GTP

```bash
katago gtp -model /path/to/model.bin.gz -config /path/to/config.cfg
```

### الصيغة الأساسية لبروتوكول GTP

```
[id] command_name [arguments]
```

- `id`: رقم الأمر الاختياري، لتتبع الردود
- `command_name`: اسم الأمر
- `arguments`: معاملات الأمر

صيغة الرد:
```
=[id] response_data     # نجاح
?[id] error_message     # فشل
```

### مثال أساسي

```
1 name
=1 KataGo

2 version
=2 1.15.3

3 boardsize 19
=3

4 komi 7.5
=4

5 play black Q16
=5

6 genmove white
=6 D4
```

## أوامر GTP الشائعة

### معلومات البرنامج

| الأمر | الشرح | مثال |
|------|------|------|
| `name` | الحصول على اسم البرنامج | `name` → `= KataGo` |
| `version` | الحصول على رقم الإصدار | `version` → `= 1.15.3` |
| `list_commands` | قائمة جميع الأوامر المدعومة | `list_commands` |
| `protocol_version` | إصدار بروتوكول GTP | `protocol_version` → `= 2` |

### إعدادات اللوحة

```
# ضبط حجم اللوحة (9، 13، 19)
boardsize 19

# ضبط الكومي
komi 7.5

# مسح اللوحة
clear_board

# ضبط القواعد (امتداد KataGo)
kata-set-rules chinese    # القواعد الصينية
kata-set-rules japanese   # القواعد اليابانية
kata-set-rules tromp-taylor
```

### متعلقة باللعب

```
# وضع حجر
play black Q16    # الأسود يلعب في Q16
play white D4     # الأبيض يلعب في D4
play black pass   # الأسود يمرر

# جعل AI يلعب
genmove black     # توليد حركة للأسود
genmove white     # توليد حركة للأبيض

# التراجع
undo              # التراجع عن حركة واحدة

# ضبط حد الحركات
kata-set-param maxVisits 1000    # ضبط الحد الأقصى للبحث
```

### استعلام الموقف

```
# عرض اللوحة
showboard

# الحصول على اللاعب الحالي
kata-get-player

# الحصول على نتائج التحليل
kata-analyze black 100    # تحليل الأسود، 100 عملية بحث
```

### متعلقة بالقواعد

```
# الحصول على القواعد الحالية
kata-get-rules

# ضبط القواعد
kata-set-rules chinese

# ضبط الهانديكاب
fixed_handicap 4     # مواقع هانديكاب قياسية لأربعة أحجار
place_free_handicap 4  # هانديكاب حر
```

## أوامر KataGo الموسعة

يوفر KataGo العديد من الأوامر الموسعة خارج GTP القياسي:

### kata-analyze

تحليل الموقف الحالي في الوقت الفعلي:

```
kata-analyze [player] [visits] [interval]
```

المعاملات:
- `player`: تحليل أي طرف (black/white)
- `visits`: عدد عمليات البحث
- `interval`: فاصل التقرير (سنتي ثانية، 1/100 ثانية)

مثال:
```
kata-analyze black 1000 100
```

الإخراج:
```
info move Q3 visits 523 winrate 0.5432 scoreMean 2.31 scoreSelfplay 2.45 prior 0.1234 order 0 pv Q3 R4 Q5 ...
info move D4 visits 312 winrate 0.5123 scoreMean 1.82 scoreSelfplay 1.95 prior 0.0987 order 1 pv D4 C6 E3 ...
...
```

شرح حقول الإخراج:

| الحقل | الشرح |
|------|------|
| `move` | نقطة اللعب |
| `visits` | عدد زيارات البحث |
| `winrate` | نسبة الفوز (0-1) |
| `scoreMean` | فرق النقاط المتوقع |
| `scoreSelfplay` | نقاط السلف-بلاي المتوقعة |
| `prior` | الاحتمال المسبق من الشبكة العصبية |
| `order` | ترتيب التصنيف |
| `pv` | التغيير الرئيسي (Principal Variation) |

### kata-raw-nn

الحصول على إخراج الشبكة العصبية الخام:

```
kata-raw-nn [symmetry]
```

الإخراج يتضمن:
- توزيع احتمالات Policy
- تنبؤ Value
- تنبؤ المنطقة وغيرها

### kata-debug-print

عرض معلومات البحث التفصيلية، للتصحيح:

```
kata-debug-print move Q16
```

### ضبط القوة

```
# ضبط الحد الأقصى للزيارات
kata-set-param maxVisits 100      # أضعف
kata-set-param maxVisits 10000    # أقوى

# ضبط وقت التفكير
kata-time-settings main 60 0      # 60 ثانية لكل طرف
kata-time-settings byoyomi 30 5   # 30 ثانية بيويومي 5 مرات
```

## استخدام Analysis Engine

Analysis Engine هو وضع تشغيل آخر يوفره KataGo، يستخدم صيغة JSON للتواصل، أكثر ملاءمة للاستخدام البرمجي.

### بدء Analysis Engine

```bash
katago analysis -model /path/to/model.bin.gz -config /path/to/config.cfg
```

### سير العمل الأساسي

```
برنامجك ──طلب JSON──> KataGo Analysis Engine ──رد JSON──> برنامجك
```

### صيغة الطلب

كل طلب هو كائن JSON، يجب أن يشغل سطراً واحداً:

```json
{
  "id": "query1",
  "moves": [["B","Q16"],["W","D4"],["B","Q4"]],
  "rules": "chinese",
  "komi": 7.5,
  "boardXSize": 19,
  "boardYSize": 19,
  "analyzeTurns": [2]
}
```

### شرح حقول الطلب

| الحقل | مطلوب | الشرح |
|------|------|------|
| `id` | نعم | معرف الاستعلام، لمطابقة الردود |
| `moves` | لا | تسلسل الحركات `[["B","Q16"],["W","D4"]]` |
| `initialStones` | لا | أحجار أولية `[["B","Q16"],["W","D4"]]` |
| `rules` | نعم | اسم القواعد |
| `komi` | نعم | الكومي |
| `boardXSize` | نعم | عرض اللوحة |
| `boardYSize` | نعم | ارتفاع اللوحة |
| `analyzeTurns` | لا | الحركات المراد تحليلها (0-indexed) |
| `maxVisits` | لا | تجاوز maxVisits من ملف التكوين |

### صيغة الرد

```json
{
  "id": "query1",
  "turnNumber": 2,
  "moveInfos": [
    {
      "move": "D16",
      "visits": 1234,
      "winrate": 0.5678,
      "scoreMean": 3.21,
      "scoreStdev": 15.4,
      "scoreLead": 3.21,
      "prior": 0.0892,
      "order": 0,
      "pv": ["D16", "Q10", "R14"]
    }
  ],
  "rootInfo": {
    "visits": 5000,
    "winrate": 0.5234,
    "scoreLead": 2.1,
    "scoreSelfplay": 2.3
  },
  "ownership": [...],
  "policy": [...]
}
```

### شرح حقول الرد

#### حقول moveInfos

| الحقل | الشرح |
|------|------|
| `move` | إحداثيات اللعب |
| `visits` | عدد زيارات البحث لهذه اللعبة |
| `winrate` | نسبة الفوز (0-1، للاعب الحالي) |
| `scoreMean` | فرق النقاط النهائي المتوقع |
| `scoreStdev` | الانحراف المعياري للنقاط |
| `scoreLead` | نقاط التقدم الحالي |
| `prior` | الاحتمال المسبق من الشبكة العصبية |
| `order` | التصنيف (0 = الأفضل) |
| `pv` | تسلسل التغيير الرئيسي |

#### حقول rootInfo

| الحقل | الشرح |
|------|------|
| `visits` | إجمالي زيارات البحث |
| `winrate` | نسبة فوز الموقف الحالي |
| `scoreLead` | نقاط التقدم الحالي |
| `scoreSelfplay` | نقاط السلف-بلاي المتوقعة |

#### حقل ownership

مصفوفة أحادية البعد، طولها boardXSize × boardYSize، كل قيمة بين -1 و1:
- -1: متوقع منطقة الأبيض
- +1: متوقع منطقة الأسود
- 0: غير محدد/حدود

### خيارات استعلام متقدمة

#### الحصول على خريطة المنطقة

```json
{
  "id": "ownership_query",
  "moves": [...],
  "rules": "chinese",
  "komi": 7.5,
  "boardXSize": 19,
  "boardYSize": 19,
  "analyzeTurns": [10],
  "includeOwnership": true
}
```

#### الحصول على توزيع Policy

```json
{
  "id": "policy_query",
  "moves": [...],
  "rules": "chinese",
  "komi": 7.5,
  "boardXSize": 19,
  "boardYSize": 19,
  "analyzeTurns": [10],
  "includePolicy": true
}
```

#### تحديد عدد الحركات المُبلغ عنها

```json
{
  "id": "limited_query",
  "moves": [...],
  "rules": "chinese",
  "komi": 7.5,
  "boardXSize": 19,
  "boardYSize": 19,
  "analyzeTurns": [10],
  "maxMoves": 5
}
```

### مثال كامل: دمج Python

```python
import subprocess
import json

class KataGoEngine:
    def __init__(self, katago_path, model_path, config_path):
        self.process = subprocess.Popen(
            [katago_path, 'analysis', '-model', model_path, '-config', config_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self.query_id = 0

    def analyze(self, moves, rules='chinese', komi=7.5):
        self.query_id += 1

        query = {
            'id': f'query_{self.query_id}',
            'moves': moves,
            'rules': rules,
            'komi': komi,
            'boardXSize': 19,
            'boardYSize': 19,
            'analyzeTurns': [len(moves)],
            'maxVisits': 500,
            'includeOwnership': True
        }

        # إرسال الاستعلام
        self.process.stdin.write(json.dumps(query) + '\n')
        self.process.stdin.flush()

        # قراءة الرد
        response_line = self.process.stdout.readline()
        return json.loads(response_line)

    def close(self):
        self.process.terminate()


# مثال استخدام
engine = KataGoEngine(
    '/usr/local/bin/katago',
    '/path/to/model.bin.gz',
    '/path/to/config.cfg'
)

# تحليل موقف
result = engine.analyze([
    ['B', 'Q16'],
    ['W', 'D4'],
    ['B', 'Q4'],
    ['W', 'D16']
])

# طباعة أفضل حركة
best_move = result['moveInfos'][0]
print(f"أفضل حركة: {best_move['move']}")
print(f"نسبة الفوز: {best_move['winrate']:.1%}")
print(f"التقدم بالنقاط: {best_move['scoreLead']:.1f}")

engine.close()
```

### مثال كامل: دمج Node.js

```javascript
const { spawn } = require('child_process');
const readline = require('readline');

class KataGoEngine {
  constructor(katagoPath, modelPath, configPath) {
    this.process = spawn(katagoPath, [
      'analysis',
      '-model', modelPath,
      '-config', configPath
    ]);

    this.rl = readline.createInterface({
      input: this.process.stdout,
      crlfDelay: Infinity
    });

    this.queryId = 0;
    this.callbacks = new Map();

    this.rl.on('line', (line) => {
      try {
        const response = JSON.parse(line);
        const callback = this.callbacks.get(response.id);
        if (callback) {
          callback(response);
          this.callbacks.delete(response.id);
        }
      } catch (e) {
        console.error('Parse error:', e);
      }
    });
  }

  analyze(moves, options = {}) {
    return new Promise((resolve) => {
      this.queryId++;
      const id = `query_${this.queryId}`;

      const query = {
        id,
        moves,
        rules: options.rules || 'chinese',
        komi: options.komi || 7.5,
        boardXSize: 19,
        boardYSize: 19,
        analyzeTurns: [moves.length],
        maxVisits: options.maxVisits || 500,
        includeOwnership: true
      };

      this.callbacks.set(id, resolve);
      this.process.stdin.write(JSON.stringify(query) + '\n');
    });
  }

  close() {
    this.process.kill();
  }
}

// مثال استخدام
async function main() {
  const engine = new KataGoEngine(
    '/usr/local/bin/katago',
    '/path/to/model.bin.gz',
    '/path/to/config.cfg'
  );

  const result = await engine.analyze([
    ['B', 'Q16'],
    ['W', 'D4'],
    ['B', 'Q4']
  ]);

  console.log('أفضل حركة:', result.moveInfos[0].move);
  console.log('نسبة الفوز:', (result.moveInfos[0].winrate * 100).toFixed(1) + '%');

  engine.close();
}

main();
```

## نظام الإحداثيات

يستخدم KataGo نظام إحداثيات Go القياسي:

### إحداثيات الحروف

```
   A B C D E F G H J K L M N O P Q R S T
19 . . . . . . . . . . . . . . . . . . . 19
18 . . . . . . . . . . . . . . . . . . . 18
17 . . . . . . . . . . . . . . . . . . . 17
16 . . . + . . . . . + . . . . . + . . . 16
15 . . . . . . . . . . . . . . . . . . . 15
...
 4 . . . + . . . . . + . . . . . + . . .  4
 3 . . . . . . . . . . . . . . . . . . .  3
 2 . . . . . . . . . . . . . . . . . . .  2
 1 . . . . . . . . . . . . . . . . . . .  1
   A B C D E F G H J K L M N O P Q R S T
```

ملاحظة: لا يوجد حرف I (لتجنب الخلط مع الرقم 1).

### تحويل الإحداثيات

```python
def coord_to_gtp(x, y, board_size=19):
    """تحويل إحداثيات (x, y) إلى صيغة GTP"""
    letters = 'ABCDEFGHJKLMNOPQRST'
    return f"{letters[x]}{board_size - y}"

def gtp_to_coord(gtp_coord, board_size=19):
    """تحويل إحداثيات GTP إلى (x, y)"""
    letters = 'ABCDEFGHJKLMNOPQRST'
    x = letters.index(gtp_coord[0].upper())
    y = board_size - int(gtp_coord[1:])
    return (x, y)
```

## أنماط الاستخدام الشائعة

### وضع اللعب

```bash
# بدء وضع GTP
katago gtp -model model.bin.gz -config gtp.cfg

# تسلسل أوامر GTP
boardsize 19
komi 7.5
play black Q16
genmove white
play black Q4
genmove white
...
```

### وضع التحليل الدفعي

```python
# تحليل جميع حركات مباراة
sgf_moves = parse_sgf('game.sgf')

for i in range(len(sgf_moves)):
    result = engine.analyze(sgf_moves[:i+1])
    winrate = result['rootInfo']['winrate']
    print(f"الحركة {i+1}: نسبة الفوز {winrate:.1%}")
```

### وضع التحليل الفوري

استخدام `kata-analyze` للتحليل الفوري:

```
kata-analyze black 1000 50
```

سيخرج نتائج التحليل كل 0.5 ثانية، حتى الوصول إلى 1000 زيارة.

## تحسين الأداء

### إعدادات البحث

```ini
# زيادة كمية البحث تزيد الدقة
maxVisits = 1000

# أو استخدام التحكم بالوقت
maxTime = 10  # الحد الأقصى 10 ثواني لكل حركة
```

### إعدادات تعدد الخيوط

```ini
# عدد خيوط CPU
numSearchThreads = 8

# معالجة دفعات GPU
numNNServerThreadsPerModel = 2
nnMaxBatchSize = 16
```

### إعدادات الذاكرة

```ini
# تقليل استخدام الذاكرة
nnCacheSizePowerOfTwo = 20  # الافتراضي 23
```

## الخطوة التالية

بعد فهم استخدام الأوامر، إذا كنت تريد التعمق في تنفيذ KataGo، تابع قراءة [هندسة الكود المصدري](./architecture.md).
