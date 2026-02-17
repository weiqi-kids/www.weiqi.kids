---
sidebar_position: 2
title: सामान्य कमांड
---

# KataGo सामान्य कमांड

यह लेख KataGo के दो मुख्य ऑपरेशन मोड परिचय करता है: GTP प्रोटोकॉल और Analysis Engine, साथ ही सामान्य कमांड का विस्तृत विवरण।

## GTP प्रोटोकॉल परिचय

GTP (Go Text Protocol) गो प्रोग्राम के बीच संवाद का मानक प्रोटोकॉल है। अधिकांश गो GUI (जैसे Sabaki, Lizzie) AI इंजन से संवाद के लिए GTP उपयोग करते हैं।

### GTP मोड शुरू करें

```bash
katago gtp -model /path/to/model.bin.gz -config /path/to/config.cfg
```

### GTP प्रोटोकॉल बुनियादी फॉर्मेट

```
[id] command_name [arguments]
```

- `id`: वैकल्पिक कमांड नंबर, प्रतिक्रिया ट्रैकिंग के लिए
- `command_name`: कमांड नाम
- `arguments`: कमांड पैरामीटर

प्रतिक्रिया फॉर्मेट:
```
=[id] response_data     # सफल
?[id] error_message     # विफल
```

### बुनियादी उदाहरण

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

## सामान्य GTP कमांड

### प्रोग्राम जानकारी

| कमांड | विवरण | उदाहरण |
|------|------|------|
| `name` | प्रोग्राम नाम प्राप्त करें | `name` → `= KataGo` |
| `version` | संस्करण नंबर प्राप्त करें | `version` → `= 1.15.3` |
| `list_commands` | सभी समर्थित कमांड सूचीबद्ध करें | `list_commands` |
| `protocol_version` | GTP प्रोटोकॉल संस्करण | `protocol_version` → `= 2` |

### बोर्ड सेटिंग

```
# बोर्ड साइज़ सेट करें (9, 13, 19)
boardsize 19

# कोमी सेट करें
komi 7.5

# बोर्ड साफ़ करें
clear_board

# नियम सेट करें (KataGo एक्सटेंशन)
kata-set-rules chinese    # चीनी नियम
kata-set-rules japanese   # जापानी नियम
kata-set-rules tromp-taylor
```

### खेल संबंधित

```
# चाल खेलें
play black Q16    # काला Q16 पर
play white D4     # सफेद D4 पर
play black pass   # काला पास

# AI से चाल उत्पन्न करवाएं
genmove black     # काले की चाल उत्पन्न
genmove white     # सफेद की चाल उत्पन्न

# पूर्ववत करें
undo              # एक चाल पूर्ववत

# चाल सीमा सेट करें
kata-set-param maxVisits 1000    # अधिकतम खोज संख्या
```

### स्थिति क्वेरी

```
# बोर्ड दिखाएं
showboard

# वर्तमान खिलाड़ी प्राप्त करें
kata-get-player

# विश्लेषण परिणाम प्राप्त करें
kata-analyze black 100    # काले का विश्लेषण, 100 खोज
```

### नियम संबंधित

```
# वर्तमान नियम प्राप्त करें
kata-get-rules

# नियम सेट करें
kata-set-rules chinese

# हैंडीकैप सेट करें
fixed_handicap 4     # मानक 4 पत्थर हैंडीकैप स्थिति
place_free_handicap 4  # मुक्त हैंडीकैप
```

## KataGo एक्सटेंशन कमांड

KataGo मानक GTP के अलावा कई एक्सटेंशन कमांड प्रदान करता है:

### kata-analyze

वर्तमान स्थिति का रीयल-टाइम विश्लेषण:

```
kata-analyze [player] [visits] [interval]
```

पैरामीटर:
- `player`: किसका विश्लेषण (black/white)
- `visits`: खोज संख्या
- `interval`: रिपोर्ट इंटरवल (centiseconds, 1/100 सेकंड)

उदाहरण:
```
kata-analyze black 1000 100
```

आउटपुट:
```
info move Q3 visits 523 winrate 0.5432 scoreMean 2.31 scoreSelfplay 2.45 prior 0.1234 order 0 pv Q3 R4 Q5 ...
info move D4 visits 312 winrate 0.5123 scoreMean 1.82 scoreSelfplay 1.95 prior 0.0987 order 1 pv D4 C6 E3 ...
...
```

आउटपुट फ़ील्ड विवरण:

| फ़ील्ड | विवरण |
|------|------|
| `move` | चाल स्थिति |
| `visits` | खोज विज़िट संख्या |
| `winrate` | जीत दर (0-1) |
| `scoreMean` | अपेक्षित अंक अंतर |
| `scoreSelfplay` | स्व-खेल अपेक्षित अंक |
| `prior` | न्यूरल नेटवर्क प्रायर प्रोबेबिलिटी |
| `order` | रैंकिंग क्रम |
| `pv` | मुख्य वेरिएशन (Principal Variation) |

### kata-raw-nn

रॉ न्यूरल नेटवर्क आउटपुट प्राप्त करें:

```
kata-raw-nn [symmetry]
```

आउटपुट में शामिल:
- Policy प्रोबेबिलिटी वितरण
- Value भविष्यवाणी
- क्षेत्र भविष्यवाणी आदि

### kata-debug-print

विस्तृत खोज जानकारी दिखाएं, डीबगिंग के लिए:

```
kata-debug-print move Q16
```

### ताकत समायोजन

```
# अधिकतम विज़िट संख्या सेट करें
kata-set-param maxVisits 100      # कमज़ोर
kata-set-param maxVisits 10000    # मजबूत

# सोचने का समय सेट करें
kata-time-settings main 60 0      # प्रति पक्ष 60 सेकंड
kata-time-settings byoyomi 30 5   # बायोयोमी 30 सेकंड 5 बार
```

## Analysis Engine उपयोग

Analysis Engine KataGo का एक और ऑपरेशन मोड है, JSON फॉर्मेट में संवाद करता है, प्रोग्रामिंग उपयोग के लिए अधिक उपयुक्त।

### Analysis Engine शुरू करें

```bash
katago analysis -model /path/to/model.bin.gz -config /path/to/config.cfg
```

### बुनियादी उपयोग प्रवाह

```
आपका प्रोग्राम ──JSON अनुरोध──> KataGo Analysis Engine ──JSON प्रतिक्रिया──> आपका प्रोग्राम
```

### अनुरोध फॉर्मेट

प्रत्येक अनुरोध एक JSON ऑब्जेक्ट है, एक लाइन होनी चाहिए:

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

### अनुरोध फ़ील्ड विवरण

| फ़ील्ड | आवश्यक | विवरण |
|------|------|------|
| `id` | हां | क्वेरी पहचानकर्ता, प्रतिक्रिया मिलान के लिए |
| `moves` | नहीं | चाल अनुक्रम `[["B","Q16"],["W","D4"]]` |
| `initialStones` | नहीं | प्रारंभिक पत्थर `[["B","Q16"],["W","D4"]]` |
| `rules` | हां | नियम नाम |
| `komi` | हां | कोमी |
| `boardXSize` | हां | बोर्ड चौड़ाई |
| `boardYSize` | हां | बोर्ड ऊंचाई |
| `analyzeTurns` | नहीं | विश्लेषण करने वाली चाल संख्या (0-indexed) |
| `maxVisits` | नहीं | कॉन्फ़िग फ़ाइल maxVisits ओवरराइड |

### प्रतिक्रिया फॉर्मेट

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

### प्रतिक्रिया फ़ील्ड विवरण

#### moveInfos फ़ील्ड

| फ़ील्ड | विवरण |
|------|------|
| `move` | चाल निर्देशांक |
| `visits` | उस चाल की खोज विज़िट संख्या |
| `winrate` | जीत दर (0-1, वर्तमान खिलाड़ी के लिए) |
| `scoreMean` | अपेक्षित अंतिम अंक अंतर |
| `scoreStdev` | अंक मानक विचलन |
| `scoreLead` | वर्तमान आगे अंक |
| `prior` | न्यूरल नेटवर्क प्रायर प्रोबेबिलिटी |
| `order` | रैंकिंग (0 = सर्वोत्तम) |
| `pv` | मुख्य वेरिएशन अनुक्रम |

#### rootInfo फ़ील्ड

| फ़ील्ड | विवरण |
|------|------|
| `visits` | कुल खोज विज़िट संख्या |
| `winrate` | वर्तमान स्थिति जीत दर |
| `scoreLead` | वर्तमान आगे अंक |
| `scoreSelfplay` | स्व-खेल अपेक्षित अंक |

#### ownership फ़ील्ड

एक-आयामी सरणी, लंबाई boardXSize × boardYSize, प्रत्येक मान -1 से 1 के बीच:
- -1: सफेद क्षेत्र भविष्यवाणी
- +1: काला क्षेत्र भविष्यवाणी
- 0: अनिर्धारित/सीमा

### एडवांस्ड क्वेरी विकल्प

#### क्षेत्र मानचित्र प्राप्त करें

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

#### Policy वितरण प्राप्त करें

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

#### रिपोर्ट की चालों की संख्या सीमित करें

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

#### विशिष्ट चालों का विश्लेषण

```json
{
  "id": "specific_moves",
  "moves": [...],
  "rules": "chinese",
  "komi": 7.5,
  "boardXSize": 19,
  "boardYSize": 19,
  "analyzeTurns": [10],
  "allowMoves": [["B","Q16"],["B","D4"],["B","Q4"]]
}
```

### पूर्ण उदाहरण: Python एकीकरण

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

        # क्वेरी भेजें
        self.process.stdin.write(json.dumps(query) + '\n')
        self.process.stdin.flush()

        # प्रतिक्रिया पढ़ें
        response_line = self.process.stdout.readline()
        return json.loads(response_line)

    def close(self):
        self.process.terminate()


# उपयोग उदाहरण
engine = KataGoEngine(
    '/usr/local/bin/katago',
    '/path/to/model.bin.gz',
    '/path/to/config.cfg'
)

# स्थिति का विश्लेषण करें
result = engine.analyze([
    ['B', 'Q16'],
    ['W', 'D4'],
    ['B', 'Q4'],
    ['W', 'D16']
])

# सर्वोत्तम चाल प्रिंट करें
best_move = result['moveInfos'][0]
print(f"सर्वोत्तम चाल: {best_move['move']}")
print(f"जीत दर: {best_move['winrate']:.1%}")
print(f"आगे अंक: {best_move['scoreLead']:.1f}")

engine.close()
```

### पूर्ण उदाहरण: Node.js एकीकरण

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

// उपयोग उदाहरण
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

  console.log('सर्वोत्तम चाल:', result.moveInfos[0].move);
  console.log('जीत दर:', (result.moveInfos[0].winrate * 100).toFixed(1) + '%');

  engine.close();
}

main();
```

## निर्देशांक प्रणाली

KataGo मानक गो निर्देशांक प्रणाली उपयोग करता है:

### अक्षर निर्देशांक

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

नोट: I अक्षर नहीं है (1 अंक से भ्रम से बचने के लिए)।

### निर्देशांक रूपांतरण

```python
def coord_to_gtp(x, y, board_size=19):
    """(x, y) निर्देशांक को GTP फॉर्मेट में बदलें"""
    letters = 'ABCDEFGHJKLMNOPQRST'
    return f"{letters[x]}{board_size - y}"

def gtp_to_coord(gtp_coord, board_size=19):
    """GTP निर्देशांक को (x, y) में बदलें"""
    letters = 'ABCDEFGHJKLMNOPQRST'
    x = letters.index(gtp_coord[0].upper())
    y = board_size - int(gtp_coord[1:])
    return (x, y)
```

## सामान्य उपयोग पैटर्न

### खेल मोड

```bash
# GTP मोड शुरू करें
katago gtp -model model.bin.gz -config gtp.cfg

# GTP कमांड अनुक्रम
boardsize 19
komi 7.5
play black Q16
genmove white
play black Q4
genmove white
...
```

### बैच विश्लेषण मोड

```python
# एक गेम की सभी चालों का विश्लेषण
sgf_moves = parse_sgf('game.sgf')

for i in range(len(sgf_moves)):
    result = engine.analyze(sgf_moves[:i+1])
    winrate = result['rootInfo']['winrate']
    print(f"चाल {i+1}: जीत दर {winrate:.1%}")
```

### रीयल-टाइम विश्लेषण मोड

`kata-analyze` से रीयल-टाइम विश्लेषण:

```
kata-analyze black 1000 50
```

हर 0.5 सेकंड विश्लेषण परिणाम आउटपुट करेगा, 1000 विज़िट तक।

## प्रदर्शन ट्यूनिंग

### खोज सेटिंग

```ini
# खोज मात्रा बढ़ाकर सटीकता बढ़ाएं
maxVisits = 1000

# या समय नियंत्रण उपयोग करें
maxTime = 10  # प्रति चाल अधिकतम 10 सेकंड
```

### मल्टी-थ्रेड सेटिंग

```ini
# CPU थ्रेड संख्या
numSearchThreads = 8

# GPU बैच प्रोसेसिंग
numNNServerThreadsPerModel = 2
nnMaxBatchSize = 16
```

### मेमोरी सेटिंग

```ini
# मेमोरी उपयोग कम करें
nnCacheSizePowerOfTwo = 20  # डिफ़ॉल्ट 23
```

## अगला कदम

कमांड उपयोग समझने के बाद, यदि आप KataGo के कार्यान्वयन में गहराई से जाना चाहते हैं, [सोर्स कोड आर्किटेक्चर](./architecture.md) पढ़ें।
