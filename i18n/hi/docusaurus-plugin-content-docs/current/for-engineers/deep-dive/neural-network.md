---
sidebar_position: 4
title: न्यूरल नेटवर्क आर्किटेक्चर विस्तृत व्याख्या
description: KataGo के न्यूरल नेटवर्क डिज़ाइन, इनपुट फीचर्स और मल्टी-हेड आउटपुट आर्किटेक्चर की गहन व्याख्या
---

# न्यूरल नेटवर्क आर्किटेक्चर विस्तृत व्याख्या

यह लेख KataGo न्यूरल नेटवर्क के पूर्ण आर्किटेक्चर की गहन व्याख्या करता है, इनपुट फीचर एनकोडिंग से लेकर मल्टी-हेड आउटपुट डिज़ाइन तक।

---

## आर्किटेक्चर अवलोकन

KataGo **एकल न्यूरल नेटवर्क, मल्टी-हेड आउटपुट** डिज़ाइन का उपयोग करता है:

```
इनपुट फीचर्स (19×19×22)
        │
        ▼
┌───────────────────┐
│     प्रारंभिक कन्वोल्यूशन लेयर     │
│   256 filters     │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│     रेसिड्यूअल टावर        │
│  20-60 रेसिड्यूअल ब्लॉक   │
│  + ग्लोबल पूलिंग लेयर     │
└─────────┬─────────┘
          │
    ┌─────┴─────┬─────────┬─────────┐
    │           │         │         │
    ▼           ▼         ▼         ▼
 Policy      Value     Score   Ownership
  Head       Head      Head      Head
    │           │         │         │
    ▼           ▼         ▼         ▼
362 प्रायिकता    जीत दर      अंक अंतर    361 स्वामित्व
(pass सहित)  (-1~+1)    (अंक)     (-1~+1)
```

---

## इनपुट फीचर एनकोडिंग

### फीचर प्लेन अवलोकन

KataGo **22 फीचर प्लेन** (19×19×22) का उपयोग करता है, प्रत्येक प्लेन एक 19×19 मैट्रिक्स है:

| प्लेन | सामग्री | विवरण |
|------|------|------|
| 0 | अपने पत्थर | 1 = अपना पत्थर है, 0 = नहीं |
| 1 | प्रतिद्वंद्वी के पत्थर | 1 = प्रतिद्वंद्वी का पत्थर है, 0 = नहीं |
| 2 | खाली बिंदु | 1 = खाली, 0 = पत्थर है |
| 3-10 | इतिहास स्थिति | पिछली 8 चालों के बोर्ड परिवर्तन |
| 11 | को निषिद्ध बिंदु | 1 = यहाँ को है, 0 = खेल सकते हैं |
| 12-17 | लिबर्टी एनकोडिंग | 1-लिबर्टी, 2-लिबर्टी, 3-लिबर्टी... समूह |
| 18-21 | नियम एनकोडिंग | चीनी/जापानी नियम, कोमी आदि |

### इतिहास स्थिति स्टैकिंग

न्यूरल नेटवर्क को स्थिति के **गतिशील परिवर्तनों** को समझने देने के लिए, KataGo पिछली 8 चालों की बोर्ड स्थिति स्टैक करता है:

```python
# इतिहास स्थिति एनकोडिंग (अवधारणा)
def encode_history(game_history, current_player):
    features = []

    for t in range(8):  # पिछली 8 चालें
        if t < len(game_history):
            board = game_history[-(t+1)]
            # उस समय बिंदु पर अपने/प्रतिद्वंद्वी के पत्थर एनकोड करें
            features.append(encode_board(board, current_player))
        else:
            # इतिहास अपर्याप्त, शून्य भरें
            features.append(np.zeros((19, 19)))

    return np.stack(features, axis=0)
```

### नियम एनकोडिंग

KataGo कई नियमों का समर्थन करता है, फीचर प्लेन के माध्यम से न्यूरल नेटवर्क को सूचित करता है:

```python
# नियम एनकोडिंग (अवधारणा)
def encode_rules(rules, komi):
    rule_features = np.zeros((4, 19, 19))

    # नियम प्रकार (one-hot)
    if rules == "chinese":
        rule_features[0] = 1.0
    elif rules == "japanese":
        rule_features[1] = 1.0

    # कोमी नॉर्मलाइज़ेशन
    normalized_komi = komi / 15.0  # [-1, 1] में नॉर्मलाइज़ करें
    rule_features[2] = normalized_komi

    # वर्तमान खिलाड़ी
    rule_features[3] = 1.0 if current_player == BLACK else 0.0

    return rule_features
```

---

## मुख्य नेटवर्क: रेसिड्यूअल टावर

### रेसिड्यूअल ब्लॉक संरचना

KataGo **Pre-activation ResNet** संरचना का उपयोग करता है:

```
इनपुट x
    │
    ├────────────────────┐
    │                    │
    ▼                    │
BatchNorm                │
    │                    │
    ▼                    │
ReLU                     │
    │                    │
    ▼                    │
Conv 3×3                 │
    │                    │
    ▼                    │
BatchNorm                │
    │                    │
    ▼                    │
ReLU                     │
    │                    │
    ▼                    │
Conv 3×3                 │
    │                    │
    ▼                    │
    +  ←─────────────────┘ (रेसिड्यूअल कनेक्शन)
    │
    ▼
आउटपुट
```

### कोड उदाहरण

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        return out + residual  # रेसिड्यूअल कनेक्शन
```

### ग्लोबल पूलिंग लेयर

KataGo की प्रमुख नवाचारों में से एक: रेसिड्यूअल ब्लॉक में **ग्लोबल पूलिंग** जोड़ना, नेटवर्क को वैश्विक जानकारी देखने देता है:

```python
class GlobalPoolingBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.fc = nn.Linear(channels, channels)

    def forward(self, x):
        # लोकल पाथ
        local = self.conv(x)

        # ग्लोबल पाथ
        global_pool = x.mean(dim=[2, 3])  # ग्लोबल एवरेज पूलिंग
        global_fc = self.fc(global_pool)
        global_broadcast = global_fc.unsqueeze(2).unsqueeze(3)
        global_broadcast = global_broadcast.expand(-1, -1, 19, 19)

        # मर्ज करें
        return local + global_broadcast
```

**ग्लोबल पूलिंग क्यों आवश्यक है?**

पारंपरिक कन्वोल्यूशन केवल लोकल (3×3 रिसेप्टिव फील्ड) देखता है, कई लेयर स्टैक करने के बाद भी, वैश्विक जानकारी की धारणा सीमित रहती है। ग्लोबल पूलिंग नेटवर्क को सीधे "देखने" देता है:
- पूरे बोर्ड पर पत्थरों का अंतर
- वैश्विक प्रभाव वितरण
- समग्र स्थिति आकलन

---

## आउटपुट हेड डिज़ाइन

### Policy Head (रणनीति हेड)

प्रत्येक स्थिति की चाल प्रायिकता आउटपुट करता है:

```python
class PolicyHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2, 1)  # 1×1 कन्वोल्यूशन
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2 * 19 * 19, 362)  # 361 + pass

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.softmax(out, dim=1)  # प्रायिकता वितरण
```

**आउटपुट प्रारूप**: 362-आयामी वेक्टर
- इंडेक्स 0-360: बोर्ड पर 361 स्थानों की चाल प्रायिकता
- इंडेक्स 361: पास की प्रायिकता

### Value Head (मूल्य हेड)

वर्तमान स्थिति की जीत दर आउटपुट करता है:

```python
class ValueHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(19 * 19, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.tanh(self.fc2(out))  # -1 से +1 आउटपुट
        return out
```

**आउटपुट प्रारूप**: एकल मान [-1, +1]
- +1: अपनी निश्चित जीत
- -1: प्रतिद्वंद्वी की निश्चित जीत
- 0: समान स्थिति

### Score Head (अंक हेड)

KataGo विशेष, अंतिम अंक अंतर का पूर्वानुमान:

```python
class ScoreHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(19 * 19, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)  # अप्रतिबंधित आउटपुट
        return out
```

**आउटपुट प्रारूप**: एकल मान (अंक)
- धनात्मक संख्या: अपनी बढ़त
- ऋणात्मक संख्या: प्रतिद्वंद्वी की बढ़त

### Ownership Head (स्वामित्व हेड)

प्रत्येक बिंदु के अंतिम स्वामित्व का पूर्वानुमान:

```python
class OwnershipHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 1)
        self.bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        out = F.relu(self.bn(self.conv1(x)))
        out = torch.tanh(self.conv2(out))  # प्रत्येक बिंदु -1 से +1
        return out.view(out.size(0), -1)  # 361 में फ्लैटन करें
```

**आउटपुट प्रारूप**: 361-आयामी वेक्टर, प्रत्येक मान [-1, +1] में
- +1: यह बिंदु अपने क्षेत्र का है
- -1: यह बिंदु प्रतिद्वंद्वी के क्षेत्र का है
- 0: तटस्थ या विवादित क्षेत्र

---

## AlphaZero से अंतर

| पहलू | AlphaZero | KataGo |
|------|-----------|--------|
| **आउटपुट हेड** | 2 (Policy + Value) | **4** (+ Score + Ownership) |
| **ग्लोबल पूलिंग** | नहीं | **हाँ** |
| **इनपुट फीचर्स** | 17 प्लेन | **22 प्लेन** (नियम एनकोडिंग सहित) |
| **रेसिड्यूअल ब्लॉक** | मानक ResNet | **Pre-activation + ग्लोबल पूलिंग** |
| **मल्टी-रूल समर्थन** | नहीं | **हाँ** (फीचर एनकोडिंग के माध्यम से) |

---

## मॉडल स्केल

KataGo विभिन्न स्केल के मॉडल प्रदान करता है:

| मॉडल | रेसिड्यूअल ब्लॉक | चैनल | पैरामीटर | उपयोग परिदृश्य |
|------|---------|--------|--------|---------|
| b10c128 | 10 | 128 | ~5M | CPU, त्वरित परीक्षण |
| b18c384 | 18 | 384 | ~75M | सामान्य GPU |
| b40c256 | 40 | 256 | ~95M | उच्च-स्तरीय GPU |
| b60c320 | 60 | 320 | ~200M | टॉप-टियर GPU |

**नामकरण नियम**: `b{रेसिड्यूअल ब्लॉक संख्या}c{चैनल संख्या}`

---

## पूर्ण नेटवर्क कार्यान्वयन

```python
class KataGoNetwork(nn.Module):
    def __init__(self, num_blocks=18, channels=384):
        super().__init__()

        # प्रारंभिक कन्वोल्यूशन
        self.initial_conv = nn.Conv2d(22, channels, 3, padding=1)
        self.initial_bn = nn.BatchNorm2d(channels)

        # रेसिड्यूअल टावर
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_blocks)
        ])

        # ग्लोबल पूलिंग ब्लॉक (हर कुछ रेसिड्यूअल ब्लॉक के बाद एक डालें)
        self.global_pooling_blocks = nn.ModuleList([
            GlobalPoolingBlock(channels) for _ in range(num_blocks // 6)
        ])

        # आउटपुट हेड
        self.policy_head = PolicyHead(channels)
        self.value_head = ValueHead(channels)
        self.score_head = ScoreHead(channels)
        self.ownership_head = OwnershipHead(channels)

    def forward(self, x):
        # प्रारंभिक कन्वोल्यूशन
        out = F.relu(self.initial_bn(self.initial_conv(x)))

        # रेसिड्यूअल टावर
        gp_idx = 0
        for i, block in enumerate(self.residual_blocks):
            out = block(out)

            # हर 6 रेसिड्यूअल ब्लॉक के बाद ग्लोबल पूलिंग डालें
            if (i + 1) % 6 == 0 and gp_idx < len(self.global_pooling_blocks):
                out = self.global_pooling_blocks[gp_idx](out)
                gp_idx += 1

        # आउटपुट हेड
        policy = self.policy_head(out)
        value = self.value_head(out)
        score = self.score_head(out)
        ownership = self.ownership_head(out)

        return {
            'policy': policy,
            'value': value,
            'score': score,
            'ownership': ownership
        }
```

---

## आगे पढ़ें

- [MCTS कार्यान्वयन विवरण](../mcts-implementation) — सर्च और न्यूरल नेटवर्क का संयोजन
- [KataGo प्रशिक्षण तंत्र विश्लेषण](../training) — नेटवर्क को कैसे प्रशिक्षित करें
- [मुख्य पेपर गाइड](../papers) — मूल पेपर की गणितीय व्युत्पत्ति
