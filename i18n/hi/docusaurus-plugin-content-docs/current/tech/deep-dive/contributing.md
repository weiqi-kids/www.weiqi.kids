---
sidebar_position: 4
title: ओपन सोर्स समुदाय में भागीदारी
description: KataGo ओपन सोर्स समुदाय में शामिल हों, कम्प्यूटिंग पावर या कोड योगदान करें
---

# ओपन सोर्स समुदाय में भागीदारी

KataGo एक सक्रिय ओपन सोर्स प्रोजेक्ट है, जिसमें योगदान करने के कई तरीके हैं।

---

## योगदान विधि अवलोकन

| विधि | कठिनाई | आवश्यकताएं |
|------|------|------|
| **कम्प्यूटिंग पावर योगदान** | कम | GPU वाला कंप्यूटर |
| **समस्या रिपोर्ट करें** | कम | GitHub खाता |
| **डॉक्यूमेंटेशन सुधारें** | मध्यम | तकनीकी सामग्री से परिचित |
| **कोड योगदान** | उच्च | C++/Python विकास क्षमता |

---

## कम्प्यूटिंग पावर योगदान: वितरित प्रशिक्षण

### KataGo Training परिचय

KataGo Training एक वैश्विक वितरित प्रशिक्षण नेटवर्क है:

- स्वयंसेवक GPU कम्प्यूटिंग पावर से सेल्फ-प्ले निष्पादित करते हैं
- सेल्फ-प्ले डेटा केंद्रीय सर्वर पर अपलोड होता है
- सर्वर नियमित रूप से नया मॉडल प्रशिक्षित करता है
- नया मॉडल स्वयंसेवकों को वितरित होता है जारी सेल्फ-प्ले के लिए

आधिकारिक वेबसाइट: https://katagotraining.org/

### भागीदारी चरण

#### 1. खाता बनाएं

https://katagotraining.org/ पर रजिस्टर करें।

#### 2. KataGo डाउनलोड करें

```bash
# नवीनतम संस्करण डाउनलोड करें
wget https://github.com/lightvector/KataGo/releases/download/v1.15.3/katago-v1.15.3-cuda11.1-linux-x64.zip
unzip katago-v1.15.3-cuda11.1-linux-x64.zip
```

#### 3. contribute मोड सेटअप करें

```bash
# पहली बार चलाने पर सेटअप गाइड मिलेगी
./katago contribute -username YOUR_USERNAME -password YOUR_PASSWORD
```

सिस्टम स्वचालित रूप से:
- नवीनतम मॉडल डाउनलोड करेगा
- सेल्फ-प्ले निष्पादित करेगा
- गेम डेटा अपलोड करेगा

#### 4. बैकग्राउंड में चलाएं

```bash
# screen या tmux से बैकग्राउंड में चलाएं
screen -S katago
./katago contribute -username YOUR_USERNAME -password YOUR_PASSWORD
# Ctrl+A, D से screen छोड़ें
```

### योगदान सांख्यिकी

आप https://katagotraining.org/contributions/ पर देख सकते हैं:
- आपकी योगदान रैंकिंग
- कुल योगदान किए गए गेम
- हाल ही में प्रशिक्षित मॉडल

---

## समस्या रिपोर्ट करें

### कहाँ रिपोर्ट करें

- **GitHub Issues**: https://github.com/lightvector/KataGo/issues
- **Discord**: https://discord.gg/bqkZAz3

### अच्छी समस्या रिपोर्ट में शामिल

1. **KataGo संस्करण**: `katago version`
2. **ऑपरेटिंग सिस्टम**: Windows/Linux/macOS
3. **हार्डवेयर**: GPU मॉडल, मेमोरी
4. **पूर्ण एरर मैसेज**: पूरा लॉग कॉपी करें
5. **रीप्रोड्यूस चरण**: समस्या कैसे ट्रिगर करें

### उदाहरण

```markdown
## समस्या विवरण
benchmark चलाते समय मेमोरी अपर्याप्त एरर

## वातावरण
- KataGo संस्करण: 1.15.3
- ऑपरेटिंग सिस्टम: Ubuntu 22.04
- GPU: RTX 3060 12GB
- मॉडल: kata-b40c256.bin.gz

## एरर मैसेज
```
CUDA error: out of memory
```

## रीप्रोड्यूस चरण
1. `katago benchmark -model kata-b40c256.bin.gz` चलाएं
2. लगभग 30 सेकंड प्रतीक्षा करें
3. एरर दिखाई देती है
```

---

## डॉक्यूमेंटेशन सुधारें

### डॉक्यूमेंटेशन स्थान

- **README**: `README.md`
- **GTP डॉक्यूमेंटेशन**: `docs/GTP_Extensions.md`
- **Analysis डॉक्यूमेंटेशन**: `docs/Analysis_Engine.md`
- **प्रशिक्षण डॉक्यूमेंटेशन**: `python/README.md`

### योगदान प्रक्रिया

1. प्रोजेक्ट Fork करें
2. नई ब्रांच बनाएं
3. डॉक्यूमेंटेशन संशोधित करें
4. Pull Request सबमिट करें

```bash
git clone https://github.com/YOUR_USERNAME/KataGo.git
cd KataGo
git checkout -b improve-docs
# डॉक्यूमेंटेशन संपादित करें
git add .
git commit -m "Improve documentation for Analysis Engine"
git push origin improve-docs
# GitHub पर Pull Request बनाएं
```

---

## कोड योगदान

### विकास वातावरण सेटअप

```bash
# प्रोजेक्ट क्लोन करें
git clone https://github.com/lightvector/KataGo.git
cd KataGo

# कंपाइल करें (Debug मोड)
cd cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=OPENCL -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# टेस्ट चलाएं
./katago runtests
```

### कोड स्टाइल

KataGo निम्न कोड स्टाइल का उपयोग करता है:

**C++**:
- 2 स्पेस इंडेंटेशन
- ब्रेस समान लाइन पर
- वेरिएबल नाम camelCase
- क्लास नाम PascalCase

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
- PEP 8 का पालन
- 4 स्पेस इंडेंटेशन

### योगदान क्षेत्र

| क्षेत्र | फ़ाइल स्थान | स्किल आवश्यकताएं |
|------|---------|---------|
| कोर इंजन | `cpp/` | C++, CUDA/OpenCL |
| प्रशिक्षण प्रोग्राम | `python/` | Python, PyTorch |
| GTP प्रोटोकॉल | `cpp/command/gtp.cpp` | C++ |
| Analysis API | `cpp/command/analysis.cpp` | C++, JSON |
| टेस्ट | `cpp/tests/` | C++ |

### Pull Request प्रक्रिया

1. **Issue बनाएं**: पहले अपना प्रस्तावित बदलाव चर्चा करें
2. **Fork & Clone**: अपनी ब्रांच बनाएं
3. **विकास और परीक्षण**: सुनिश्चित करें सभी टेस्ट पास हों
4. **PR सबमिट करें**: बदलाव का विस्तृत विवरण दें
5. **Code Review**: मेंटेनर की प्रतिक्रिया का जवाब दें
6. **Merge**: मेंटेनर आपका कोड मर्ज करेगा

### PR उदाहरण

```markdown
## बदलाव विवरण
New Zealand नियमों के लिए समर्थन जोड़ें

## बदलाव सामग्री
- rules.cpp में NEW_ZEALAND नियम जोड़ें
- GTP कमांड `kata-set-rules nz` समर्थन अपडेट करें
- यूनिट टेस्ट जोड़ें

## परीक्षण परिणाम
- सभी मौजूदा टेस्ट पास
- नए टेस्ट पास

## संबंधित Issue
Fixes #123
```

---

## समुदाय संसाधन

### आधिकारिक लिंक

| संसाधन | लिंक |
|------|------|
| GitHub | https://github.com/lightvector/KataGo |
| Discord | https://discord.gg/bqkZAz3 |
| प्रशिक्षण नेटवर्क | https://katagotraining.org/ |

### चर्चा मंच

- **Discord**: तत्काल चर्चा, तकनीकी प्रश्नोत्तर
- **GitHub Discussions**: लंबी चर्चा, फीचर प्रस्ताव
- **Reddit r/baduk**: सामान्य गो AI चर्चा

### संबंधित प्रोजेक्ट

| प्रोजेक्ट | विवरण | लिंक |
|------|------|------|
| KaTrain | शिक्षण विश्लेषण टूल | github.com/sanderland/katrain |
| Lizzie | विश्लेषण इंटरफेस | github.com/featurecat/lizzie |
| Sabaki | गेम रिकॉर्ड एडिटर | sabaki.yichuanshen.de |
| BadukAI | ऑनलाइन विश्लेषण | baduk.ai |

---

## मान्यता और पुरस्कार

### योगदानकर्ता सूची

सभी योगदानकर्ता इन पर सूचीबद्ध होंगे:
- GitHub Contributors पेज
- KataGo Training योगदान लीडरबोर्ड

### सीखने के लाभ

ओपन सोर्स प्रोजेक्ट में भाग लेने के लाभ:
- इंडस्ट्री-ग्रेड AI सिस्टम आर्किटेक्चर सीखें
- वैश्विक डेवलपर्स के साथ बातचीत
- ओपन सोर्स योगदान रिकॉर्ड जमा करें
- गो AI तकनीक की गहन समझ

---

## आगे पढ़ें

- [स्रोत कोड गाइड](../source-code) — कोड संरचना समझें
- [KataGo प्रशिक्षण तंत्र विश्लेषण](../training) — लोकल प्रशिक्षण प्रयोग
- [एक लेख में गो AI समझें](../../how-it-works/) — तकनीकी सिद्धांत
