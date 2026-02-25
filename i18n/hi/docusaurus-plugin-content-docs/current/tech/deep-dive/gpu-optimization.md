---
sidebar_position: 6
title: GPU बैकएंड और अनुकूलन
description: KataGo के CUDA, OpenCL, Metal बैकएंड की तुलना और प्रदर्शन ट्यूनिंग गाइड
---

# GPU बैकएंड और अनुकूलन

यह लेख KataGo द्वारा समर्थित विभिन्न GPU बैकएंड, प्रदर्शन अंतर, और इष्टतम प्रदर्शन प्राप्त करने के लिए ट्यूनिंग का परिचय देता है।

---

## बैकएंड अवलोकन

KataGo चार कम्प्यूटेशन बैकएंड का समर्थन करता है:

| बैकएंड | हार्डवेयर समर्थन | प्रदर्शन | इंस्टॉल कठिनाई |
|------|---------|------|---------|
| **CUDA** | NVIDIA GPU | सर्वश्रेष्ठ | मध्यम |
| **OpenCL** | NVIDIA/AMD/Intel GPU | अच्छा | सरल |
| **Metal** | Apple Silicon | अच्छा | सरल |
| **Eigen** | शुद्ध CPU | धीमा | सबसे सरल |

---

## CUDA बैकएंड

### उपयोग परिदृश्य

- NVIDIA GPU (GTX 10 सीरीज़ और ऊपर)
- उच्चतम प्रदर्शन की आवश्यकता
- CUDA विकास वातावरण उपलब्ध

### इंस्टॉलेशन आवश्यकताएँ

```bash
# CUDA संस्करण जांचें
nvcc --version

# cuDNN जांचें
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR
```

| घटक | सुझाया संस्करण |
|------|---------|
| CUDA | 11.x या 12.x |
| cuDNN | 8.x |
| ड्राइवर | 470+ |

### कंपाइल करें

```bash
cd KataGo/cpp
mkdir build && cd build

cmake .. -DUSE_BACKEND=CUDA \
         -DCUDNN_INCLUDE_DIR=/usr/local/cuda/include \
         -DCUDNN_LIBRARY=/usr/local/cuda/lib64/libcudnn.so

make -j$(nproc)
```

### प्रदर्शन विशेषताएं

- **Tensor Cores**: FP16 त्वरण समर्थन (RTX सीरीज़)
- **बैच इन्फरेंस**: GPU उपयोग सबसे अच्छा
- **मेमोरी प्रबंधन**: VRAM उपयोग का सूक्ष्म नियंत्रण

---

## OpenCL बैकएंड

### उपयोग परिदृश्य

- AMD GPU
- Intel इंटीग्रेटेड ग्राफिक्स
- NVIDIA GPU (बिना CUDA वातावरण)
- क्रॉस-प्लेटफॉर्म डिप्लॉयमेंट

### इंस्टॉलेशन आवश्यकताएँ

```bash
# Linux - OpenCL डेवलपमेंट किट इंस्टॉल करें
sudo apt install ocl-icd-opencl-dev

# उपलब्ध OpenCL डिवाइस जांचें
clinfo
```

### कंपाइल करें

```bash
cmake .. -DUSE_BACKEND=OPENCL
make -j$(nproc)
```

### ड्राइवर चयन

| GPU प्रकार | सुझाया ड्राइवर |
|---------|---------|
| AMD | ROCm या AMDGPU-PRO |
| Intel | intel-opencl-icd |
| NVIDIA | nvidia-opencl-icd |

### प्रदर्शन ट्यूनिंग

```ini
# config.cfg
openclDeviceToUse = 0          # GPU नंबर
openclUseFP16 = auto           # हाफ प्रिसिजन (समर्थित होने पर)
openclUseFP16Storage = true    # FP16 स्टोरेज
```

---

## Metal बैकएंड

### उपयोग परिदृश्य

- Apple Silicon (M1/M2/M3)
- macOS सिस्टम

### कंपाइल करें

```bash
cmake .. -DUSE_BACKEND=METAL
make -j$(sysctl -n hw.ncpu)
```

### Apple Silicon ऑप्टिमाइज़ेशन

Apple Silicon का यूनिफाइड मेमोरी आर्किटेक्चर विशेष लाभ देता है:

```ini
# Apple Silicon सुझाई सेटिंग
numNNServerThreadsPerModel = 1
nnMaxBatchSize = 16
numSearchThreads = 4
```

### प्रदर्शन तुलना

| चिप | सापेक्ष प्रदर्शन |
|------|---------|
| M1 | ~RTX 2060 |
| M1 Pro/Max | ~RTX 3060 |
| M2 | ~RTX 2070 |
| M3 Pro/Max | ~RTX 3070 |

---

## Eigen बैकएंड (शुद्ध CPU)

### उपयोग परिदृश्य

- GPU रहित वातावरण
- त्वरित परीक्षण
- हल्का उपयोग

### कंपाइल करें

```bash
sudo apt install libeigen3-dev
cmake .. -DUSE_BACKEND=EIGEN
make -j$(nproc)
```

### प्रदर्शन अपेक्षाएं

```
CPU सिंगल-कोर: ~10-30 playouts/sec
CPU मल्टी-कोर: ~50-150 playouts/sec
GPU (मध्यम): ~1000-3000 playouts/sec
```

---

## प्रदर्शन ट्यूनिंग पैरामीटर

### मूल पैरामीटर

```ini
# config.cfg

# === न्यूरल नेटवर्क सेटिंग ===
# GPU नंबर (मल्टी GPU के लिए)
nnDeviceIdxs = 0

# प्रति मॉडल इन्फरेंस थ्रेड
numNNServerThreadsPerModel = 2

# अधिकतम बैच साइज़
nnMaxBatchSize = 16

# कैश साइज़ (2^N पोजीशन)
nnCacheSizePowerOfTwo = 20

# === सर्च सेटिंग ===
# सर्च थ्रेड्स
numSearchThreads = 8

# प्रति चाल अधिकतम विज़िट
maxVisits = 800
```

### पैरामीटर ट्यूनिंग गाइड

#### nnMaxBatchSize

```
बहुत छोटा: GPU उपयोग कम, इन्फरेंस विलंबता अधिक
बहुत बड़ा: VRAM अपर्याप्त, प्रतीक्षा समय लंबा

सुझाए मान:
- 4GB VRAM: 8-12
- 8GB VRAM: 16-24
- 16GB+ VRAM: 32-64
```

#### numSearchThreads

```
बहुत कम: GPU को फीड नहीं कर सकता
बहुत अधिक: CPU बॉटलनेक, मेमोरी दबाव

सुझाए मान:
- CPU कोर की 1-2 गुना
- nnMaxBatchSize के करीब
```

#### numNNServerThreadsPerModel

```
CUDA: 1-2
OpenCL: 1-2
Eigen: CPU कोर संख्या
```

### मेमोरी ट्यूनिंग

```ini
# VRAM उपयोग कम करें
nnMaxBatchSize = 8
nnCacheSizePowerOfTwo = 18

# VRAM उपयोग बढ़ाएं (प्रदर्शन बढ़ाएं)
nnMaxBatchSize = 32
nnCacheSizePowerOfTwo = 22
```

---

## मल्टी GPU सेटअप

### सिंगल मशीन मल्टी GPU

```ini
# GPU 0 और GPU 1 का उपयोग
nnDeviceIdxs = 0,1

# प्रति GPU थ्रेड्स
numNNServerThreadsPerModel = 2
```

### लोड बैलेंसिंग

```ini
# GPU प्रदर्शन के अनुसार वेट आवंटित करें
# GPU 0 अधिक शक्तिशाली, अधिक काम आवंटित करें
nnDeviceIdxs = 0,0,1
```

---

## बेंचमार्क टेस्टिंग

### बेंचमार्क चलाएं

```bash
katago benchmark -model model.bin.gz -config config.cfg
```

### आउटपुट व्याख्या

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

### सामान्य प्रदर्शन डेटा

| GPU | मॉडल | Playouts/सेकंड |
|-----|------|------------|
| RTX 3060 | b18c384 | ~2500 |
| RTX 3080 | b18c384 | ~4500 |
| RTX 4090 | b18c384 | ~8000 |
| M1 Pro | b18c384 | ~1500 |
| M2 Max | b18c384 | ~2200 |

---

## TensorRT त्वरण

### उपयोग परिदृश्य

- NVIDIA GPU
- चरम प्रदर्शन चाहिए
- लंबे इनिशियलाइज़ेशन समय स्वीकार्य

### सक्षम करने का तरीका

```bash
# कंपाइल करते समय सक्षम करें
cmake .. -DUSE_BACKEND=CUDA -DUSE_TENSORRT=ON

# या प्री-कंपाइल्ड संस्करण
katago-tensorrt
```

### प्रदर्शन वृद्धि

```
मानक CUDA: 100%
TensorRT FP32: +20-30%
TensorRT FP16: +50-80% (RTX सीरीज़)
TensorRT INT8: +100-150% (कैलिब्रेशन आवश्यक)
```

### नोट्स

- पहली स्टार्टअप पर TensorRT इंजन कंपाइल करना होगा (कुछ मिनट)
- अलग GPU के लिए फिर से कंपाइल करना होगा
- FP16/INT8 से सटीकता थोड़ी कम हो सकती है

---

## आम समस्याएं

### GPU नहीं पहचाना गया

```bash
# GPU स्थिति जांचें
nvidia-smi  # NVIDIA
rocm-smi    # AMD
clinfo      # OpenCL

# KataGo उपलब्ध GPU सूची
katago gpuinfo
```

### VRAM अपर्याप्त

```ini
# छोटा मॉडल उपयोग करें
# b18c384 → b10c128

# बैच साइज़ कम करें
nnMaxBatchSize = 4

# कैश कम करें
nnCacheSizePowerOfTwo = 16
```

### अपेक्षित प्रदर्शन नहीं

1. सही बैकएंड उपयोग हो रहा है सुनिश्चित करें (CUDA > OpenCL > Eigen)
2. जांचें `numSearchThreads` पर्याप्त है
3. सुनिश्चित करें GPU अन्य प्रोग्राम द्वारा उपयोग नहीं हो रहा
4. `benchmark` कमांड से प्रदर्शन सत्यापित करें

---

## प्रदर्शन ऑप्टिमाइज़ेशन चेकलिस्ट

- [ ] सही बैकएंड चुनें (CUDA/OpenCL/Metal)
- [ ] नवीनतम GPU ड्राइवर इंस्टॉल करें
- [ ] `nnMaxBatchSize` को VRAM के अनुसार समायोजित करें
- [ ] `numSearchThreads` को CPU के अनुसार समायोजित करें
- [ ] `benchmark` चलाकर प्रदर्शन सत्यापित करें
- [ ] GPU उपयोग मॉनिटर करें (> 80% होना चाहिए)

---

## आगे पढ़ें

- [MCTS कार्यान्वयन विवरण](../mcts-implementation) — बैच इन्फरेंस की आवश्यकता का स्रोत
- [मॉडल क्वांटाइज़ेशन और डिप्लॉयमेंट](../quantization-deploy) — और प्रदर्शन ऑप्टिमाइज़ेशन
- [पूर्ण इंस्टॉलेशन गाइड](../../hands-on/setup) — विभिन्न प्लेटफॉर्म इंस्टॉलेशन चरण
