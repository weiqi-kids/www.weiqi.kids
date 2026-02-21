---
sidebar_position: 4
title: 參與開源社群
description: 加入 KataGo 開源社群，貢獻算力或程式碼
---

# 參與開源社群

KataGo 是一個活躍的開源專案，有多種方式可以參與貢獻。

---

## 貢獻方式總覽

| 方式 | 難度 | 需求 |
|------|------|------|
| **貢獻算力** | 低 | 有 GPU 的電腦 |
| **回報問題** | 低 | GitHub 帳號 |
| **改進文件** | 中 | 熟悉技術內容 |
| **貢獻程式碼** | 高 | C++/Python 開發能力 |

---

## 貢獻算力：分散式訓練

### KataGo Training 簡介

KataGo Training 是一個全球分散式訓練網路：

- 志願者貢獻 GPU 算力執行自我對弈
- 自我對弈資料上傳到中央伺服器
- 伺服器定期訓練新模型
- 新模型分發給志願者繼續對弈

官網：https://katagotraining.org/

### 參與步驟

#### 1. 建立帳號

前往 https://katagotraining.org/ 註冊帳號。

#### 2. 下載 KataGo

```bash
# 下載最新版本
wget https://github.com/lightvector/KataGo/releases/download/v1.15.3/katago-v1.15.3-cuda11.1-linux-x64.zip
unzip katago-v1.15.3-cuda11.1-linux-x64.zip
```

#### 3. 設定 contribute 模式

```bash
# 首次執行會引導你設定
./katago contribute -username YOUR_USERNAME -password YOUR_PASSWORD
```

系統會自動：
- 下載最新模型
- 執行自我對弈
- 上傳對弈資料

#### 4. 背景執行

```bash
# 使用 screen 或 tmux 背景執行
screen -S katago
./katago contribute -username YOUR_USERNAME -password YOUR_PASSWORD
# Ctrl+A, D 離開 screen
```

### 貢獻統計

你可以在 https://katagotraining.org/contributions/ 查看：
- 你的貢獻排名
- 總貢獻對局數
- 最近訓練的模型

---

## 回報問題

### 在哪裡回報

- **GitHub Issues**：https://github.com/lightvector/KataGo/issues
- **Discord**：https://discord.gg/bqkZAz3

### 好的問題報告包含

1. **KataGo 版本**：`katago version`
2. **作業系統**：Windows/Linux/macOS
3. **硬體**：GPU 型號、記憶體
4. **完整錯誤訊息**：複製完整 log
5. **重現步驟**：如何觸發這個問題

### 範例

```markdown
## 問題描述
執行 benchmark 時出現記憶體不足錯誤

## 環境
- KataGo 版本：1.15.3
- 作業系統：Ubuntu 22.04
- GPU：RTX 3060 12GB
- 模型：kata-b40c256.bin.gz

## 錯誤訊息
```
CUDA error: out of memory
```

## 重現步驟
1. 執行 `katago benchmark -model kata-b40c256.bin.gz`
2. 等待約 30 秒
3. 出現錯誤
```

---

## 改進文件

### 文件位置

- **README**：`README.md`
- **GTP 文件**：`docs/GTP_Extensions.md`
- **Analysis 文件**：`docs/Analysis_Engine.md`
- **訓練文件**：`python/README.md`

### 貢獻流程

1. Fork 專案
2. 建立新分支
3. 修改文件
4. 提交 Pull Request

```bash
git clone https://github.com/YOUR_USERNAME/KataGo.git
cd KataGo
git checkout -b improve-docs
# 編輯文件
git add .
git commit -m "Improve documentation for Analysis Engine"
git push origin improve-docs
# 在 GitHub 上建立 Pull Request
```

---

## 貢獻程式碼

### 開發環境設定

```bash
# 複製專案
git clone https://github.com/lightvector/KataGo.git
cd KataGo

# 編譯（Debug 模式）
cd cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=OPENCL -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# 執行測試
./katago runtests
```

### 程式碼風格

KataGo 使用以下程式碼風格：

**C++**：
- 2 空格縮排
- 大括號同行
- 變數名使用 camelCase
- 類別名使用 PascalCase

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

**Python**：
- 遵循 PEP 8
- 4 空格縮排

### 貢獻領域

| 領域 | 檔案位置 | 技能需求 |
|------|---------|---------|
| 核心引擎 | `cpp/` | C++, CUDA/OpenCL |
| 訓練程式 | `python/` | Python, PyTorch |
| GTP 協議 | `cpp/command/gtp.cpp` | C++ |
| Analysis API | `cpp/command/analysis.cpp` | C++, JSON |
| 測試 | `cpp/tests/` | C++ |

### Pull Request 流程

1. **建立 Issue**：先討論你想做的改動
2. **Fork & Clone**：建立自己的分支
3. **開發與測試**：確保所有測試通過
4. **提交 PR**：詳細描述改動內容
5. **Code Review**：回應維護者的反饋
6. **合併**：維護者合併你的程式碼

### PR 範例

```markdown
## 改動描述
新增對 New Zealand 規則的支援

## 改動內容
- 在 rules.cpp 新增 NEW_ZEALAND 規則
- 更新 GTP 指令支援 `kata-set-rules nz`
- 新增單元測試

## 測試結果
- 所有現有測試通過
- 新增測試通過

## 相關 Issue
Fixes #123
```

---

## 社群資源

### 官方連結

| 資源 | 連結 |
|------|------|
| GitHub | https://github.com/lightvector/KataGo |
| Discord | https://discord.gg/bqkZAz3 |
| 訓練網路 | https://katagotraining.org/ |

### 討論區

- **Discord**：即時討論、技術問答
- **GitHub Discussions**：長篇討論、功能提議
- **Reddit r/baduk**：一般圍棋 AI 討論

### 相關專案

| 專案 | 說明 | 連結 |
|------|------|------|
| KaTrain | 教學分析工具 | github.com/sanderland/katrain |
| Lizzie | 分析介面 | github.com/featurecat/lizzie |
| Sabaki | 棋譜編輯器 | sabaki.yichuanshen.de |
| BadukAI | 線上分析 | baduk.ai |

---

## 認可與獎勵

### 貢獻者名單

所有貢獻者都會列在：
- GitHub Contributors 頁面
- KataGo Training 貢獻排行榜

### 學習收穫

參與開源專案的收穫：
- 學習工業級 AI 系統架構
- 與全球開發者交流
- 累積開源貢獻紀錄
- 深入理解圍棋 AI 技術

---

## 延伸閱讀

- [原始碼導讀](../source-code) — 理解程式碼結構
- [訓練自己的模型](../training) — 本地訓練實驗
- [一篇文章搞懂圍棋 AI](../../how-it-works/) — 技術原理
