---
sidebar_position: 1
title: 30 分鐘跑起第一個圍棋 AI
description: 快速上手 KataGo，從安裝到對弈只需 30 分鐘
---

# 30 分鐘跑起第一個圍棋 AI

這份教學將帶你快速安裝並運行 KataGo。完成後你將能夠：

- ✅ 在終端機與 KataGo 對弈
- ✅ 分析一盤棋的每步棋勝率
- ✅ 理解基本的 GTP 協議

---

## 步驟 1：安裝 KataGo（5 分鐘）

### macOS

```bash
# 使用 Homebrew 安裝（最簡單）
brew install katago

# 確認安裝成功
katago version
# 輸出：1.15.3 或更新版本
```

### Linux

```bash
# 下載預編譯版本
wget https://github.com/lightvector/KataGo/releases/download/v1.15.3/katago-v1.15.3-opencl-linux-x64.zip

# 解壓縮
unzip katago-v1.15.3-opencl-linux-x64.zip

# 賦予執行權限
chmod +x katago

# 確認安裝
./katago version
```

### Windows

1. 前往 [KataGo Releases](https://github.com/lightvector/KataGo/releases)
2. 下載 `katago-v1.15.3-opencl-windows-x64.zip`
3. 解壓縮到任意目錄
4. 在命令提示字元中測試：`katago.exe version`

---

## 步驟 2：下載模型（2 分鐘）

KataGo 需要神經網路模型檔案。下載推薦的 b18c384 模型：

```bash
# macOS / Linux
curl -L -o kata-b18c384.bin.gz \
  "https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz"

# Windows（PowerShell）
Invoke-WebRequest -Uri "https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz" -OutFile "kata-b18c384.bin.gz"
```

**模型大小說明**：

| 模型 | 檔案大小 | 棋力 | 速度 |
|------|---------|------|------|
| b10c128 | ~20 MB | 中等 | 最快 |
| **b18c384** | ~140 MB | 強 | 快 |
| b40c256 | ~250 MB | 很強 | 中 |

---

## 步驟 3：第一次對弈（10 分鐘）

### 啟動 GTP 模式

```bash
# macOS (Homebrew)
katago gtp -model kata-b18c384.bin.gz

# Linux / Windows
./katago gtp -model kata-b18c384.bin.gz
```

### 基本對弈指令

啟動後，你會看到 KataGo 等待輸入。試試以下指令：

```
name
= KataGo

version
= 1.15.3

boardsize 9
=

komi 7.5
=

play black E5
=

genmove white
= C3

showboard
=
   A B C D E F G H J
 9 . . . . . . . . . 9
 8 . . . . . . . . . 8
 7 . . . . . . . . . 7
 6 . . . . . . . . . 6
 5 . . . . X . . . . 5
 4 . . . . . . . . . 4
 3 . . O . . . . . . 3
 2 . . . . . . . . . 2
 1 . . . . . . . . . 1
   A B C D E F G H J
```

### 常用指令速查

| 指令 | 功能 | 範例 |
|------|------|------|
| `boardsize N` | 設定棋盤大小 | `boardsize 19` |
| `komi N` | 設定貼目 | `komi 7.5` |
| `play COLOR COORD` | 下一步棋 | `play black Q16` |
| `genmove COLOR` | AI 下一步 | `genmove white` |
| `showboard` | 顯示棋盤 | |
| `undo` | 悔棋 | |
| `clear_board` | 清空棋盤 | |
| `quit` | 退出 | |

---

## 步驟 4：分析一盤棋（10 分鐘）

### 使用 kata-analyze

`kata-analyze` 指令可以分析當前局面：

```
boardsize 19
=

play black Q16
=

play white D4
=

play black Q4
=

kata-analyze black 500 100
```

輸出解讀：

```
info move D16 visits 234 winrate 0.5432 scoreMean 2.31 prior 0.1234 pv D16 R14 D10
info move R14 visits 156 winrate 0.5312 scoreMean 1.82 prior 0.0987 pv R14 D16 R10
```

| 欄位 | 意義 |
|------|------|
| `move` | 建議的下法 |
| `visits` | 搜索次數（越多越可信） |
| `winrate` | 勝率（0.54 = 54%） |
| `scoreMean` | 預期贏幾目 |
| `prior` | 神經網路的直覺機率 |
| `pv` | 預測的後續變化 |

### 使用 Analysis Engine

Analysis Engine 使用 JSON 格式，更適合程式化使用：

```bash
katago analysis -model kata-b18c384.bin.gz
```

輸入（一行 JSON）：

```json
{"id":"test1","moves":[["B","Q16"],["W","D4"]],"rules":"chinese","komi":7.5,"boardXSize":19,"boardYSize":19,"analyzeTurns":[2]}
```

輸出（一行 JSON）：

```json
{"id":"test1","turnNumber":2,"moveInfos":[{"move":"Q4","visits":234,"winrate":0.5432,...}],"rootInfo":{"winrate":0.52,...}}
```

---

## 步驟 5：驗證效能

運行基準測試確認硬體效能：

```bash
katago benchmark -model kata-b18c384.bin.gz -v 500
```

輸出範例：

```
Testing with 500 visits...
Visits/s: 342.5
Neural net evals/s: 187.3
Recommended numSearchThreads: 4
```

**效能參考**：

| 硬體 | 預期效能 |
|------|---------|
| RTX 3060 | ~500 visits/s |
| RTX 4080 | ~1500 visits/s |
| Apple M1 | ~200-400 visits/s |
| 純 CPU | ~10-30 visits/s |

---

## 常見問題

### 找不到 GPU

```bash
# 列出可用的 GPU
katago gpuinfo
```

如果沒有顯示 GPU，可能需要安裝 OpenCL 驅動。

### 模型路徑錯誤

確認模型檔案路徑正確：

```bash
# 使用絕對路徑
katago gtp -model /full/path/to/kata-b18c384.bin.gz
```

### 記憶體不足

使用較小的模型（b10c128）或調整設定：

```bash
katago gtp -model model.bin.gz -override-config nnMaxBatchSize=8
```

---

## 下一步

恭喜！你已經成功運行 KataGo。接下來可以：

- [完整安裝指南](./setup) — 進階設定與編譯
- [基本使用](./basic-usage) — GTP 與 Analysis Engine 詳解
- [整合到你的專案](./integration) — Python/Node.js API
- [一篇文章搞懂圍棋 AI](../how-it-works/) — 了解技術原理
