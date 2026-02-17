---
sidebar_position: 2
title: KataGo 實戰入門
---

# KataGo 實戰入門指南

本章節會帶你由安裝到實際使用 KataGo，涵蓋所有實用操作知識。無論你係想將 KataGo 整合到自己嘅應用程式，定係想深入研究佢嘅原始碼，呢度都係你嘅起點。

## 點解揀 KataGo？

喺眾多圍棋 AI 入面，KataGo 係目前最佳選擇，原因如下：

| 優勢 | 說明 |
|------|------|
| **棋力最強** | 喺公開測試入面持續保持最高水準 |
| **功能最全** | 目數預測、領地分析、多規則支援 |
| **完全開源** | MIT 授權，可自由使用同修改 |
| **持續更新** | 活躍嘅開發同社群支援 |
| **文件完善** | 官方文件詳盡，社群資源豐富 |
| **多平台支援** | Linux、macOS、Windows 都可以運行 |

## 本章內容

### [安裝同設定](./setup.md)

由零開始建置 KataGo 環境：

- 系統需求同硬件建議
- 各平台安裝步驟（macOS / Linux / Windows）
- 模型下載同選擇指南
- 設定檔詳細說明

### [常用指令](./commands.md)

掌握 KataGo 嘅使用方式：

- GTP（Go Text Protocol）協議介紹
- 常用 GTP 指令同範例
- Analysis Engine 使用方法
- JSON API 完整說明

### [原始碼架構](./architecture.md)

深入了解 KataGo 嘅實作細節：

- 專案目錄結構概覽
- 神經網絡架構解析
- 搜索引擎實作細節
- 訓練流程概述

## 快速開始

如果你淨係想快速試吓 KataGo，以下係最簡單嘅方式：

### macOS（使用 Homebrew）

```bash
# 安裝
brew install katago

# 下載模型（揀較細嘅模型用嚟測試）
curl -L -o kata-b18c384.bin.gz \
  https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz

# 運行 GTP 模式
katago gtp -model kata-b18c384.bin.gz -config gtp_example.cfg
```

### Linux（預編譯版）

```bash
# 下載預編譯版本
wget https://github.com/lightvector/KataGo/releases/download/v1.15.3/katago-v1.15.3-opencl-linux-x64.zip

# 解壓縮
unzip katago-v1.15.3-opencl-linux-x64.zip

# 下載模型
wget https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz

# 運行
./katago gtp -model kata-b18c384nbt-*.bin.gz -config default_gtp.cfg
```

### 驗證安裝

成功啟動之後，你會見到 GTP 提示符。試吓輸入以下指令：

```
name
= KataGo

version
= 1.15.3

boardsize 19
=

genmove black
= Q16
```

## 使用場景指南

根據你嘅需求，以下係建議嘅閱讀順序同重點：

### 場景 1：整合到圍棋 App

你想喺自己嘅圍棋應用入面使用 KataGo 作為 AI 引擎。

**重點閱讀**：
1. [安裝同設定](./setup.md) - 了解部署需求
2. [常用指令](./commands.md) - 特別係 Analysis Engine 部分

**關鍵知識**：
- 使用 Analysis Engine 模式而唔係 GTP 模式
- 透過 JSON API 同 KataGo 通訊
- 根據硬件調整搜索參數

### 場景 2：建置對弈伺服器

你想架設一個畀用戶同 AI 對弈嘅伺服器。

**重點閱讀**：
1. [安裝同設定](./setup.md) - GPU 設定部分
2. [常用指令](./commands.md) - GTP 協議部分

**關鍵知識**：
- 使用 GTP 模式進行對弈
- 多實例部署策略
- 棋力調整方法

### 場景 3：研究 AI 演算法

你想深入研究 KataGo 嘅實作，可能想修改或實驗。

**重點閱讀**：
1. [原始碼架構](./architecture.md) - 全文精讀
2. 背景知識章節嘅所有論文解讀

**關鍵知識**：
- C++ 程式碼結構
- 神經網絡架構細節
- MCTS 實作方式

### 場景 4：訓練自己嘅模型

你想由頭訓練或微調 KataGo 模型。

**重點閱讀**：
1. [原始碼架構](./architecture.md) - 訓練流程部分
2. [KataGo 論文解讀](../background-info/katago-paper.md)

**關鍵知識**：
- 訓練資料格式
- 訓練腳本使用
- 超參數設定

## 硬件建議

KataGo 可以喺各種硬件上運行，但效能差異好大：

| 硬件配置 | 預期效能 | 適用場景 |
|---------|---------|---------|
| **高階 GPU**（RTX 4090）| ~2000 playouts/sec | 頂級分析、快速搜索 |
| **中階 GPU**（RTX 3060）| ~500 playouts/sec | 一般分析、對弈 |
| **入門 GPU**（GTX 1650）| ~100 playouts/sec | 基本使用 |
| **Apple Silicon**（M1/M2）| ~200-400 playouts/sec | macOS 開發 |
| **純 CPU** | ~10-30 playouts/sec | 學習、測試 |

:::tip
即使係較慢嘅硬件，KataGo 都可以提供有價值嘅分析。搜索量減少會降低精確度，但對於教學同學習通常已經足夠。
:::

## 常見問題

### KataGo 同 Leela Zero 有乜嘢唔同？

| 面向 | KataGo | Leela Zero |
|------|--------|------------|
| 棋力 | 更強 | 較弱 |
| 功能 | 豐富（目數、領地） | 基本 |
| 多規則 | 支援 | 唔支援 |
| 開發狀態 | 活躍 | 維護模式 |
| 訓練效率 | 高 | 較低 |

### 需要 GPU 嗎？

唔係必須嘅，但強烈建議：
- **有 GPU**：可以進行快速分析，獲得高品質結果
- **冇 GPU**：可以使用 Eigen 後端，但速度較慢

### 模型檔案差異？

| 模型大細 | 檔案大細 | 棋力 | 速度 |
|---------|---------|------|------|
| b10c128 | ~20 MB | 中等 | 最快 |
| b18c384 | ~140 MB | 強 | 快 |
| b40c256 | ~250 MB | 好強 | 中 |
| b60c320 | ~500 MB | 最強 | 慢 |

通常建議使用 b18c384 或 b40c256，喺棋力同速度之間取得平衡。

## 相關資源

- [KataGo GitHub](https://github.com/lightvector/KataGo)
- [KataGo 訓練網站](https://katagotraining.org/)
- [KataGo Discord 社群](https://discord.gg/bqkZAz3)
- [Lizzie](https://github.com/featurecat/lizzie) - 配合 KataGo 使用嘅 GUI

準備好未？等我哋由[安裝同設定](./setup.md)開始啦！

