---
sidebar_position: 2
title: 給工程師的圍棋 AI 指南
---

# 給工程師的圍棋 AI 指南

歡迎來到圍棋 AI 技術文件區！這裡為想要深入理解、部署或開發圍棋 AI 的工程師與開發者，提供完整的技術資源與指南。

## 本區塊內容

本區塊涵蓋以下主題：

### 背景知識
- **AlphaGo 論文解讀**：深入分析 DeepMind 的突破性研究，包括 Policy Network、Value Network 與 MCTS 的結合
- **KataGo 論文解讀**：了解目前最先進的開源圍棋 AI 的創新設計
- **其他圍棋 AI 介紹**：商業與開源圍棋 AI 的全面比較

### KataGo 實戰
- **安裝與設定**：從零開始在各平台上建置 KataGo 環境
- **常用指令**：GTP 協議與 Analysis Engine 的實用指南
- **原始碼架構**：深入探索 KataGo 的程式碼結構與實作細節

## 適合誰閱讀

本區塊適合以下讀者：

| 讀者類型 | 建議閱讀內容 |
|---------|-------------|
| **軟體工程師** | 想在專案中整合圍棋 AI → 從「安裝與設定」開始 |
| **機器學習工程師** | 想了解圍棋 AI 的演算法 → 從「AlphaGo 論文解讀」開始 |
| **研究者** | 想進行圍棋 AI 研究 → 閱讀所有背景知識後深入原始碼架構 |
| **圍棋 App 開發者** | 想開發圍棋相關應用 → 重點閱讀「常用指令」與「Analysis Engine」 |
| **系統管理員** | 需要部署圍棋 AI 服務 → 專注於「安裝與設定」章節 |

## 學習路徑建議

根據你的目標，我們建議以下學習路徑：

### 路徑 A：快速上手（1-2 天）

適合想要快速部署 KataGo 的開發者：

1. [KataGo 安裝與設定](./katago-source/setup.md) - 建置執行環境
2. [KataGo 常用指令](./katago-source/commands.md) - 學習基本操作

### 路徑 B：深入理解（1-2 週）

適合想要完整理解圍棋 AI 技術的工程師：

1. [AlphaGo 論文解讀](./background-info/alphago.md) - 理解基礎架構
2. [KataGo 論文解讀](./background-info/katago-paper.md) - 了解最新改進
3. [其他圍棋 AI 介紹](./background-info/zen.md) - 認識產業生態
4. [KataGo 安裝與設定](./katago-source/setup.md) - 實際動手操作
5. [KataGo 常用指令](./katago-source/commands.md) - 深入使用功能

### 路徑 C：開發貢獻（1 個月以上）

適合想要貢獻 KataGo 開源專案或開發自己的圍棋 AI：

1. 完成路徑 B 的所有內容
2. [KataGo 原始碼架構](./katago-source/architecture.md) - 深入程式碼
3. 閱讀 KataGo GitHub 上的 Issues 與 Pull Requests
4. 嘗試修改與實驗

## 預備知識

為了順利閱讀本區塊內容，建議具備以下基礎知識：

- **程式設計**：熟悉至少一種程式語言（Python、C++ 尤佳）
- **機器學習基礎**：了解神經網路、反向傳播等基本概念
- **圍棋規則**：知道圍棋的基本規則與術語
- **命令列操作**：熟悉終端機/命令提示字元的基本操作

不具備以上知識也可以閱讀，但可能需要額外查閱相關資料。

## 開始探索

準備好了嗎？從[背景知識](./background-info/)開始你的圍棋 AI 技術之旅吧！

如果你已經有機器學習背景，想要快速上手，可以直接前往 [KataGo 實戰入門指南](./katago-source/)。
