---
sidebar_position: 2
title: 畀工程師嘅圍棋 AI 指南
---

# 畀工程師嘅圍棋 AI 指南

歡迎嚟到圍棋 AI 技術文件區！呢度為想深入理解、部署或開發圍棋 AI 嘅工程師同開發者，提供完整嘅技術資源同指南。

## 本區塊內容

本區塊涵蓋以下主題：

### 背景知識
- **AlphaGo 論文解讀**：深入分析 DeepMind 嘅突破性研究，包括 Policy Network、Value Network 同 MCTS 嘅結合
- **KataGo 論文解讀**：了解目前最先進嘅開源圍棋 AI 嘅創新設計
- **其他圍棋 AI 介紹**：商業同開源圍棋 AI 嘅全面比較

### KataGo 實戰
- **安裝同設定**：由零開始喺各平台上建置 KataGo 環境
- **常用指令**：GTP 協議同 Analysis Engine 嘅實用指南
- **原始碼架構**：深入探索 KataGo 嘅程式碼結構同實作細節

## 適合邊個閱讀

本區塊適合以下讀者：

| 讀者類型 | 建議閱讀內容 |
|---------|-------------|
| **軟件工程師** | 想喺專案入面整合圍棋 AI → 由「安裝同設定」開始 |
| **機器學習工程師** | 想了解圍棋 AI 嘅演算法 → 由「AlphaGo 論文解讀」開始 |
| **研究者** | 想進行圍棋 AI 研究 → 閱讀所有背景知識然後深入原始碼架構 |
| **圍棋 App 開發者** | 想開發圍棋相關應用 → 重點閱讀「常用指令」同「Analysis Engine」 |
| **系統管理員** | 需要部署圍棋 AI 服務 → 專注於「安裝同設定」章節 |

## 學習路徑建議

根據你嘅目標，我哋建議以下學習路徑：

### 路徑 A：快速上手（1-2 日）

適合想快速部署 KataGo 嘅開發者：

1. [KataGo 安裝同設定](./katago-source/setup.md) - 建置執行環境
2. [KataGo 常用指令](./katago-source/commands.md) - 學習基本操作

### 路徑 B：深入理解（1-2 星期）

適合想完整理解圍棋 AI 技術嘅工程師：

1. [AlphaGo 論文解讀](./background-info/alphago.md) - 理解基礎架構
2. [KataGo 論文解讀](./background-info/katago-paper.md) - 了解最新改進
3. [其他圍棋 AI 介紹](./background-info/zen.md) - 認識產業生態
4. [KataGo 安裝同設定](./katago-source/setup.md) - 實際動手操作
5. [KataGo 常用指令](./katago-source/commands.md) - 深入使用功能

### 路徑 C：開發貢獻（1 個月以上）

適合想貢獻 KataGo 開源專案或開發自己嘅圍棋 AI：

1. 完成路徑 B 嘅所有內容
2. [KataGo 原始碼架構](./katago-source/architecture.md) - 深入程式碼
3. 閱讀 KataGo GitHub 上嘅 Issues 同 Pull Requests
4. 嘗試修改同實驗

## 預備知識

為咗順利閱讀本區塊內容，建議具備以下基礎知識：

- **程式設計**：熟悉至少一種程式語言（Python、C++ 尤佳）
- **機器學習基礎**：了解神經網絡、反向傳播等基本概念
- **圍棋規則**：知道圍棋嘅基本規則同術語
- **命令列操作**：熟悉終端機/命令提示字元嘅基本操作

唔具備以上知識都可以閱讀，但可能需要額外查閱相關資料。

## 開始探索

準備好未？由[背景知識](./background-info/)開始你嘅圍棋 AI 技術之旅啦！

如果你已經有機器學習背景，想快速上手，可以直接去 [KataGo 實戰入門指南](./katago-source/)。

