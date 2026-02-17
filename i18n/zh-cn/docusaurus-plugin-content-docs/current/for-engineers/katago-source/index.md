---
sidebar_position: 2
title: KataGo 实战入门
---

# KataGo 实战入门指南

本章节将带你从安装到实际使用 KataGo，涵盖所有实用操作知识。无论你是想将 KataGo 集成到自己的应用程序，还是想深入研究其源代码，这里都是你的起点。

## 为什么选择 KataGo？

在众多围棋 AI 中，KataGo 是目前最佳选择，原因如下：

| 优势 | 说明 |
|------|------|
| **棋力最强** | 在公开测试中持续保持最高水准 |
| **功能最全** | 目数预测、领地分析、多规则支持 |
| **完全开源** | MIT 授权，可自由使用和修改 |
| **持续更新** | 活跃的开发和社区支持 |
| **文档完善** | 官方文档详尽，社区资源丰富 |
| **多平台支持** | Linux、macOS、Windows 皆可运行 |

## 本章内容

### [安装与设置](./setup.md)

从零开始搭建 KataGo 环境：

- 系统需求与硬件建议
- 各平台安装步骤（macOS / Linux / Windows）
- 模型下载与选择指南
- 配置文件详细说明

### [常用指令](./commands.md)

掌握 KataGo 的使用方式：

- GTP（Go Text Protocol）协议介绍
- 常用 GTP 指令与示例
- Analysis Engine 使用方法
- JSON API 完整说明

### [源代码架构](./architecture.md)

深入了解 KataGo 的实现细节：

- 项目目录结构概览
- 神经网络架构解析
- 搜索引擎实现细节
- 训练流程概述

## 快速开始

如果你只是想快速尝试 KataGo，以下是最简单的方式：

### macOS（使用 Homebrew）

```bash
# 安装
brew install katago

# 下载模型（选择较小的模型用于测试）
curl -L -o kata-b18c384.bin.gz \
  https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz

# 运行 GTP 模式
katago gtp -model kata-b18c384.bin.gz -config gtp_example.cfg
```

### Linux（预编译版）

```bash
# 下载预编译版本
wget https://github.com/lightvector/KataGo/releases/download/v1.15.3/katago-v1.15.3-opencl-linux-x64.zip

# 解压缩
unzip katago-v1.15.3-opencl-linux-x64.zip

# 下载模型
wget https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz

# 运行
./katago gtp -model kata-b18c384nbt-*.bin.gz -config default_gtp.cfg
```

### 验证安装

成功启动后，你会看到 GTP 提示符。试着输入以下指令：

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

## 使用场景指南

根据你的需求，以下是建议的阅读顺序和重点：

### 场景 1：集成到围棋 App

你想要在自己的围棋应用中使用 KataGo 作为 AI 引擎。

**重点阅读**：
1. [安装与设置](./setup.md) - 了解部署需求
2. [常用指令](./commands.md) - 特别是 Analysis Engine 部分

**关键知识**：
- 使用 Analysis Engine 模式而非 GTP 模式
- 通过 JSON API 与 KataGo 通信
- 根据硬件调整搜索参数

### 场景 2：搭建对弈服务器

你想要架设一个让用户与 AI 对弈的服务器。

**重点阅读**：
1. [安装与设置](./setup.md) - GPU 设置部分
2. [常用指令](./commands.md) - GTP 协议部分

**关键知识**：
- 使用 GTP 模式进行对弈
- 多实例部署策略
- 棋力调整方法

### 场景 3：研究 AI 算法

你想要深入研究 KataGo 的实现，可能想要修改或实验。

**重点阅读**：
1. [源代码架构](./architecture.md) - 全文精读
2. 背景知识章节的所有论文解读

**关键知识**：
- C++ 代码结构
- 神经网络架构细节
- MCTS 实现方式

### 场景 4：训练自己的模型

你想要从头训练或微调 KataGo 模型。

**重点阅读**：
1. [源代码架构](./architecture.md) - 训练流程部分
2. [KataGo 论文解读](../background-info/katago-paper.md)

**关键知识**：
- 训练数据格式
- 训练脚本使用
- 超参数设置

## 硬件建议

KataGo 可以在各种硬件上运行，但性能差异很大：

| 硬件配置 | 预期性能 | 适用场景 |
|---------|---------|---------|
| **高端 GPU**（RTX 4090）| ~2000 playouts/sec | 顶级分析、快速搜索 |
| **中端 GPU**（RTX 3060）| ~500 playouts/sec | 一般分析、对弈 |
| **入门 GPU**（GTX 1650）| ~100 playouts/sec | 基本使用 |
| **Apple Silicon**（M1/M2）| ~200-400 playouts/sec | macOS 开发 |
| **纯 CPU** | ~10-30 playouts/sec | 学习、测试 |

:::tip
即使是较慢的硬件，KataGo 也能提供有价值的分析。搜索量减少会降低精确度，但对于教学和学习通常已经足够。
:::

## 常见问题

### KataGo 与 Leela Zero 有什么不同？

| 方面 | KataGo | Leela Zero |
|------|--------|------------|
| 棋力 | 更强 | 较弱 |
| 功能 | 丰富（目数、领地） | 基本 |
| 多规则 | 支持 | 不支持 |
| 开发状态 | 活跃 | 维护模式 |
| 训练效率 | 高 | 较低 |

### 需要 GPU 吗？

不是必须的，但强烈建议：
- **有 GPU**：可以进行快速分析，获得高品质结果
- **无 GPU**：可以使用 Eigen 后端，但速度较慢

### 模型文件差异？

| 模型大小 | 文件大小 | 棋力 | 速度 |
|---------|---------|------|------|
| b10c128 | ~20 MB | 中等 | 最快 |
| b18c384 | ~140 MB | 强 | 快 |
| b40c256 | ~250 MB | 很强 | 中 |
| b60c320 | ~500 MB | 最强 | 慢 |

通常建议使用 b18c384 或 b40c256，在棋力和速度之间取得平衡。

## 相关资源

- [KataGo GitHub](https://github.com/lightvector/KataGo)
- [KataGo 训练网站](https://katagotraining.org/)
- [KataGo Discord 社区](https://discord.gg/bqkZAz3)
- [Lizzie](https://github.com/featurecat/lizzie) - 搭配 KataGo 使用的 GUI

准备好了吗？让我们从[安装与设置](./setup.md)开始吧！
