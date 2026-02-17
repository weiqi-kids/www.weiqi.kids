---
sidebar_position: 2
title: 给工程师的围棋 AI 指南
---

# 给工程师的围棋 AI 指南

欢迎来到围棋 AI 技术文档区！这里为想要深入理解、部署或开发围棋 AI 的工程师与开发者，提供完整的技术资源与指南。

## 本区块内容

本区块涵盖以下主题：

### 背景知识
- **AlphaGo 论文解读**：深入分析 DeepMind 的突破性研究，包括 Policy Network、Value Network 与 MCTS 的结合
- **KataGo 论文解读**：了解目前最先进的开源围棋 AI 的创新设计
- **其他围棋 AI 介绍**：商业与开源围棋 AI 的全面比较

### KataGo 实战
- **安装与设置**：从零开始在各平台上搭建 KataGo 环境
- **常用指令**：GTP 协议与 Analysis Engine 的实用指南
- **源代码架构**：深入探索 KataGo 的代码结构与实现细节

## 适合谁阅读

本区块适合以下读者：

| 读者类型 | 建议阅读内容 |
|---------|-------------|
| **软件工程师** | 想在项目中集成围棋 AI → 从「安装与设置」开始 |
| **机器学习工程师** | 想了解围棋 AI 的算法 → 从「AlphaGo 论文解读」开始 |
| **研究者** | 想进行围棋 AI 研究 → 阅读所有背景知识后深入源代码架构 |
| **围棋 App 开发者** | 想开发围棋相关应用 → 重点阅读「常用指令」与「Analysis Engine」 |
| **系统管理员** | 需要部署围棋 AI 服务 → 专注于「安装与设置」章节 |

## 学习路径建议

根据你的目标，我们建议以下学习路径：

### 路径 A：快速上手（1-2 天）

适合想要快速部署 KataGo 的开发者：

1. [KataGo 安装与设置](./katago-source/setup.md) - 搭建执行环境
2. [KataGo 常用指令](./katago-source/commands.md) - 学习基本操作

### 路径 B：深入理解（1-2 周）

适合想要完整理解围棋 AI 技术的工程师：

1. [AlphaGo 论文解读](./background-info/alphago.md) - 理解基础架构
2. [KataGo 论文解读](./background-info/katago-paper.md) - 了解最新改进
3. [其他围棋 AI 介绍](./background-info/zen.md) - 认识产业生态
4. [KataGo 安装与设置](./katago-source/setup.md) - 实际动手操作
5. [KataGo 常用指令](./katago-source/commands.md) - 深入使用功能

### 路径 C：开发贡献（1 个月以上）

适合想要贡献 KataGo 开源项目或开发自己的围棋 AI：

1. 完成路径 B 的所有内容
2. [KataGo 源代码架构](./katago-source/architecture.md) - 深入代码
3. 阅读 KataGo GitHub 上的 Issues 与 Pull Requests
4. 尝试修改与实验

## 预备知识

为了顺利阅读本区块内容，建议具备以下基础知识：

- **程序设计**：熟悉至少一种编程语言（Python、C++ 尤佳）
- **机器学习基础**：了解神经网络、反向传播等基本概念
- **围棋规则**：知道围棋的基本规则与术语
- **命令行操作**：熟悉终端/命令提示符的基本操作

不具备以上知识也可以阅读，但可能需要额外查阅相关资料。

## 开始探索

准备好了吗？从[背景知识](./background-info/)开始你的围棋 AI 技术之旅吧！

如果你已经有机器学习背景，想要快速上手，可以直接前往 [KataGo 实战入门指南](./katago-source/)。
