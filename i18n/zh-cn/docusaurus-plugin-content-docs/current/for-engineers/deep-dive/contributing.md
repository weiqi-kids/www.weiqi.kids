---
sidebar_position: 4
title: 参与开源社区
description: 加入 KataGo 开源社区，贡献算力或代码
---

# 参与开源社区

KataGo 是一个活跃的开源项目，有多种方式可以参与贡献。

---

## 贡献方式总览

| 方式 | 难度 | 需求 |
|------|------|------|
| **贡献算力** | 低 | 有 GPU 的电脑 |
| **报告问题** | 低 | GitHub 账号 |
| **改进文档** | 中 | 熟悉技术内容 |
| **贡献代码** | 高 | C++/Python 开发能力 |

---

## 贡献算力：分布式训练

### KataGo Training 简介

KataGo Training 是一个全球分布式训练网络：

- 志愿者贡献 GPU 算力执行自我对弈
- 自我对弈数据上传到中央服务器
- 服务器定期训练新模型
- 新模型分发给志愿者继续对弈

官网：https://katagotraining.org/

### 参与步骤

#### 1. 创建账号

前往 https://katagotraining.org/ 注册账号。

#### 2. 下载 KataGo

```bash
# 下载最新版本
wget https://github.com/lightvector/KataGo/releases/download/v1.15.3/katago-v1.15.3-cuda11.1-linux-x64.zip
unzip katago-v1.15.3-cuda11.1-linux-x64.zip
```

#### 3. 配置 contribute 模式

```bash
# 首次执行会引导你配置
./katago contribute -username YOUR_USERNAME -password YOUR_PASSWORD
```

系统会自动：
- 下载最新模型
- 执行自我对弈
- 上传对弈数据

#### 4. 后台执行

```bash
# 使用 screen 或 tmux 后台执行
screen -S katago
./katago contribute -username YOUR_USERNAME -password YOUR_PASSWORD
# Ctrl+A, D 离开 screen
```

### 贡献统计

你可以在 https://katagotraining.org/contributions/ 查看：
- 你的贡献排名
- 总贡献对局数
- 最近训练的模型

---

## 报告问题

### 在哪里报告

- **GitHub Issues**：https://github.com/lightvector/KataGo/issues
- **Discord**：https://discord.gg/bqkZAz3

### 好的问题报告包含

1. **KataGo 版本**：`katago version`
2. **操作系统**：Windows/Linux/macOS
3. **硬件**：GPU 型号、内存
4. **完整错误信息**：复制完整 log
5. **重现步骤**：如何触发这个问题

### 示例

```markdown
## 问题描述
执行 benchmark 时出现显存不足错误

## 环境
- KataGo 版本：1.15.3
- 操作系统：Ubuntu 22.04
- GPU：RTX 3060 12GB
- 模型：kata-b40c256.bin.gz

## 错误信息
```
CUDA error: out of memory
```

## 重现步骤
1. 执行 `katago benchmark -model kata-b40c256.bin.gz`
2. 等待约 30 秒
3. 出现错误
```

---

## 改进文档

### 文档位置

- **README**：`README.md`
- **GTP 文档**：`docs/GTP_Extensions.md`
- **Analysis 文档**：`docs/Analysis_Engine.md`
- **训练文档**：`python/README.md`

### 贡献流程

1. Fork 项目
2. 创建新分支
3. 修改文档
4. 提交 Pull Request

```bash
git clone https://github.com/YOUR_USERNAME/KataGo.git
cd KataGo
git checkout -b improve-docs
# 编辑文档
git add .
git commit -m "Improve documentation for Analysis Engine"
git push origin improve-docs
# 在 GitHub 上创建 Pull Request
```

---

## 贡献代码

### 开发环境设置

```bash
# 克隆项目
git clone https://github.com/lightvector/KataGo.git
cd KataGo

# 编译（Debug 模式）
cd cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=OPENCL -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# 执行测试
./katago runtests
```

### 代码风格

KataGo 使用以下代码风格：

**C++**：
- 2 空格缩进
- 大括号同行
- 变量名使用 camelCase
- 类名使用 PascalCase

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
- 4 空格缩进

### 贡献领域

| 领域 | 文件位置 | 技能需求 |
|------|---------|---------|
| 核心引擎 | `cpp/` | C++, CUDA/OpenCL |
| 训练程序 | `python/` | Python, PyTorch |
| GTP 协议 | `cpp/command/gtp.cpp` | C++ |
| Analysis API | `cpp/command/analysis.cpp` | C++, JSON |
| 测试 | `cpp/tests/` | C++ |

### Pull Request 流程

1. **创建 Issue**：先讨论你想做的改动
2. **Fork & Clone**：创建自己的分支
3. **开发与测试**：确保所有测试通过
4. **提交 PR**：详细描述改动内容
5. **Code Review**：回应维护者的反馈
6. **合并**：维护者合并你的代码

### PR 示例

```markdown
## 改动描述
新增对 New Zealand 规则的支持

## 改动内容
- 在 rules.cpp 新增 NEW_ZEALAND 规则
- 更新 GTP 指令支持 `kata-set-rules nz`
- 新增单元测试

## 测试结果
- 所有现有测试通过
- 新增测试通过

## 相关 Issue
Fixes #123
```

---

## 社区资源

### 官方链接

| 资源 | 链接 |
|------|------|
| GitHub | https://github.com/lightvector/KataGo |
| Discord | https://discord.gg/bqkZAz3 |
| 训练网络 | https://katagotraining.org/ |

### 讨论区

- **Discord**：即时讨论、技术问答
- **GitHub Discussions**：长篇讨论、功能提议
- **Reddit r/baduk**：一般围棋 AI 讨论

### 相关项目

| 项目 | 说明 | 链接 |
|------|------|------|
| KaTrain | 教学分析工具 | github.com/sanderland/katrain |
| Lizzie | 分析界面 | github.com/featurecat/lizzie |
| Sabaki | 棋谱编辑器 | sabaki.yichuanshen.de |
| BadukAI | 在线分析 | baduk.ai |

---

## 认可与奖励

### 贡献者名单

所有贡献者都会列在：
- GitHub Contributors 页面
- KataGo Training 贡献排行榜

### 学习收获

参与开源项目的收获：
- 学习工业级 AI 系统架构
- 与全球开发者交流
- 积累开源贡献记录
- 深入理解围棋 AI 技术

---

## 延伸阅读

- [源代码导读](../source-code) — 理解代码结构
- [KataGo 训练机制解析](../training) — 本地训练实验
- [一篇文章搞懂围棋 AI](../../how-it-works/) — 技术原理
