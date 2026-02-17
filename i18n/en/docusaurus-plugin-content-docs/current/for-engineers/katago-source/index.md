---
sidebar_position: 2
title: KataGo Practical Guide
---

# KataGo Practical Getting Started Guide

This chapter takes you from installation to practical KataGo usage, covering all essential operational knowledge. Whether you want to integrate KataGo into your application or dive deep into its source code, this is your starting point.

## Why Choose KataGo?

Among many Go AIs, KataGo is currently the best choice for these reasons:

| Advantage | Description |
|------|------|
| **Strongest strength** | Consistently maintains highest level in public tests |
| **Most features** | Score prediction, territory analysis, multiple rule support |
| **Fully open source** | MIT license, free to use and modify |
| **Active updates** | Active development and community support |
| **Well documented** | Detailed official documentation, rich community resources |
| **Multi-platform** | Runs on Linux, macOS, Windows |

## Chapter Contents

### [Installation and Setup](./setup.md)

Build KataGo environment from scratch:

- System requirements and hardware recommendations
- Installation steps for each platform (macOS / Linux / Windows)
- Model download and selection guide
- Detailed configuration file explanations

### [Common Commands](./commands.md)

Master how to use KataGo:

- GTP (Go Text Protocol) introduction
- Common GTP commands and examples
- Analysis Engine usage
- Complete JSON API documentation

### [Source Code Architecture](./architecture.md)

Deep understanding of KataGo implementation details:

- Project directory structure overview
- Neural network architecture analysis
- Search engine implementation details
- Training process overview

## Quick Start

If you just want to quickly try KataGo, here's the simplest approach:

### macOS (using Homebrew)

```bash
# Install
brew install katago

# Download model (choose smaller model for testing)
curl -L -o kata-b18c384.bin.gz \
  https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz

# Run GTP mode
katago gtp -model kata-b18c384.bin.gz -config gtp_example.cfg
```

### Linux (pre-built version)

```bash
# Download pre-built version
wget https://github.com/lightvector/KataGo/releases/download/v1.15.3/katago-v1.15.3-opencl-linux-x64.zip

# Extract
unzip katago-v1.15.3-opencl-linux-x64.zip

# Download model
wget https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz

# Run
./katago gtp -model kata-b18c384nbt-*.bin.gz -config default_gtp.cfg
```

### Verify Installation

After successful startup, you'll see the GTP prompt. Try entering these commands:

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

## Use Case Guides

Based on your needs, here are suggested reading order and focus areas:

### Scenario 1: Integrate into Go App

You want to use KataGo as AI engine in your Go application.

**Focus reading**:
1. [Installation and Setup](./setup.md) - Understand deployment requirements
2. [Common Commands](./commands.md) - Especially the Analysis Engine section

**Key knowledge**:
- Use Analysis Engine mode rather than GTP mode
- Communicate with KataGo via JSON API
- Adjust search parameters based on hardware

### Scenario 2: Build Game Server

You want to set up a server for users to play against AI.

**Focus reading**:
1. [Installation and Setup](./setup.md) - GPU setup section
2. [Common Commands](./commands.md) - GTP protocol section

**Key knowledge**:
- Use GTP mode for playing games
- Multi-instance deployment strategy
- Strength adjustment methods

### Scenario 3: Research AI Algorithms

You want to deeply study KataGo's implementation, possibly modify or experiment.

**Focus reading**:
1. [Source Code Architecture](./architecture.md) - Read thoroughly
2. All paper analyses in background knowledge chapter

**Key knowledge**:
- C++ code structure
- Neural network architecture details
- MCTS implementation

### Scenario 4: Train Your Own Model

You want to train from scratch or fine-tune KataGo models.

**Focus reading**:
1. [Source Code Architecture](./architecture.md) - Training process section
2. [KataGo Paper Analysis](../background-info/katago-paper.md)

**Key knowledge**:
- Training data format
- Training script usage
- Hyperparameter configuration

## Hardware Recommendations

KataGo can run on various hardware, but performance varies significantly:

| Hardware Configuration | Expected Performance | Use Case |
|---------|---------|---------|
| **High-end GPU** (RTX 4090) | ~2000 playouts/sec | Top analysis, fast search |
| **Mid-range GPU** (RTX 3060) | ~500 playouts/sec | General analysis, playing |
| **Entry GPU** (GTX 1650) | ~100 playouts/sec | Basic use |
| **Apple Silicon** (M1/M2) | ~200-400 playouts/sec | macOS development |
| **CPU only** | ~10-30 playouts/sec | Learning, testing |

:::tip
Even on slower hardware, KataGo can provide valuable analysis. Reduced search amounts decrease precision, but for teaching and learning it's usually sufficient.
:::

## Frequently Asked Questions

### What's the difference between KataGo and Leela Zero?

| Aspect | KataGo | Leela Zero |
|------|--------|------------|
| Strength | Stronger | Weaker |
| Features | Rich (score, territory) | Basic |
| Multiple rules | Supported | Not supported |
| Development status | Active | Maintenance mode |
| Training efficiency | High | Lower |

### Do I need a GPU?

Not required, but strongly recommended:
- **With GPU**: Can do fast analysis, get high quality results
- **Without GPU**: Can use Eigen backend, but slower

### Model file differences?

| Model Size | File Size | Strength | Speed |
|---------|---------|------|------|
| b10c128 | ~20 MB | Medium | Fastest |
| b18c384 | ~140 MB | Strong | Fast |
| b40c256 | ~250 MB | Very strong | Medium |
| b60c320 | ~500 MB | Strongest | Slow |

Usually recommend b18c384 or b40c256, balancing strength and speed.

## Related Resources

- [KataGo GitHub](https://github.com/lightvector/KataGo)
- [KataGo Training Website](https://katagotraining.org/)
- [KataGo Discord Community](https://discord.gg/bqkZAz3)
- [Lizzie](https://github.com/featurecat/lizzie) - GUI to use with KataGo

Ready? Let's start with [Installation and Setup](./setup.md)!

