---
sidebar_position: 1
title: For Deep Learners
description: Advanced topic guide covering neural networks, MCTS, training, optimization, and deployment
---

# For Deep Learners

This section is for engineers who want to dive deep into Go AI, covering technical implementation, theoretical foundations, and practical applications.

---

## Article Overview

### Core Technologies

| Article | Description |
|---------|-------------|
| [Neural Network Architecture](./neural-network) | KataGo's residual network, input features, multi-head output design |
| [MCTS Implementation Details](./mcts-implementation) | PUCT selection, virtual loss, batch evaluation, parallelization |
| [KataGo Training Mechanism](./training) | Self-play, loss functions, training loop |

### Performance Optimization

| Article | Description |
|---------|-------------|
| [GPU Backend & Optimization](./gpu-optimization) | CUDA, OpenCL, Metal backend comparison and tuning |
| [Model Quantization & Deployment](./quantization-deploy) | FP16, INT8, TensorRT, cross-platform deployment |
| [Evaluation & Benchmarking](./evaluation) | Elo rating, match testing, SPRT statistical methods |

### Advanced Topics

| Article | Description |
|---------|-------------|
| [Distributed Training Architecture](./distributed-training) | Self-play Worker, data collection, model release |
| [Custom Rules & Variants](./custom-rules) | Chinese, Japanese, AGA rules, board size variants |
| [Key Papers Guide](./papers) | AlphaGo, AlphaZero, KataGo paper highlights |

### Open Source & Implementation

| Article | Description |
|---------|-------------|
| [KataGo Source Code Guide](./source-code) | Directory structure, core modules, code style |
| [Contributing to Open Source](./contributing) | Contribution methods, distributed training, community participation |
| [Build Go AI from Scratch](./build-from-scratch) | Step-by-step implementation of a simplified AlphaGo Zero |

---

## What Do You Want to Do?

| Goal | Recommended Path |
|------|------------------|
| Understand neural network design | [Neural Network Architecture](./neural-network) → [MCTS Implementation Details](./mcts-implementation) |
| Optimize execution performance | [GPU Backend & Optimization](./gpu-optimization) → [Model Quantization & Deployment](./quantization-deploy) |
| Research training methods | [KataGo Training Mechanism](./training) → [Distributed Training Architecture](./distributed-training) |
| Understand paper principles | [Key Papers Guide](./papers) → [Neural Network Architecture](./neural-network) |
| Hands-on coding | [Build Go AI from Scratch](./build-from-scratch) → [KataGo Source Code Guide](./source-code) |
| Contribute to open source | [Contributing to Open Source](./contributing) → [KataGo Source Code Guide](./source-code) |

---

## Advanced Concept Index

When diving deep, you'll encounter the following advanced concepts:

### F Series: Scaling (8)

| ID | Go Concept | Physics/Math Correspondence |
|----|-----------|----------------------------|
| F1 | Board size vs complexity | Complexity scaling |
| F2 | Network size vs strength | Capacity scaling |
| F3 | Training time vs returns | Diminishing returns |
| F4 | Data volume vs generalization | Sample complexity |
| F5 | Compute resource scaling | Scaling laws |
| F6 | Neural scaling laws | Log-log relationship |
| F7 | Large batch training | Critical batch size |
| F8 | Parameter efficiency | Compression bounds |

### G Series: Dimensions (6)

| ID | Go Concept | Physics/Math Correspondence |
|----|-----------|----------------------------|
| G1 | High-dimensional representation | Vector space |
| G2 | Curse of dimensionality | High-dimensional challenges |
| G3 | Manifold hypothesis | Low-dimensional manifold |
| G4 | Intermediate representation | Latent space |
| G5 | Feature disentanglement | Independent components |
| G6 | Semantic directions | Geometric algebra |

### H Series: Reinforcement Learning (9)

| ID | Go Concept | Physics/Math Correspondence |
|----|-----------|----------------------------|
| H1 | MDP | Markov chain |
| H2 | Bellman equation | Dynamic programming |
| H3 | Value iteration | Fixed-point theorem |
| H4 | Policy gradient | Stochastic optimization |
| H5 | Experience replay | Importance sampling |
| H6 | Discount factor | Time preference |
| H7 | TD learning | Incremental estimation |
| H8 | Advantage function | Baseline variance reduction |
| H9 | PPO clipping | Trust region |

### K Series: Optimization Methods (6)

| ID | Go Concept | Physics/Math Correspondence |
|----|-----------|----------------------------|
| K1 | SGD | Stochastic approximation |
| K2 | Momentum | Inertia |
| K3 | Adam | Adaptive step size |
| K4 | Learning rate decay | Annealing |
| K5 | Gradient clipping | Saturation limits |
| K6 | SGD noise | Stochastic perturbation |

### L Series: Generalization & Stability (5)

| ID | Go Concept | Physics/Math Correspondence |
|----|-----------|----------------------------|
| L1 | Overfitting | Over-adaptation |
| L2 | Regularization | Constrained optimization |
| L3 | Dropout | Sparse activation |
| L4 | Data augmentation | Symmetry breaking |
| L5 | Early stopping | Optimal stopping |

---

## Hardware Requirements

### Reading & Learning

No special requirements, any computer will work.

### Training Models

| Scale | Recommended Hardware | Training Time |
|-------|---------------------|---------------|
| Mini (b6c96) | GTX 1060 6GB | Several hours |
| Small (b10c128) | RTX 3060 12GB | 1-2 days |
| Medium (b18c384) | RTX 4090 24GB | 1-2 weeks |
| Full (b40c256) | Multi-GPU cluster | Several weeks |

### Contributing to Distributed Training

- Any computer with a GPU can participate
- GTX 1060 or equivalent recommended minimum
- Stable internet connection required

---

## Getting Started

**Recommended starting points:**

- Want to understand principles? → [Neural Network Architecture](./neural-network)
- Want to code hands-on? → [Build Go AI from Scratch](./build-from-scratch)
- Want to read papers? → [Key Papers Guide](./papers)
