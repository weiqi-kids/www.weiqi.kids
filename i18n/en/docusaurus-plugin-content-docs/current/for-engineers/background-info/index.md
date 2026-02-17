---
sidebar_position: 1
title: Background Knowledge
---

# Background Knowledge Overview

Before diving into KataGo practice, understanding the development history and core technology of Go AI is very important. This chapter will guide you through the technological evolution from AlphaGo to modern Go AI.

## Why Learn Background Knowledge?

The development of Go AI is one of the most exciting breakthroughs in artificial intelligence. The 2016 match where AlphaGo defeated Lee Sedol was not only a milestone in Go history but also marked the tremendous success of combining deep learning with reinforcement learning.

Understanding this background knowledge helps you:

- **Make better technical decisions**: Understand the pros and cons of various approaches, choose the right solution for your project
- **Debug more effectively**: Understanding underlying principles makes diagnosing problems easier
- **Keep up with latest developments**: Mastering fundamentals makes it easier to understand new papers and technologies
- **Contribute to open source projects**: Participating in KataGo and similar project development requires deep understanding of design philosophy

## Chapter Contents

### [AlphaGo Paper Analysis](./alphago.md)

In-depth analysis of DeepMind's classic paper, including:

- Historical significance and impact of AlphaGo
- Policy Network and Value Network design
- Monte Carlo Tree Search (MCTS) principles and implementation
- Self-play training method innovation
- Evolution from AlphaGo to AlphaGo Zero to AlphaZero

### [KataGo Paper Analysis](./katago-paper.md)

Understanding the technical innovations of the current strongest open-source Go AI:

- KataGo improvements over AlphaGo
- More efficient training methods and resource utilization
- Technical implementation for supporting multiple Go rules
- Design for simultaneously predicting win rate and score
- Why KataGo achieves stronger play with fewer resources

### [Other Go AI Introduction](./zen.md)

Comprehensive understanding of the Go AI ecosystem:

- Commercial AI: Zen, Fine Art (Tencent), Golaxy
- Open source AI: Leela Zero, ELF OpenGo, SAI
- Comparison of technical features and use cases for each AI

## Technical Development Timeline

| Time | Event | Significance |
|------|------|--------|
| October 2015 | AlphaGo defeats Fan Hui | First AI defeat of professional player |
| March 2016 | AlphaGo defeats Lee Sedol | World-shaking human-machine match |
| May 2017 | AlphaGo defeats Ke Jie | Confirms AI surpasses top human level |
| October 2017 | AlphaGo Zero published | Pure self-play, no human games needed |
| December 2017 | AlphaZero published | Generalized design, conquers Go, chess, and shogi |
| 2018 | Leela Zero reaches superhuman level | Open source community victory |
| 2019 | KataGo published | More efficient training methods |
| 2020-present | KataGo continues improving | Becomes strongest open-source Go AI |

## Core Concepts Preview

Before reading detailed chapters, here's a brief introduction to several core concepts:

### Role of Neural Networks in Go

```
Board state → Neural Network → { Policy (move probabilities), Value (win rate estimate) }
```

The neural network receives current board state as input and outputs two types of information:
- **Policy**: Probability of playing at each position, guides search direction
- **Value**: Win rate estimate of current position, used to evaluate positions

### Monte Carlo Tree Search (MCTS)

MCTS is a search algorithm that combines with neural networks to determine the best move:

1. **Selection**: From root node, select the most promising path
2. **Expansion**: At leaf node, expand new possible moves
3. **Evaluation**: Use neural network to evaluate position value
4. **Backpropagation**: Pass evaluation results back to update nodes along the path

### Self-play

AI plays against itself to generate training data:

```
Initial model → Self-play → Collect games → Train new model → Stronger model → Repeat
```

This cycle allows AI to continuously improve itself without depending on human games.

## Recommended Reading Order

1. **Read AlphaGo Paper Analysis first**: Establish basic theoretical framework
2. **Then read KataGo Paper Analysis**: Understand latest improvements and optimizations
3. **Finally read Other Go AI Introduction**: Expand perspective, understand different implementations

Ready? Let's start with [AlphaGo Paper Analysis](./alphago.md)!

