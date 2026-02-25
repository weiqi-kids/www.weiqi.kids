---
sidebar_position: 1
title: The AlphaGo Era
---

# The AlphaGo Era (2015-2017)

From 2015 to 2017, Google DeepMind's AlphaGo series created one of the most iconic breakthroughs in artificial intelligence history. In just two years, Go went from "a game AI cannot conquer" to "a field where AI completely surpasses humans."

## October 2015: AlphaGo Defeats Fan Hui

### The Historic Secret Match

In October 2015, in a London office, DeepMind arranged a secret match. The opponent was European Go champion and professional 2-dan player **Fan Hui**.

Match result: AlphaGo won 5:0.

This was the first time in history a computer program defeated a professional Go player under fair conditions (no handicap). The news was officially announced in January 2016, immediately causing a global sensation.

### First-Generation AlphaGo's Technology

This version of AlphaGo used a combination of two key technologies:

1. **Deep neural networks**: By learning hundreds of thousands of professional human games, trained a "value network" that could evaluate positions and a "policy network" that could predict the next move

2. **Monte Carlo Tree Search (MCTS)**: Used neural network output to guide search, greatly reducing the number of variations that needed calculation

This combination of "intuition" plus "calculation" is exactly how human players think about problems - except AI did both better.

## March 2016: AlphaGo vs Lee Sedol

### The Match of the Century

From March 9-15, 2016, AlphaGo played a five-game match against world top player **Lee Sedol** in Seoul. The match attracted over 200 million viewers worldwide, becoming one of the most-watched events in AI history.

### Match Results

| Game | Date | Result | Notes |
|------|------|------|------|
| Game 1 | March 9 | AlphaGo wins | Resignation |
| Game 2 | March 10 | AlphaGo wins | Resignation, famous "Move 37" |
| Game 3 | March 12 | AlphaGo wins | Resignation |
| Game 4 | March 13 | Lee Sedol wins | **Lee Sedol's Move 78 "Divine Move"** |
| Game 5 | March 15 | AlphaGo wins | Resignation |

Final score: **AlphaGo 4:1 Lee Sedol**

### Game 2, Move 37: "The Divine Move"

In Game 2, AlphaGo played a "shoulder hit" on the right side that puzzled all watching players.

This move looked completely unreasonable, not matching any known human joseki. Commentators estimated the probability of a human playing this move at less than 1 in 10,000. However, as the game progressed, this move's deep meaning gradually became apparent - it simultaneously exerted influence in multiple directions with extremely high efficiency.

This move was called "the divine move," symbolizing that AI had developed Go concepts humans couldn't understand.

### Game 4, Move 78: Humanity Strikes Back

After losing three straight games, Lee Sedol played an equally astonishing move in Game 4 - Move 78, the "wedge."

This was a clever tesuji that created variations in complex fighting that AlphaGo hadn't foreseen. AlphaGo showed obvious confusion after this move and eventually resigned.

This was the only time humans defeated AlphaGo in an official match, and Lee Sedol's move will be forever remembered as a symbol of human intelligence.

### Impact of the Match

The impact of this match extended far beyond the Go world:

- **AI milestone**: Proved deep learning could handle extremely complex problems
- **Korean national attention**: Statistics showed over half Korea's population watched the match
- **New era for Go**: Professional players began realizing they must learn from AI
- **Tech investment boom**: Drove global investment in AI research

## January 2017: Master's 60-Game Winning Streak

### The Mysterious Online Player

From late 2016 to early 2017, an account named "Master" appeared on Go servers like Yike and Fox Go. It defeated all challengers at extremely fast speed, including Ke Jie, Park Junghwan, Iyama Yuta and other world top players.

Final record: **60 wins, 0 losses** (including one draw due to opponent disconnection)

After the 60th game, DeepMind officially announced: Master was a new version of AlphaGo.

### New Concepts Shown by Master

Master's playing style was clearly different from the version that defeated Lee Sedol a year earlier:

- **Faster calculation speed**: Only tens of seconds per move
- **More aggressive play**: Frequently used moves that traditional theory considered "bad"
- **3-3 invasion became mainstream**: Master often invaded 3-3 early in the opening

These moves completely overturned hundreds of years of accumulated human Go theory, and professional players began extensively imitating AI's moves.

## May 2017: AlphaGo vs Ke Jie

### Humanity's Final Challenge

In May 2017, in Wuzhen, China, AlphaGo played a three-game match against then world-ranked #1 **Ke Jie**. This was viewed as "humanity's final challenge."

### Match Results

| Game | Date | Result | Notes |
|------|------|------|------|
| Game 1 | May 23 | AlphaGo wins | Win by 1/4 point (smallest margin) |
| Game 2 | May 25 | AlphaGo wins | Resignation |
| Game 3 | May 27 | AlphaGo wins | Resignation |

Final score: **AlphaGo 3:0 Ke Jie**

### Ke Jie's Tears

During Game 2, Ke Jie left his seat mid-game and returned with red eyes. After the match he said:

> "It is too perfect. I don't see any hope of victory."

> "Playing against AlphaGo, I felt its love for Go."

After this match, DeepMind announced AlphaGo's retirement from public competition.

## October 2017: The AlphaZero Paper

### Surpassing Everything Starting from Zero

In October 2017, DeepMind published the AlphaZero paper, demonstrating an even more astonishing achievement.

AlphaZero's breakthrough was: **It required no human games at all.**

The program was only told Go's rules, then learned through self-play. Starting from "zero," AlphaZero surpassed all previous AlphaGo versions with only **40 days** of self-training.

### Unified Intelligence

Even more amazingly, the same AlphaZero program (with only game rules changed) reached levels surpassing all humans and previous strongest programs in Go, chess, and shogi.

This proved deep reinforcement learning's generality - the same algorithm could master completely different intellectual games.

## Technical Analysis

### Deep Neural Networks

AlphaGo's neural networks had two main parts:

**Policy Network**
- Input: Current board position
- Output: Probability of playing at each position
- Function: Simulates human "intuition," quickly narrowing search range

**Value Network**
- Input: Current board position
- Output: Estimated win rate of current position
- Function: Evaluates position quality, replacing traditional exhaustive search

### Monte Carlo Tree Search (MCTS)

MCTS is a search algorithm that works through these steps:

1. **Selection**: From root node, select child nodes according to some strategy
2. **Expansion**: At leaf nodes, add new child nodes
3. **Simulation**: From new nodes, run random simulations to game end
4. **Backpropagation**: Pass simulation results upward, updating statistics for all nodes along the path

AlphaGo's innovation was replacing random simulation with neural networks, greatly improving search efficiency.

### Reinforcement Learning

From AlphaGo Lee to AlphaZero, reinforcement learning played an increasingly important role:

- **AlphaGo Fan** (defeated Fan Hui): Mainly trained on human games
- **AlphaGo Lee** (defeated Lee Sedol): Human games + self-play
- **AlphaGo Master** (60-game streak): Enhanced self-play training
- **AlphaZero**: Complete self-play, no human games needed

This evolution showed AI could ultimately reach superhuman level through pure self-learning.

---

The AlphaGo era ended in 2017, but the technology and concepts it pioneered continue to influence Go and the field of artificial intelligence. The subsequent KataGo era brought these technologies to every Go enthusiast's computer and phone.

Next: [The KataGo Era](/docs/learn/history/ai-history/katago-era)

