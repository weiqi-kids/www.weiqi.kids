---
sidebar_position: 2
title: AI Go Development History
---

# AI Go Development History

For a long time, Go was considered the game most difficult for artificial intelligence to conquer. On a board with 19x19 = 361 intersections, each point could potentially be played. The number of variations exceeds the total atoms in the universe (approximately 10^170 possible games). Traditional exhaustive search methods completely failed against Go.

However, between 2015 and 2017, DeepMind's AlphaGo series completely changed everything. This revolution not only affected Go but also pushed forward the entire field of artificial intelligence.

## Why Is Go So Hard?

### Enormous Search Space

Taking chess as an example, on average there are about 35 legal moves per turn, with a game lasting about 80 moves. In Go, there are on average about 250 legal moves per turn, with a game lasting about 150 moves. This means Go's search space is hundreds of orders of magnitude larger than chess.

### Difficult Position Evaluation

In chess, each piece has a clear value (queen 9 points, rook 5 points, etc.), allowing simple formulas to evaluate positions. But in Go, a stone's value depends on its relationship with surrounding stones, with no simple evaluation method.

Is a group alive or dead? How many points is an area of influence worth? These questions often require deep calculation and judgment even for human experts.

### The Predicament of Early Go Programs

Before AlphaGo, the strongest Go programs were only amateur 5-6 dan level, far from professional players. These programs mainly used "Monte Carlo Tree Search" (MCTS), evaluating positions through large-scale random simulations.

But this method had obvious limitations: random simulations couldn't capture strategic thinking in Go, and programs often made mistakes that seemed very foolish to humans.

## Two Eras of AI Go

### [The AlphaGo Era (2015-2017)](/docs/learn/history/ai-history/alphago-era)

This era began with AlphaGo defeating Fan Hui and ended with the publication of the AlphaZero paper. In just two years, DeepMind achieved a leap from defeating professional players to surpassing human limits.

Key milestones:
- 2015.10: Defeats Fan Hui (first defeat of a professional player)
- 2016.03: Defeats Lee Sedol (4:1)
- 2017.01: Master 60-game online winning streak
- 2017.05: Defeats Ke Jie (3:0)
- 2017.10: AlphaZero published

### [The KataGo Era (2019-Present)](/docs/learn/history/ai-history/katago-era)

After AlphaGo retired, the open source community took up the torch. Open source AI like KataGo and Leela Zero allowed everyone to use top-tier Go engines, completely changing how Go is learned and trained.

Characteristics of this era:
- Democratization of AI tools
- Professional players widely using AI for training
- AI-ification of human playing styles
- Overall improvement in Go level

## Cognitive Impact from AI

### Redefining "Correct Play"

Before AI appeared, humans had built through thousands of years what was considered "correct" Go theory. However, many AI moves contradicted traditional human understanding:

- **The 3-3 invasion**: Traditional view was that invading 3-3 early in the opening was "vulgar," but AI often plays this way
- **Shoulder hits**: Previously considered "bad moves," in certain positions AI proves them to be optimal
- **Close-contact attacks**: AI likes close-range fighting, different from traditional human idea of "start attacks from a distance"

### Human Limitations and Potential

AI's appearance made humans recognize their limitations, but also revealed human potential.

With AI's help, young players' growth rate has greatly accelerated. Levels that previously took ten years to reach may now take only three to five years. The overall level of Go has been rising.

### The Future of Go

Some worried AI would make Go meaningless - if you can never beat AI, why play?

But facts proved this worry was unnecessary. AI didn't end Go but opened a new era for it. Games between humans still are full of creativity, emotion, and unpredictability - these are exactly what make Go interesting.

---

Next, let's understand the specific developments of these two eras in detail.

- **[The AlphaGo Era](/docs/learn/history/ai-history/alphago-era)** - From defeating professional players to surpassing human limits
- **[The KataGo Era](/docs/learn/history/ai-history/katago-era)** - Open source AI and Go's new ecosystem

