---
sidebar_position: 1
title: Go Rules
---

# Go Rules

The rules of Go are very simple, but the variations they produce are endless. This is exactly what makes Go fascinating.

## Basic Concepts

### Board and Stones

- **Board**: The standard board is 19x19 lines; beginners often use 9x9 or 13x13
- **Stones**: Black and white colors; Black plays first
- **Placement**: Stones are placed on line intersections, not in the squares

### Game Objective

The objective of Go is to **surround territory**. At the end of the game, the side with more surrounded empty space wins.

---

## The Concept of Liberties

**Liberties** are the most important concept in Go. Liberties are the empty intersections around a stone - they are the stone's "lifeline."

### Liberties of a Single Stone

A single stone has different numbers of liberties in different positions:

| Position | Number of Liberties |
|:---:|:---:|
| Center | 4 liberties |
| Side | 3 liberties |
| Corner | 2 liberties |

### Connected Stones

When stones of the same color are adjacent (connected horizontally or vertically), they become a single unit and share all their liberties.

:::info Important Concept
Connected stones share the same fate. If this group is captured, all connected stones are removed together.
:::

---

## Capturing

When a group of stones has all its liberties blocked by the opponent (liberties = 0), the group is "captured" and removed from the board.

### Steps for Capturing

1. The opponent's stones have only one liberty left
2. You play to block that last liberty
3. The opponent's stones are captured (removed from the board)

### Atari

When a group has only one liberty remaining, this state is called **atari**. The opponent must then try to escape or abandon the stones.

---

## Illegal Moves (Suicide Rule)

Some positions cannot be played - these are called **illegal moves** or **forbidden points**.

### Determining Illegal Moves

A position is an illegal move if both conditions are met:

1. Playing there would leave your own stones with no liberties
2. AND you cannot capture any opponent stones

:::tip Simple Rule
If playing there would capture opponent stones, it's not an illegal move.
:::

### Suicide

Playing a move that leaves your own stones with no liberties without capturing any opponent stones is called "suicide." Go rules prohibit suicide.

---

## Ko Rule

**Ko** is a special shape in Go that would create an endless loop.

### What is Ko

When both sides can alternately capture a single stone, and after capturing, the other side could immediately capture back, this forms a ko.

### The Ko Rule

**No immediate recapture**. After a ko capture, you must play elsewhere first (called "finding a ko threat"), and only then can you recapture.

### The Ko Process

1. Player A captures the ko
2. Player B cannot recapture immediately; must play elsewhere
3. Player A responds to Player B's move
4. Player B recaptures the ko
5. This continues until one side gives up

:::note Why This Rule Exists
Without the ko rule, both sides could infinitely capture back and forth, and the game would never end.
:::

---

## Eyes and Living Groups

**Eyes** are one of the most important concepts in Go. Understanding eyes means understanding life and death.

### What is an Eye

An eye is an empty point completely surrounded by your own stones. The opponent cannot play on an eye point (it would be an illegal move).

### Conditions for a Living Group

**A group must have two or more true eyes to live.**

Why two eyes?

- With only one eye, the opponent can gradually reduce liberties from the outside
- Eventually that eye becomes the last liberty, and the opponent can capture the entire group
- With two eyes, the opponent cannot occupy both eye spaces simultaneously, so the group can never be captured

### True Eyes and False Eyes

- **True eyes**: Complete eyes that the opponent cannot destroy
- **False eyes**: Look like eyes but have defects and may be destroyed

Determining true vs. false eyes requires looking at the diagonal positions - this is advanced life and death knowledge.

---

## Scoring

At the end of the game, you need to count both sides' territory to determine the winner. There are two main counting methods.

### Territory Scoring (Japanese/Korean Rules)

Count the number of empty points each side has surrounded.

**Calculation method**:
- Empty points (territory) surrounded by each side
- Stones on the board are not counted

**Winner determination**: The side with more territory wins.

### Area Scoring (Chinese Rules)

Count the total of stones and territory for each side.

**Calculation method**:
- Number of living stones on the board
- Plus empty points (territory) surrounded

**Winner determination**:
- A standard board has 361 intersections
- The side with more than 180.5 points wins

### Komi (Compensation)

Because Black plays first and has an advantage, White receives compensation points called "komi."

| Rules | Komi |
|:---:|:---:|
| Chinese Rules | 7.5 points (Black gives 3.75 stones) |
| Japanese Rules | 6.5 points |
| Korean Rules | 6.5 points |

:::tip Beginner's Advice
When first learning, don't worry too much about scoring details. First understand the concepts of "liberties" and "eyes" - this is the most important foundation.
:::

---

## End of Game

### When Does the Game End

When both players agree there are no more useful moves to make, the game ends. In practice, both players pass consecutively.

### End of Game Procedure

1. Confirm all dead stones
2. Remove dead stones from the board
3. Count each side's territory
4. Determine the winner according to the rules

### Resignation

During the game, if one side believes there is no possibility of winning, they may resign at any time. Resignation is a common and polite way to end the game.

