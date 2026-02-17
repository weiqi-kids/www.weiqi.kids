---
sidebar_position: 2
title: Common Commands
---

# KataGo Common Commands

This article introduces KataGo's two main operation modes: GTP protocol and Analysis Engine, along with detailed explanations of common commands.

## GTP Protocol Introduction

GTP (Go Text Protocol) is the standard protocol for communication between Go programs. Most Go GUIs (like Sabaki, Lizzie) use GTP to communicate with AI engines.

### Start GTP Mode

```bash
katago gtp -model /path/to/model.bin.gz -config /path/to/config.cfg
```

### GTP Protocol Basic Format

```
[id] command_name [arguments]
```

- `id`: Optional command number for tracking responses
- `command_name`: Command name
- `arguments`: Command parameters

Response format:
```
=[id] response_data     # Success
?[id] error_message     # Failure
```

### Basic Example

```
1 name
=1 KataGo

2 version
=2 1.15.3

3 boardsize 19
=3

4 komi 7.5
=4

5 play black Q16
=5

6 genmove white
=6 D4
```

## Common GTP Commands

### Program Information

| Command | Description | Example |
|------|------|------|
| `name` | Get program name | `name` → `= KataGo` |
| `version` | Get version number | `version` → `= 1.15.3` |
| `list_commands` | List all supported commands | `list_commands` |
| `protocol_version` | GTP protocol version | `protocol_version` → `= 2` |

### Board Setup

```
# Set board size (9, 13, 19)
boardsize 19

# Set komi
komi 7.5

# Clear board
clear_board

# Set rules (KataGo extension)
kata-set-rules chinese    # Chinese rules
kata-set-rules japanese   # Japanese rules
kata-set-rules tromp-taylor
```

### Playing

```
# Play a move
play black Q16    # Black plays at Q16
play white D4     # White plays at D4
play black pass   # Black passes

# Have AI generate a move
genmove black     # Generate Black's move
genmove white     # Generate White's move

# Undo
undo              # Undo one move

# Set visit limit
kata-set-param maxVisits 1000    # Set maximum search visits
```

### Position Query

```
# Show board
showboard

# Get current player
kata-get-player

# Get analysis results
kata-analyze black 100    # Analyze for Black, 100 visits
```

### Rules Related

```
# Get current rules
kata-get-rules

# Set rules
kata-set-rules chinese

# Set handicap
fixed_handicap 4     # Standard 4-stone handicap positions
place_free_handicap 4  # Free handicap placement
```

## KataGo Extension Commands

KataGo provides many extension commands beyond standard GTP:

### kata-analyze

Real-time analysis of current position:

```
kata-analyze [player] [visits] [interval]
```

Parameters:
- `player`: Which side to analyze (black/white)
- `visits`: Number of search visits
- `interval`: Report interval (centiseconds, 1/100 second)

Example:
```
kata-analyze black 1000 100
```

Output:
```
info move Q3 visits 523 winrate 0.5432 scoreMean 2.31 scoreSelfplay 2.45 prior 0.1234 order 0 pv Q3 R4 Q5 ...
info move D4 visits 312 winrate 0.5123 scoreMean 1.82 scoreSelfplay 1.95 prior 0.0987 order 1 pv D4 C6 E3 ...
...
```

Output field explanations:

| Field | Description |
|------|------|
| `move` | Move location |
| `visits` | Search visit count |
| `winrate` | Win rate (0-1) |
| `scoreMean` | Expected score difference |
| `scoreSelfplay` | Self-play expected score |
| `prior` | Neural network prior probability |
| `order` | Ranking order |
| `pv` | Principal Variation |

### kata-raw-nn

Get raw neural network output:

```
kata-raw-nn [symmetry]
```

Output includes:
- Policy probability distribution
- Value prediction
- Territory prediction, etc.

### kata-debug-print

Display detailed search information for debugging:

```
kata-debug-print move Q16
```

### Strength Adjustment

```
# Set maximum visits
kata-set-param maxVisits 100      # Weaker
kata-set-param maxVisits 10000    # Stronger

# Set thinking time
kata-time-settings main 60 0      # 60 seconds per side
kata-time-settings byoyomi 30 5   # 30 second byo-yomi, 5 periods
```

## Analysis Engine Usage

Analysis Engine is another operation mode KataGo provides, using JSON format communication, more suitable for programmatic use.

### Start Analysis Engine

```bash
katago analysis -model /path/to/model.bin.gz -config /path/to/config.cfg
```

### Basic Usage Flow

```
Your program ──JSON request──> KataGo Analysis Engine ──JSON response──> Your program
```

### Request Format

Each request is a JSON object that must be on one line:

```json
{
  "id": "query1",
  "moves": [["B","Q16"],["W","D4"],["B","Q4"]],
  "rules": "chinese",
  "komi": 7.5,
  "boardXSize": 19,
  "boardYSize": 19,
  "analyzeTurns": [2]
}
```

### Request Field Explanations

| Field | Required | Description |
|------|------|------|
| `id` | Yes | Query identifier for matching responses |
| `moves` | No | Move sequence `[["B","Q16"],["W","D4"]]` |
| `initialStones` | No | Initial stones `[["B","Q16"],["W","D4"]]` |
| `rules` | Yes | Rule name |
| `komi` | Yes | Komi |
| `boardXSize` | Yes | Board width |
| `boardYSize` | Yes | Board height |
| `analyzeTurns` | No | Which turns to analyze (0-indexed) |
| `maxVisits` | No | Override config's maxVisits |

### Response Format

```json
{
  "id": "query1",
  "turnNumber": 2,
  "moveInfos": [
    {
      "move": "D16",
      "visits": 1234,
      "winrate": 0.5678,
      "scoreMean": 3.21,
      "scoreStdev": 15.4,
      "scoreLead": 3.21,
      "prior": 0.0892,
      "order": 0,
      "pv": ["D16", "Q10", "R14"]
    }
  ],
  "rootInfo": {
    "visits": 5000,
    "winrate": 0.5234,
    "scoreLead": 2.1,
    "scoreSelfplay": 2.3
  },
  "ownership": [...],
  "policy": [...]
}
```

### Response Field Explanations

#### moveInfos Fields

| Field | Description |
|------|------|
| `move` | Move coordinate |
| `visits` | Search visit count for this move |
| `winrate` | Win rate (0-1, for current player) |
| `scoreMean` | Expected final score difference |
| `scoreStdev` | Score standard deviation |
| `scoreLead` | Current point lead |
| `prior` | Neural network prior probability |
| `order` | Rank (0 = best) |
| `pv` | Principal variation sequence |

#### rootInfo Fields

| Field | Description |
|------|------|
| `visits` | Total search visits |
| `winrate` | Current position win rate |
| `scoreLead` | Current point lead |
| `scoreSelfplay` | Self-play expected score |

#### ownership Field

1D array, length boardXSize × boardYSize, each value between -1 and 1:
- -1: Predicted White territory
- +1: Predicted Black territory
- 0: Undetermined/border

### Advanced Query Options

#### Get Ownership Map

```json
{
  "id": "ownership_query",
  "moves": [...],
  "rules": "chinese",
  "komi": 7.5,
  "boardXSize": 19,
  "boardYSize": 19,
  "analyzeTurns": [10],
  "includeOwnership": true
}
```

#### Get Policy Distribution

```json
{
  "id": "policy_query",
  "moves": [...],
  "rules": "chinese",
  "komi": 7.5,
  "boardXSize": 19,
  "boardYSize": 19,
  "analyzeTurns": [10],
  "includePolicy": true
}
```

#### Limit Reported Moves

```json
{
  "id": "limited_query",
  "moves": [...],
  "rules": "chinese",
  "komi": 7.5,
  "boardXSize": 19,
  "boardYSize": 19,
  "analyzeTurns": [10],
  "maxMoves": 5
}
```

#### Analyze Specific Moves

```json
{
  "id": "specific_moves",
  "moves": [...],
  "rules": "chinese",
  "komi": 7.5,
  "boardXSize": 19,
  "boardYSize": 19,
  "analyzeTurns": [10],
  "allowMoves": [["B","Q16"],["B","D4"],["B","Q4"]]
}
```

### Complete Example: Python Integration

```python
import subprocess
import json

class KataGoEngine:
    def __init__(self, katago_path, model_path, config_path):
        self.process = subprocess.Popen(
            [katago_path, 'analysis', '-model', model_path, '-config', config_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self.query_id = 0

    def analyze(self, moves, rules='chinese', komi=7.5):
        self.query_id += 1

        query = {
            'id': f'query_{self.query_id}',
            'moves': moves,
            'rules': rules,
            'komi': komi,
            'boardXSize': 19,
            'boardYSize': 19,
            'analyzeTurns': [len(moves)],
            'maxVisits': 500,
            'includeOwnership': True
        }

        # Send query
        self.process.stdin.write(json.dumps(query) + '\n')
        self.process.stdin.flush()

        # Read response
        response_line = self.process.stdout.readline()
        return json.loads(response_line)

    def close(self):
        self.process.terminate()


# Usage example
engine = KataGoEngine(
    '/usr/local/bin/katago',
    '/path/to/model.bin.gz',
    '/path/to/config.cfg'
)

# Analyze a position
result = engine.analyze([
    ['B', 'Q16'],
    ['W', 'D4'],
    ['B', 'Q4'],
    ['W', 'D16']
])

# Print best move
best_move = result['moveInfos'][0]
print(f"Best move: {best_move['move']}")
print(f"Win rate: {best_move['winrate']:.1%}")
print(f"Score lead: {best_move['scoreLead']:.1f}")

engine.close()
```

### Complete Example: Node.js Integration

```javascript
const { spawn } = require('child_process');
const readline = require('readline');

class KataGoEngine {
  constructor(katagoPath, modelPath, configPath) {
    this.process = spawn(katagoPath, [
      'analysis',
      '-model', modelPath,
      '-config', configPath
    ]);

    this.rl = readline.createInterface({
      input: this.process.stdout,
      crlfDelay: Infinity
    });

    this.queryId = 0;
    this.callbacks = new Map();

    this.rl.on('line', (line) => {
      try {
        const response = JSON.parse(line);
        const callback = this.callbacks.get(response.id);
        if (callback) {
          callback(response);
          this.callbacks.delete(response.id);
        }
      } catch (e) {
        console.error('Parse error:', e);
      }
    });
  }

  analyze(moves, options = {}) {
    return new Promise((resolve) => {
      this.queryId++;
      const id = `query_${this.queryId}`;

      const query = {
        id,
        moves,
        rules: options.rules || 'chinese',
        komi: options.komi || 7.5,
        boardXSize: 19,
        boardYSize: 19,
        analyzeTurns: [moves.length],
        maxVisits: options.maxVisits || 500,
        includeOwnership: true
      };

      this.callbacks.set(id, resolve);
      this.process.stdin.write(JSON.stringify(query) + '\n');
    });
  }

  close() {
    this.process.kill();
  }
}

// Usage example
async function main() {
  const engine = new KataGoEngine(
    '/usr/local/bin/katago',
    '/path/to/model.bin.gz',
    '/path/to/config.cfg'
  );

  const result = await engine.analyze([
    ['B', 'Q16'],
    ['W', 'D4'],
    ['B', 'Q4']
  ]);

  console.log('Best move:', result.moveInfos[0].move);
  console.log('Win rate:', (result.moveInfos[0].winrate * 100).toFixed(1) + '%');

  engine.close();
}

main();
```

## Coordinate System

KataGo uses the standard Go coordinate system:

### Letter Coordinates

```
   A B C D E F G H J K L M N O P Q R S T
19 . . . . . . . . . . . . . . . . . . . 19
18 . . . . . . . . . . . . . . . . . . . 18
17 . . . . . . . . . . . . . . . . . . . 17
16 . . . + . . . . . + . . . . . + . . . 16
15 . . . . . . . . . . . . . . . . . . . 15
...
 4 . . . + . . . . . + . . . . . + . . .  4
 3 . . . . . . . . . . . . . . . . . . .  3
 2 . . . . . . . . . . . . . . . . . . .  2
 1 . . . . . . . . . . . . . . . . . . .  1
   A B C D E F G H J K L M N O P Q R S T
```

Note: There's no letter I (to avoid confusion with number 1).

### Coordinate Conversion

```python
def coord_to_gtp(x, y, board_size=19):
    """Convert (x, y) coordinates to GTP format"""
    letters = 'ABCDEFGHJKLMNOPQRST'
    return f"{letters[x]}{board_size - y}"

def gtp_to_coord(gtp_coord, board_size=19):
    """Convert GTP coordinates to (x, y)"""
    letters = 'ABCDEFGHJKLMNOPQRST'
    x = letters.index(gtp_coord[0].upper())
    y = board_size - int(gtp_coord[1:])
    return (x, y)
```

## Common Usage Patterns

### Playing Mode

```bash
# Start GTP mode
katago gtp -model model.bin.gz -config gtp.cfg

# GTP command sequence
boardsize 19
komi 7.5
play black Q16
genmove white
play black Q4
genmove white
...
```

### Batch Analysis Mode

```python
# Analyze all moves of a game
sgf_moves = parse_sgf('game.sgf')

for i in range(len(sgf_moves)):
    result = engine.analyze(sgf_moves[:i+1])
    winrate = result['rootInfo']['winrate']
    print(f"Move {i+1}: Win rate {winrate:.1%}")
```

### Real-time Analysis Mode

Use `kata-analyze` for real-time analysis:

```
kata-analyze black 1000 50
```

Outputs analysis results every 0.5 seconds until reaching 1000 visits.

## Performance Tuning

### Search Settings

```ini
# Increase search amount for better accuracy
maxVisits = 1000

# Or use time control
maxTime = 10  # Max 10 seconds thinking per move
```

### Multi-threading Settings

```ini
# CPU thread count
numSearchThreads = 8

# GPU batch processing
numNNServerThreadsPerModel = 2
nnMaxBatchSize = 16
```

### Memory Settings

```ini
# Reduce memory usage
nnCacheSizePowerOfTwo = 20  # Default 23
```

## Next Steps

After understanding command usage, if you want to study KataGo's implementation in depth, continue reading [Source Code Architecture](./architecture.md).

