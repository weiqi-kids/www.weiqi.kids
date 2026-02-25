---
sidebar_position: 4
title: 整合到你的專案
description: Python 與 Node.js 的 KataGo API 整合指南
---

# 整合到你的專案

本文介紹如何將 KataGo 整合到你的應用程式中。

---

## 整合方式選擇

| 方式 | 適用場景 | 複雜度 |
|------|---------|--------|
| **Analysis Engine** | 批次分析、API 服務 | 低 |
| **GTP 模式** | 即時對弈、互動式 | 中 |
| **直接呼叫函式庫** | 深度整合、自訂功能 | 高 |

本文聚焦於 **Analysis Engine**，這是最適合程式整合的方式。

---

## Python 整合

### 基本架構

```python
import subprocess
import json

class KataGoEngine:
    def __init__(self, katago_path, model_path, config_path=None):
        cmd = [katago_path, "analysis", "-model", model_path]
        if config_path:
            cmd.extend(["-config", config_path])

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

    def analyze(self, query):
        """送出分析請求並取得回應"""
        query_str = json.dumps(query) + "\n"
        self.process.stdin.write(query_str)
        self.process.stdin.flush()

        response_line = self.process.stdout.readline()
        return json.loads(response_line)

    def close(self):
        self.process.terminate()
        self.process.wait()
```

### 完整範例：分析一盤棋

```python
import json

# 初始化引擎
engine = KataGoEngine(
    katago_path="/usr/local/bin/katago",
    model_path="./kata-b18c384.bin.gz"
)

# 準備分析請求
query = {
    "id": "game1",
    "moves": [
        ["B", "Q16"],
        ["W", "D4"],
        ["B", "Q4"],
        ["W", "D16"]
    ],
    "rules": "chinese",
    "komi": 7.5,
    "boardXSize": 19,
    "boardYSize": 19,
    "analyzeTurns": [0, 1, 2, 3, 4],  # 分析每一手
    "maxVisits": 500
}

# 執行分析
result = engine.analyze(query)

# 處理結果
print(f"查詢 ID: {result['id']}")
print(f"分析手數: {result['turnNumber']}")
print(f"當前勝率: {result['rootInfo']['winrate']:.1%}")
print(f"預期領先: {result['rootInfo']['scoreLead']:.1f} 目")

print("\n候選下法：")
for move_info in result['moveInfos'][:5]:
    print(f"  {move_info['move']}: "
          f"勝率 {move_info['winrate']:.1%}, "
          f"訪問 {move_info['visits']}, "
          f"變化 {' '.join(move_info['pv'][:3])}")

engine.close()
```

### 批次分析多盤棋

```python
def analyze_game(engine, moves, game_id="game"):
    """分析一盤棋的所有手數"""
    results = []

    for turn in range(len(moves) + 1):
        query = {
            "id": f"{game_id}_turn{turn}",
            "moves": moves[:turn],
            "rules": "chinese",
            "komi": 7.5,
            "boardXSize": 19,
            "boardYSize": 19,
            "analyzeTurns": [turn],
            "maxVisits": 200
        }

        result = engine.analyze(query)
        results.append({
            "turn": turn,
            "winrate": result["rootInfo"]["winrate"],
            "scoreLead": result["rootInfo"]["scoreLead"],
            "bestMove": result["moveInfos"][0]["move"] if result["moveInfos"] else None
        })

    return results

# 使用範例
moves = [["B", "Q16"], ["W", "D4"], ["B", "Q4"], ["W", "D16"]]
analysis = analyze_game(engine, moves)

for turn_data in analysis:
    print(f"手 {turn_data['turn']}: "
          f"勝率 {turn_data['winrate']:.1%}, "
          f"最佳 {turn_data['bestMove']}")
```

### 取得領地預測

```python
query = {
    "id": "ownership_query",
    "moves": [["B", "Q16"], ["W", "D4"]],
    "rules": "chinese",
    "komi": 7.5,
    "boardXSize": 19,
    "boardYSize": 19,
    "analyzeTurns": [2],
    "includeOwnership": True,  # 關鍵參數
    "maxVisits": 500
}

result = engine.analyze(query)

# ownership 是一個 361 元素的陣列
# 值在 -1（白方領地）到 +1（黑方領地）之間
ownership = result.get("ownership", [])

# 轉換為 19x19 棋盤
def ownership_to_board(ownership, size=19):
    board = []
    for row in range(size):
        board.append(ownership[row * size:(row + 1) * size])
    return board

board = ownership_to_board(ownership)

# 視覺化
for row in board:
    line = ""
    for val in row:
        if val > 0.5:
            line += "●"  # 黑方領地
        elif val < -0.5:
            line += "○"  # 白方領地
        else:
            line += "·"  # 中立
    print(line)
```

---

## Node.js 整合

### 基本架構

```javascript
const { spawn } = require('child_process');
const readline = require('readline');

class KataGoEngine {
    constructor(katagoPath, modelPath, configPath = null) {
        const args = ['analysis', '-model', modelPath];
        if (configPath) {
            args.push('-config', configPath);
        }

        this.process = spawn(katagoPath, args);
        this.pendingQueries = new Map();

        const rl = readline.createInterface({
            input: this.process.stdout,
            crlfDelay: Infinity
        });

        rl.on('line', (line) => {
            try {
                const result = JSON.parse(line);
                const resolver = this.pendingQueries.get(result.id);
                if (resolver) {
                    resolver(result);
                    this.pendingQueries.delete(result.id);
                }
            } catch (e) {
                console.error('Parse error:', e);
            }
        });
    }

    analyze(query) {
        return new Promise((resolve) => {
            this.pendingQueries.set(query.id, resolve);
            this.process.stdin.write(JSON.stringify(query) + '\n');
        });
    }

    close() {
        this.process.kill();
    }
}

module.exports = KataGoEngine;
```

### 完整範例

```javascript
const KataGoEngine = require('./katago-engine');

async function main() {
    const engine = new KataGoEngine(
        '/usr/local/bin/katago',
        './kata-b18c384.bin.gz'
    );

    // 等待引擎啟動
    await new Promise(resolve => setTimeout(resolve, 2000));

    const query = {
        id: 'test1',
        moves: [['B', 'Q16'], ['W', 'D4'], ['B', 'Q4']],
        rules: 'chinese',
        komi: 7.5,
        boardXSize: 19,
        boardYSize: 19,
        analyzeTurns: [3],
        maxVisits: 500
    };

    const result = await engine.analyze(query);

    console.log(`勝率: ${(result.rootInfo.winrate * 100).toFixed(1)}%`);
    console.log(`領先: ${result.rootInfo.scoreLead.toFixed(1)} 目`);

    console.log('\n候選下法:');
    result.moveInfos.slice(0, 5).forEach((info, i) => {
        console.log(`  ${i + 1}. ${info.move}: ${(info.winrate * 100).toFixed(1)}%`);
    });

    engine.close();
}

main().catch(console.error);
```

### Express API 服務

```javascript
const express = require('express');
const KataGoEngine = require('./katago-engine');

const app = express();
app.use(express.json());

const engine = new KataGoEngine(
    '/usr/local/bin/katago',
    './kata-b18c384.bin.gz'
);

let queryCounter = 0;

app.post('/analyze', async (req, res) => {
    try {
        const { moves, rules = 'chinese', komi = 7.5 } = req.body;

        const query = {
            id: `query_${++queryCounter}`,
            moves: moves,
            rules: rules,
            komi: komi,
            boardXSize: 19,
            boardYSize: 19,
            analyzeTurns: [moves.length],
            maxVisits: 500
        };

        const result = await engine.analyze(query);

        res.json({
            winrate: result.rootInfo.winrate,
            scoreLead: result.rootInfo.scoreLead,
            bestMoves: result.moveInfos.slice(0, 5).map(info => ({
                move: info.move,
                winrate: info.winrate,
                visits: info.visits
            }))
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.listen(3000, () => {
    console.log('KataGo API 服務啟動於 http://localhost:3000');
});
```

### 使用 API

```bash
curl -X POST http://localhost:3000/analyze \
  -H "Content-Type: application/json" \
  -d '{"moves": [["B","Q16"],["W","D4"],["B","Q4"]]}'
```

---

## 座標轉換工具

### Python

```python
def coord_to_gtp(x, y, board_size=19):
    """將 (x, y) 座標轉換為 GTP 格式（如 Q16）"""
    letters = 'ABCDEFGHJKLMNOPQRST'
    return f"{letters[x]}{board_size - y}"

def gtp_to_coord(gtp_coord, board_size=19):
    """將 GTP 格式轉換為 (x, y) 座標"""
    letters = 'ABCDEFGHJKLMNOPQRST'
    x = letters.index(gtp_coord[0].upper())
    y = board_size - int(gtp_coord[1:])
    return (x, y)

def sgf_to_gtp(sgf_coord, board_size=19):
    """將 SGF 格式（如 'pd'）轉換為 GTP 格式（如 'Q16'）"""
    if not sgf_coord or sgf_coord == '':
        return 'pass'
    x = ord(sgf_coord[0]) - ord('a')
    y = ord(sgf_coord[1]) - ord('a')
    return coord_to_gtp(x, y, board_size)

# 使用範例
print(coord_to_gtp(15, 3))    # Q16
print(gtp_to_coord("Q16"))    # (15, 3)
print(sgf_to_gtp("pd"))       # Q16
```

### JavaScript

```javascript
function coordToGtp(x, y, boardSize = 19) {
    const letters = 'ABCDEFGHJKLMNOPQRST';
    return `${letters[x]}${boardSize - y}`;
}

function gtpToCoord(gtpCoord, boardSize = 19) {
    const letters = 'ABCDEFGHJKLMNOPQRST';
    const x = letters.indexOf(gtpCoord[0].toUpperCase());
    const y = boardSize - parseInt(gtpCoord.slice(1));
    return { x, y };
}

function sgfToGtp(sgfCoord, boardSize = 19) {
    if (!sgfCoord || sgfCoord === '') return 'pass';
    const x = sgfCoord.charCodeAt(0) - 'a'.charCodeAt(0);
    const y = sgfCoord.charCodeAt(1) - 'a'.charCodeAt(0);
    return coordToGtp(x, y, boardSize);
}
```

---

## 錯誤處理

### 常見錯誤

| 錯誤 | 原因 | 解決方案 |
|------|------|---------|
| `Could not load model` | 模型路徑錯誤 | 使用絕對路徑 |
| `No GPU found` | GPU 未正確設定 | 檢查驅動程式或使用 CPU 版本 |
| `Out of memory` | GPU 記憶體不足 | 減少 `nnMaxBatchSize` |
| `Invalid JSON` | 請求格式錯誤 | 確認 JSON 格式正確 |

### 健壯的錯誤處理

```python
import json
import subprocess
import time

class RobustKataGoEngine:
    def __init__(self, katago_path, model_path, max_retries=3):
        self.katago_path = katago_path
        self.model_path = model_path
        self.max_retries = max_retries
        self.process = None
        self._start_engine()

    def _start_engine(self):
        if self.process:
            self.process.terminate()

        self.process = subprocess.Popen(
            [self.katago_path, "analysis", "-model", self.model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        time.sleep(2)  # 等待引擎啟動

    def analyze(self, query, timeout=30):
        for attempt in range(self.max_retries):
            try:
                query_str = json.dumps(query) + "\n"
                self.process.stdin.write(query_str)
                self.process.stdin.flush()

                # 設定讀取超時
                import select
                ready, _, _ = select.select(
                    [self.process.stdout], [], [], timeout
                )

                if ready:
                    response_line = self.process.stdout.readline()
                    return json.loads(response_line)
                else:
                    raise TimeoutError("分析超時")

            except Exception as e:
                print(f"嘗試 {attempt + 1} 失敗: {e}")
                if attempt < self.max_retries - 1:
                    self._start_engine()
                else:
                    raise

    def close(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
```

---

## 效能優化

### 批次處理

```python
def batch_analyze(engine, queries):
    """同時送出多個查詢以提升效率"""
    results = {}

    # 送出所有查詢
    for query in queries:
        query_str = json.dumps(query) + "\n"
        engine.process.stdin.write(query_str)
    engine.process.stdin.flush()

    # 收集所有回應
    for _ in queries:
        response_line = engine.process.stdout.readline()
        result = json.loads(response_line)
        results[result['id']] = result

    return results
```

### 設定調整

```ini
# 高效能設定（適合批次分析）
numSearchThreads = 8
nnMaxBatchSize = 32
maxVisits = 500

# 低延遲設定（適合即時回應）
numSearchThreads = 4
nnMaxBatchSize = 8
maxVisits = 200
```

---

## 延伸閱讀

- [基本使用](../basic-usage) — GTP 與 Analysis Engine 指令
- [一篇文章搞懂圍棋 AI](../../how-it-works/) — 了解技術原理
- [KataGo 原始碼導讀](../../deep-dive/source-code) — 深入程式碼
