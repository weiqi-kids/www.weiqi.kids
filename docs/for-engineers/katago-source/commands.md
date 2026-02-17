---
sidebar_position: 2
title: 常用指令
---

# KataGo 常用指令

本文介紹 KataGo 的兩種主要操作模式：GTP 協議和 Analysis Engine，以及常用指令的詳細說明。

## GTP 協議介紹

GTP（Go Text Protocol）是圍棋程式之間通訊的標準協議。大多數圍棋 GUI（如 Sabaki、Lizzie）都使用 GTP 與 AI 引擎溝通。

### 啟動 GTP 模式

```bash
katago gtp -model /path/to/model.bin.gz -config /path/to/config.cfg
```

### GTP 協議基本格式

```
[id] command_name [arguments]
```

- `id`：可選的指令編號，用於追蹤回應
- `command_name`：指令名稱
- `arguments`：指令參數

回應格式：
```
=[id] response_data     # 成功
?[id] error_message     # 失敗
```

### 基本範例

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

## 常用 GTP 指令

### 程式資訊

| 指令 | 說明 | 範例 |
|------|------|------|
| `name` | 取得程式名稱 | `name` → `= KataGo` |
| `version` | 取得版本號 | `version` → `= 1.15.3` |
| `list_commands` | 列出所有支援的指令 | `list_commands` |
| `protocol_version` | GTP 協議版本 | `protocol_version` → `= 2` |

### 棋盤設定

```
# 設定棋盤大小（9、13、19）
boardsize 19

# 設定貼目
komi 7.5

# 清除棋盤
clear_board

# 設定規則（KataGo 擴充）
kata-set-rules chinese    # 中國規則
kata-set-rules japanese   # 日本規則
kata-set-rules tromp-taylor
```

### 下棋相關

```
# 落子
play black Q16    # 黑棋下在 Q16
play white D4     # 白棋下在 D4
play black pass   # 黑棋虛手

# 讓 AI 下一手
genmove black     # 生成黑棋的一手
genmove white     # 生成白棋的一手

# 撤銷
undo              # 撤銷一手

# 設定手數限制
kata-set-param maxVisits 1000    # 設定最大搜索次數
```

### 局面查詢

```
# 顯示棋盤
showboard

# 取得當前落子方
kata-get-player

# 取得分析結果
kata-analyze black 100    # 分析黑棋，搜索 100 次
```

### 規則相關

```
# 取得當前規則
kata-get-rules

# 設定規則
kata-set-rules chinese

# 設定讓子
fixed_handicap 4     # 標準讓四子位置
place_free_handicap 4  # 自由讓子
```

## KataGo 擴充指令

KataGo 在標準 GTP 之外提供許多擴充指令：

### kata-analyze

即時分析當前局面：

```
kata-analyze [player] [visits] [interval]
```

參數：
- `player`：分析哪方（black/white）
- `visits`：搜索次數
- `interval`：報告間隔（centiseconds，1/100 秒）

範例：
```
kata-analyze black 1000 100
```

輸出：
```
info move Q3 visits 523 winrate 0.5432 scoreMean 2.31 scoreSelfplay 2.45 prior 0.1234 order 0 pv Q3 R4 Q5 ...
info move D4 visits 312 winrate 0.5123 scoreMean 1.82 scoreSelfplay 1.95 prior 0.0987 order 1 pv D4 C6 E3 ...
...
```

輸出欄位說明：

| 欄位 | 說明 |
|------|------|
| `move` | 著點 |
| `visits` | 搜索訪問次數 |
| `winrate` | 勝率（0-1） |
| `scoreMean` | 預期目數差 |
| `scoreSelfplay` | 自我對弈預期目數 |
| `prior` | 神經網路的先驗機率 |
| `order` | 排名順序 |
| `pv` | 主要變化（Principal Variation） |

### kata-raw-nn

取得原始神經網路輸出：

```
kata-raw-nn [symmetry]
```

輸出包含：
- Policy 機率分佈
- Value 預測
- 領地預測等

### kata-debug-print

顯示詳細的搜索資訊，用於除錯：

```
kata-debug-print move Q16
```

### 棋力調整

```
# 設定最大訪問次數
kata-set-param maxVisits 100      # 較弱
kata-set-param maxVisits 10000    # 較強

# 設定思考時間
kata-time-settings main 60 0      # 每方 60 秒
kata-time-settings byoyomi 30 5   # 讀秒 30 秒 5 次
```

## Analysis Engine 使用

Analysis Engine 是 KataGo 提供的另一種操作模式，使用 JSON 格式通訊，更適合程式化使用。

### 啟動 Analysis Engine

```bash
katago analysis -model /path/to/model.bin.gz -config /path/to/config.cfg
```

### 基本使用流程

```
你的程式 ──JSON請求──> KataGo Analysis Engine ──JSON回應──> 你的程式
```

### 請求格式

每個請求是一個 JSON 物件，必須佔一行：

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

### 請求欄位說明

| 欄位 | 必須 | 說明 |
|------|------|------|
| `id` | 是 | 查詢識別碼，用於對應回應 |
| `moves` | 否 | 棋步序列 `[["B","Q16"],["W","D4"]]` |
| `initialStones` | 否 | 初始棋子 `[["B","Q16"],["W","D4"]]` |
| `rules` | 是 | 規則名稱 |
| `komi` | 是 | 貼目 |
| `boardXSize` | 是 | 棋盤寬度 |
| `boardYSize` | 是 | 棋盤高度 |
| `analyzeTurns` | 否 | 要分析的手數（0-indexed） |
| `maxVisits` | 否 | 覆蓋設定檔的 maxVisits |

### 回應格式

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

### 回應欄位說明

#### moveInfos 欄位

| 欄位 | 說明 |
|------|------|
| `move` | 著點座標 |
| `visits` | 該著點的搜索訪問次數 |
| `winrate` | 勝率（0-1，對當前落子方） |
| `scoreMean` | 預期最終目數差 |
| `scoreStdev` | 目數標準差 |
| `scoreLead` | 當前領先目數 |
| `prior` | 神經網路先驗機率 |
| `order` | 排名（0 = 最佳） |
| `pv` | 主要變化序列 |

#### rootInfo 欄位

| 欄位 | 說明 |
|------|------|
| `visits` | 總搜索訪問次數 |
| `winrate` | 當前局面勝率 |
| `scoreLead` | 當前領先目數 |
| `scoreSelfplay` | 自我對弈預期目數 |

#### ownership 欄位

一維陣列，長度為 boardXSize × boardYSize，每個值在 -1 到 1 之間：
- -1：預測為白方領地
- +1：預測為黑方領地
- 0：未定/邊界

### 進階查詢選項

#### 取得領地圖

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

#### 取得 Policy 分佈

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

#### 限制回報的著法數量

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

#### 分析特定著法

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

### 完整範例：Python 整合

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

        # 發送查詢
        self.process.stdin.write(json.dumps(query) + '\n')
        self.process.stdin.flush()

        # 讀取回應
        response_line = self.process.stdout.readline()
        return json.loads(response_line)

    def close(self):
        self.process.terminate()


# 使用範例
engine = KataGoEngine(
    '/usr/local/bin/katago',
    '/path/to/model.bin.gz',
    '/path/to/config.cfg'
)

# 分析一個局面
result = engine.analyze([
    ['B', 'Q16'],
    ['W', 'D4'],
    ['B', 'Q4'],
    ['W', 'D16']
])

# 印出最佳著法
best_move = result['moveInfos'][0]
print(f"最佳著法：{best_move['move']}")
print(f"勝率：{best_move['winrate']:.1%}")
print(f"領先目數：{best_move['scoreLead']:.1f}")

engine.close()
```

### 完整範例：Node.js 整合

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

// 使用範例
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

  console.log('最佳著法:', result.moveInfos[0].move);
  console.log('勝率:', (result.moveInfos[0].winrate * 100).toFixed(1) + '%');

  engine.close();
}

main();
```

## 座標系統

KataGo 使用標準的圍棋座標系統：

### 字母座標

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

注意：沒有 I 這個字母（避免與數字 1 混淆）。

### 座標轉換

```python
def coord_to_gtp(x, y, board_size=19):
    """將 (x, y) 座標轉換為 GTP 格式"""
    letters = 'ABCDEFGHJKLMNOPQRST'
    return f"{letters[x]}{board_size - y}"

def gtp_to_coord(gtp_coord, board_size=19):
    """將 GTP 座標轉換為 (x, y)"""
    letters = 'ABCDEFGHJKLMNOPQRST'
    x = letters.index(gtp_coord[0].upper())
    y = board_size - int(gtp_coord[1:])
    return (x, y)
```

## 常見使用模式

### 對弈模式

```bash
# 啟動 GTP 模式
katago gtp -model model.bin.gz -config gtp.cfg

# GTP 指令序列
boardsize 19
komi 7.5
play black Q16
genmove white
play black Q4
genmove white
...
```

### 批次分析模式

```python
# 分析一盤棋的所有著法
sgf_moves = parse_sgf('game.sgf')

for i in range(len(sgf_moves)):
    result = engine.analyze(sgf_moves[:i+1])
    winrate = result['rootInfo']['winrate']
    print(f"手 {i+1}: 勝率 {winrate:.1%}")
```

### 即時分析模式

使用 `kata-analyze` 進行即時分析：

```
kata-analyze black 1000 50
```

會每 0.5 秒輸出一次分析結果，直到達到 1000 次訪問。

## 效能調優

### 搜索設定

```ini
# 增加搜索量提高準確度
maxVisits = 1000

# 或使用時間控制
maxTime = 10  # 每手最多思考 10 秒
```

### 多執行緒設定

```ini
# CPU 執行緒數
numSearchThreads = 8

# GPU 批次處理
numNNServerThreadsPerModel = 2
nnMaxBatchSize = 16
```

### 記憶體設定

```ini
# 減少記憶體使用
nnCacheSizePowerOfTwo = 20  # 預設 23
```

## 下一步

了解指令使用後，如果你想深入研究 KataGo 的實作，請繼續閱讀 [原始碼架構](./architecture.md)。
