---
sidebar_position: 2
title: 常用指令
---

# KataGo 常用指令

本文介绍 KataGo 的两种主要操作模式：GTP 协议和 Analysis Engine，以及常用指令的详细说明。

## GTP 协议介绍

GTP（Go Text Protocol）是围棋程序之间通信的标准协议。大多数围棋 GUI（如 Sabaki、Lizzie）都使用 GTP 与 AI 引擎沟通。

### 启动 GTP 模式

```bash
katago gtp -model /path/to/model.bin.gz -config /path/to/config.cfg
```

### GTP 协议基本格式

```
[id] command_name [arguments]
```

- `id`：可选的指令编号，用于追踪响应
- `command_name`：指令名称
- `arguments`：指令参数

响应格式：
```
=[id] response_data     # 成功
?[id] error_message     # 失败
```

### 基本示例

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

### 程序信息

| 指令 | 说明 | 示例 |
|------|------|------|
| `name` | 获取程序名称 | `name` → `= KataGo` |
| `version` | 获取版本号 | `version` → `= 1.15.3` |
| `list_commands` | 列出所有支持的指令 | `list_commands` |
| `protocol_version` | GTP 协议版本 | `protocol_version` → `= 2` |

### 棋盘设置

```
# 设置棋盘大小（9、13、19）
boardsize 19

# 设置贴目
komi 7.5

# 清除棋盘
clear_board

# 设置规则（KataGo 扩展）
kata-set-rules chinese    # 中国规则
kata-set-rules japanese   # 日本规则
kata-set-rules tromp-taylor
```

### 下棋相关

```
# 落子
play black Q16    # 黑棋下在 Q16
play white D4     # 白棋下在 D4
play black pass   # 黑棋虚手

# 让 AI 下一手
genmove black     # 生成黑棋的一手
genmove white     # 生成白棋的一手

# 撤销
undo              # 撤销一手

# 设置手数限制
kata-set-param maxVisits 1000    # 设置最大搜索次数
```

### 局面查询

```
# 显示棋盘
showboard

# 获取当前落子方
kata-get-player

# 获取分析结果
kata-analyze black 100    # 分析黑棋，搜索 100 次
```

### 规则相关

```
# 获取当前规则
kata-get-rules

# 设置规则
kata-set-rules chinese

# 设置让子
fixed_handicap 4     # 标准让四子位置
place_free_handicap 4  # 自由让子
```

## KataGo 扩展指令

KataGo 在标准 GTP 之外提供许多扩展指令：

### kata-analyze

即时分析当前局面：

```
kata-analyze [player] [visits] [interval]
```

参数：
- `player`：分析哪方（black/white）
- `visits`：搜索次数
- `interval`：报告间隔（centiseconds，1/100 秒）

示例：
```
kata-analyze black 1000 100
```

输出：
```
info move Q3 visits 523 winrate 0.5432 scoreMean 2.31 scoreSelfplay 2.45 prior 0.1234 order 0 pv Q3 R4 Q5 ...
info move D4 visits 312 winrate 0.5123 scoreMean 1.82 scoreSelfplay 1.95 prior 0.0987 order 1 pv D4 C6 E3 ...
...
```

输出字段说明：

| 字段 | 说明 |
|------|------|
| `move` | 着点 |
| `visits` | 搜索访问次数 |
| `winrate` | 胜率（0-1） |
| `scoreMean` | 预期目数差 |
| `scoreSelfplay` | 自我对弈预期目数 |
| `prior` | 神经网络的先验概率 |
| `order` | 排名顺序 |
| `pv` | 主要变化（Principal Variation） |

### kata-raw-nn

获取原始神经网络输出：

```
kata-raw-nn [symmetry]
```

输出包含：
- Policy 概率分布
- Value 预测
- 领地预测等

### kata-debug-print

显示详细的搜索信息，用于调试：

```
kata-debug-print move Q16
```

### 棋力调整

```
# 设置最大访问次数
kata-set-param maxVisits 100      # 较弱
kata-set-param maxVisits 10000    # 较强

# 设置思考时间
kata-time-settings main 60 0      # 每方 60 秒
kata-time-settings byoyomi 30 5   # 读秒 30 秒 5 次
```

## Analysis Engine 使用

Analysis Engine 是 KataGo 提供的另一种操作模式，使用 JSON 格式通信，更适合程序化使用。

### 启动 Analysis Engine

```bash
katago analysis -model /path/to/model.bin.gz -config /path/to/config.cfg
```

### 基本使用流程

```
你的程序 ──JSON请求──> KataGo Analysis Engine ──JSON响应──> 你的程序
```

### 请求格式

每个请求是一个 JSON 对象，必须占一行：

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

### 请求字段说明

| 字段 | 必须 | 说明 |
|------|------|------|
| `id` | 是 | 查询标识符，用于对应响应 |
| `moves` | 否 | 棋步序列 `[["B","Q16"],["W","D4"]]` |
| `initialStones` | 否 | 初始棋子 `[["B","Q16"],["W","D4"]]` |
| `rules` | 是 | 规则名称 |
| `komi` | 是 | 贴目 |
| `boardXSize` | 是 | 棋盘宽度 |
| `boardYSize` | 是 | 棋盘高度 |
| `analyzeTurns` | 否 | 要分析的手数（0-indexed） |
| `maxVisits` | 否 | 覆盖配置文件的 maxVisits |

### 响应格式

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

### 响应字段说明

#### moveInfos 字段

| 字段 | 说明 |
|------|------|
| `move` | 着点坐标 |
| `visits` | 该着点的搜索访问次数 |
| `winrate` | 胜率（0-1，对当前落子方） |
| `scoreMean` | 预期最终目数差 |
| `scoreStdev` | 目数标准差 |
| `scoreLead` | 当前领先目数 |
| `prior` | 神经网络先验概率 |
| `order` | 排名（0 = 最佳） |
| `pv` | 主要变化序列 |

#### rootInfo 字段

| 字段 | 说明 |
|------|------|
| `visits` | 总搜索访问次数 |
| `winrate` | 当前局面胜率 |
| `scoreLead` | 当前领先目数 |
| `scoreSelfplay` | 自我对弈预期目数 |

#### ownership 字段

一维数组，长度为 boardXSize × boardYSize，每个值在 -1 到 1 之间：
- -1：预测为白方领地
- +1：预测为黑方领地
- 0：未定/边界

### 高级查询选项

#### 获取领地图

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

#### 获取 Policy 分布

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

#### 限制报告的着法数量

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

#### 分析特定着法

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

### 完整示例：Python 集成

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

        # 发送查询
        self.process.stdin.write(json.dumps(query) + '\n')
        self.process.stdin.flush()

        # 读取响应
        response_line = self.process.stdout.readline()
        return json.loads(response_line)

    def close(self):
        self.process.terminate()


# 使用示例
engine = KataGoEngine(
    '/usr/local/bin/katago',
    '/path/to/model.bin.gz',
    '/path/to/config.cfg'
)

# 分析一个局面
result = engine.analyze([
    ['B', 'Q16'],
    ['W', 'D4'],
    ['B', 'Q4'],
    ['W', 'D16']
])

# 打印最佳着法
best_move = result['moveInfos'][0]
print(f"最佳着法：{best_move['move']}")
print(f"胜率：{best_move['winrate']:.1%}")
print(f"领先目数：{best_move['scoreLead']:.1f}")

engine.close()
```

### 完整示例：Node.js 集成

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

// 使用示例
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

  console.log('最佳着法:', result.moveInfos[0].move);
  console.log('胜率:', (result.moveInfos[0].winrate * 100).toFixed(1) + '%');

  engine.close();
}

main();
```

## 坐标系统

KataGo 使用标准的围棋坐标系统：

### 字母坐标

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

注意：没有 I 这个字母（避免与数字 1 混淆）。

### 坐标转换

```python
def coord_to_gtp(x, y, board_size=19):
    """将 (x, y) 坐标转换为 GTP 格式"""
    letters = 'ABCDEFGHJKLMNOPQRST'
    return f"{letters[x]}{board_size - y}"

def gtp_to_coord(gtp_coord, board_size=19):
    """将 GTP 坐标转换为 (x, y)"""
    letters = 'ABCDEFGHJKLMNOPQRST'
    x = letters.index(gtp_coord[0].upper())
    y = board_size - int(gtp_coord[1:])
    return (x, y)
```

## 常见使用模式

### 对弈模式

```bash
# 启动 GTP 模式
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

### 批量分析模式

```python
# 分析一盘棋的所有着法
sgf_moves = parse_sgf('game.sgf')

for i in range(len(sgf_moves)):
    result = engine.analyze(sgf_moves[:i+1])
    winrate = result['rootInfo']['winrate']
    print(f"手 {i+1}: 胜率 {winrate:.1%}")
```

### 即时分析模式

使用 `kata-analyze` 进行即时分析：

```
kata-analyze black 1000 50
```

会每 0.5 秒输出一次分析结果，直到达到 1000 次访问。

## 性能调优

### 搜索设置

```ini
# 增加搜索量提高准确度
maxVisits = 1000

# 或使用时间控制
maxTime = 10  # 每手最多思考 10 秒
```

### 多线程设置

```ini
# CPU 线程数
numSearchThreads = 8

# GPU 批量处理
numNNServerThreadsPerModel = 2
nnMaxBatchSize = 16
```

### 内存设置

```ini
# 减少内存使用
nnCacheSizePowerOfTwo = 20  # 默认 23
```

## 下一步

了解指令使用后，如果你想深入研究 KataGo 的实现，请继续阅读 [源代码架构](./architecture.md)。
