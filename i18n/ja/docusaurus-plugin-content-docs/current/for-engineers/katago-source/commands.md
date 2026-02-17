---
sidebar_position: 2
title: よく使うコマンド
---

# KataGoよく使うコマンド

本文ではKataGoの2つの主要な操作モード：GTPプロトコルとAnalysis Engine、およびよく使うコマンドの詳細な説明を紹介します。

## GTPプロトコル紹介

GTP（Go Text Protocol）は囲碁プログラム間の通信の標準プロトコルです。ほとんどの囲碁GUI（Sabaki、Lizzieなど）はGTPでAIエンジンと通信します。

### GTPモードの起動

```bash
katago gtp -model /path/to/model.bin.gz -config /path/to/config.cfg
```

### GTPプロトコル基本フォーマット

```
[id] command_name [arguments]
```

- `id`：オプションのコマンド番号、レスポンス追跡用
- `command_name`：コマンド名
- `arguments`：コマンド引数

レスポンスフォーマット：
```
=[id] response_data     # 成功
?[id] error_message     # 失敗
```

### 基本例

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

## よく使うGTPコマンド

### プログラム情報

| コマンド | 説明 | 例 |
|------|------|------|
| `name` | プログラム名を取得 | `name` → `= KataGo` |
| `version` | バージョン番号を取得 | `version` → `= 1.15.3` |
| `list_commands` | サポートされているすべてのコマンドを一覧 | `list_commands` |
| `protocol_version` | GTPプロトコルバージョン | `protocol_version` → `= 2` |

### 碁盤設定

```
# 碁盤サイズを設定（9、13、19）
boardsize 19

# コミを設定
komi 7.5

# 碁盤をクリア
clear_board

# ルールを設定（KataGo拡張）
kata-set-rules chinese    # 中国ルール
kata-set-rules japanese   # 日本ルール
kata-set-rules tromp-taylor
```

### 着手関連

```
# 着手
play black Q16    # 黒がQ16に打つ
play white D4     # 白がD4に打つ
play black pass   # 黒がパス

# AIに一手打たせる
genmove black     # 黒の一手を生成
genmove white     # 白の一手を生成

# 取り消し
undo              # 一手取り消し

# 手数制限を設定
kata-set-param maxVisits 1000    # 最大探索回数を設定
```

### 局面クエリ

```
# 碁盤を表示
showboard

# 現在の手番を取得
kata-get-player

# 分析結果を取得
kata-analyze black 100    # 黒を分析、100回探索
```

### ルール関連

```
# 現在のルールを取得
kata-get-rules

# ルールを設定
kata-set-rules chinese

# 置き石を設定
fixed_handicap 4     # 標準四子局の位置
place_free_handicap 4  # 自由置き石
```

## KataGo拡張コマンド

KataGoは標準GTP以外にも多くの拡張コマンドを提供：

### kata-analyze

現在局面をリアルタイム分析：

```
kata-analyze [player] [visits] [interval]
```

パラメータ：
- `player`：どちら側を分析（black/white）
- `visits`：探索回数
- `interval`：報告間隔（centiseconds、1/100秒）

例：
```
kata-analyze black 1000 100
```

出力：
```
info move Q3 visits 523 winrate 0.5432 scoreMean 2.31 scoreSelfplay 2.45 prior 0.1234 order 0 pv Q3 R4 Q5 ...
info move D4 visits 312 winrate 0.5123 scoreMean 1.82 scoreSelfplay 1.95 prior 0.0987 order 1 pv D4 C6 E3 ...
...
```

出力フィールド説明：

| フィールド | 説明 |
|------|------|
| `move` | 着点 |
| `visits` | 探索訪問回数 |
| `winrate` | 勝率（0-1） |
| `scoreMean` | 予想目数差 |
| `scoreSelfplay` | 自己対局予想目数 |
| `prior` | ニューラルネットワークの事前確率 |
| `order` | ランキング順序 |
| `pv` | 主要変化（Principal Variation） |

### kata-raw-nn

生のニューラルネットワーク出力を取得：

```
kata-raw-nn [symmetry]
```

出力に含まれるもの：
- Policy確率分布
- Value予測
- 領地予測など

### kata-debug-print

詳細な探索情報を表示、デバッグ用：

```
kata-debug-print move Q16
```

### 棋力調整

```
# 最大訪問回数を設定
kata-set-param maxVisits 100      # 弱め
kata-set-param maxVisits 10000    # 強め

# 思考時間を設定
kata-time-settings main 60 0      # 各方60秒
kata-time-settings byoyomi 30 5   # 秒読み30秒5回
```

## Analysis Engineの使用

Analysis EngineはKataGoが提供するもう一つの操作モードで、JSON形式で通信し、プログラム的な使用により適しています。

### Analysis Engineの起動

```bash
katago analysis -model /path/to/model.bin.gz -config /path/to/config.cfg
```

### 基本使用フロー

```
あなたのプログラム ──JSONリクエスト──> KataGo Analysis Engine ──JSONレスポンス──> あなたのプログラム
```

### リクエストフォーマット

各リクエストは1行のJSONオブジェクト：

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

### リクエストフィールド説明

| フィールド | 必須 | 説明 |
|------|------|------|
| `id` | はい | クエリ識別子、レスポンスとの対応用 |
| `moves` | いいえ | 着手シーケンス `[["B","Q16"],["W","D4"]]` |
| `initialStones` | いいえ | 初期石 `[["B","Q16"],["W","D4"]]` |
| `rules` | はい | ルール名 |
| `komi` | はい | コミ |
| `boardXSize` | はい | 碁盤幅 |
| `boardYSize` | はい | 碁盤高さ |
| `analyzeTurns` | いいえ | 分析する手数（0インデックス） |
| `maxVisits` | いいえ | 設定ファイルのmaxVisitsを上書き |

### レスポンスフォーマット

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

### レスポンスフィールド説明

#### moveInfosフィールド

| フィールド | 説明 |
|------|------|
| `move` | 着点座標 |
| `visits` | その着点の探索訪問回数 |
| `winrate` | 勝率（0-1、現在の手番に対して） |
| `scoreMean` | 予想最終目数差 |
| `scoreStdev` | 目数標準偏差 |
| `scoreLead` | 現在のリード目数 |
| `prior` | ニューラルネットワーク事前確率 |
| `order` | ランキング（0 = 最善） |
| `pv` | 主要変化シーケンス |

#### rootInfoフィールド

| フィールド | 説明 |
|------|------|
| `visits` | 総探索訪問回数 |
| `winrate` | 現在局面の勝率 |
| `scoreLead` | 現在のリード目数 |
| `scoreSelfplay` | 自己対局予想目数 |

#### ownershipフィールド

1次元配列、長さはboardXSize × boardYSize、各値は-1から1の間：
- -1：白方の領地と予測
- +1：黒方の領地と予測
- 0：未定/境界

### 高度なクエリオプション

#### 領地マップの取得

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

#### Policy分布の取得

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

#### 報告する着手数の制限

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

#### 特定の着手を分析

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

### 完全な例：Python統合

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

        # クエリを送信
        self.process.stdin.write(json.dumps(query) + '\n')
        self.process.stdin.flush()

        # レスポンスを読み取り
        response_line = self.process.stdout.readline()
        return json.loads(response_line)

    def close(self):
        self.process.terminate()


# 使用例
engine = KataGoEngine(
    '/usr/local/bin/katago',
    '/path/to/model.bin.gz',
    '/path/to/config.cfg'
)

# 局面を分析
result = engine.analyze([
    ['B', 'Q16'],
    ['W', 'D4'],
    ['B', 'Q4'],
    ['W', 'D16']
])

# 最善手を表示
best_move = result['moveInfos'][0]
print(f"最善手：{best_move['move']}")
print(f"勝率：{best_move['winrate']:.1%}")
print(f"リード目数：{best_move['scoreLead']:.1f}")

engine.close()
```

### 完全な例：Node.js統合

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

// 使用例
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

  console.log('最善手:', result.moveInfos[0].move);
  console.log('勝率:', (result.moveInfos[0].winrate * 100).toFixed(1) + '%');

  engine.close();
}

main();
```

## 座標システム

KataGoは標準的な囲碁座標システムを使用：

### アルファベット座標

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

注意：Iという文字はありません（数字の1との混同を避けるため）。

### 座標変換

```python
def coord_to_gtp(x, y, board_size=19):
    """(x, y)座標をGTPフォーマットに変換"""
    letters = 'ABCDEFGHJKLMNOPQRST'
    return f"{letters[x]}{board_size - y}"

def gtp_to_coord(gtp_coord, board_size=19):
    """GTP座標を(x, y)に変換"""
    letters = 'ABCDEFGHJKLMNOPQRST'
    x = letters.index(gtp_coord[0].upper())
    y = board_size - int(gtp_coord[1:])
    return (x, y)
```

## よく使う使用パターン

### 対局モード

```bash
# GTPモードを起動
katago gtp -model model.bin.gz -config gtp.cfg

# GTPコマンドシーケンス
boardsize 19
komi 7.5
play black Q16
genmove white
play black Q4
genmove white
...
```

### バッチ分析モード

```python
# 一局のすべての着手を分析
sgf_moves = parse_sgf('game.sgf')

for i in range(len(sgf_moves)):
    result = engine.analyze(sgf_moves[:i+1])
    winrate = result['rootInfo']['winrate']
    print(f"手{i+1}: 勝率{winrate:.1%}")
```

### リアルタイム分析モード

`kata-analyze`を使用してリアルタイム分析：

```
kata-analyze black 1000 50
```

1000回の訪問に達するまで、0.5秒ごとに分析結果を出力します。

## 性能チューニング

### 探索設定

```ini
# 探索量を増やして精度を向上
maxVisits = 1000

# または時間制御を使用
maxTime = 10  # 1手あたり最大10秒思考
```

### マルチスレッド設定

```ini
# CPUスレッド数
numSearchThreads = 8

# GPUバッチ処理
numNNServerThreadsPerModel = 2
nnMaxBatchSize = 16
```

### メモリ設定

```ini
# メモリ使用量を削減
nnCacheSizePowerOfTwo = 20  # デフォルト23
```

## 次のステップ

コマンドの使用を理解した後、KataGoの実装を深く研究したい場合は、[ソースコードアーキテクチャ](./architecture.md)を続けてお読みください。

