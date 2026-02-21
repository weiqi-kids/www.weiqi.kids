---
sidebar_position: 3
title: 基本使用
description: GTP 協議與 Analysis Engine 的完整使用指南
---

# KataGo 基本使用

KataGo 提供兩種操作模式：GTP 模式（適合對弈）和 Analysis Engine（適合程式整合）。

---

## GTP 模式

GTP（Go Text Protocol）是圍棋程式間通訊的標準協議。

### 啟動

```bash
katago gtp -model kata-b18c384.bin.gz -config my_config.cfg
```

### 指令格式

```
[id] command_name [arguments]
```

回應格式：
```
=[id] response_data     # 成功
?[id] error_message     # 失敗
```

### 常用指令

#### 程式資訊

```
name
= KataGo

version
= 1.15.3

list_commands
= name
version
boardsize
...
```

#### 棋盤設定

```
boardsize 19
=

komi 7.5
=

clear_board
=

kata-set-rules chinese
=
```

#### 下棋相關

```
play black Q16
=

play white D4
=

genmove black
= Q4

undo
=

showboard
=
```

### KataGo 擴充指令

#### kata-analyze

即時分析當前局面：

```
kata-analyze black 1000 100
```

參數：
- 顏色（black/white）
- 搜索次數
- 報告間隔（1/100 秒）

輸出：

```
info move Q3 visits 523 winrate 0.5432 scoreMean 2.31 prior 0.1234 pv Q3 R4 Q5
info move D4 visits 312 winrate 0.5123 scoreMean 1.82 prior 0.0987 pv D4 C6 E3
```

#### 輸出欄位說明

| 欄位 | 說明 |
|------|------|
| `move` | 著點 |
| `visits` | 搜索訪問次數 |
| `winrate` | 勝率（0-1） |
| `scoreMean` | 預期目數差 |
| `prior` | 神經網路先驗機率 |
| `pv` | 主要變化（Principal Variation） |

#### kata-raw-nn

取得原始神經網路輸出：

```
kata-raw-nn 0
```

#### 棋力調整

```
kata-set-param maxVisits 100      # 較弱
kata-set-param maxVisits 10000    # 較強
```

---

## Analysis Engine

Analysis Engine 使用 JSON 格式通訊，更適合程式化使用。

### 啟動

```bash
katago analysis -model kata-b18c384.bin.gz -config my_config.cfg
```

### 請求格式

每個請求是一行 JSON：

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

### 請求欄位

| 欄位 | 必須 | 說明 |
|------|------|------|
| `id` | 是 | 查詢識別碼 |
| `moves` | 否 | 棋步序列 |
| `initialStones` | 否 | 初始棋子 |
| `rules` | 是 | 規則名稱 |
| `komi` | 是 | 貼目 |
| `boardXSize` | 是 | 棋盤寬度 |
| `boardYSize` | 是 | 棋盤高度 |
| `analyzeTurns` | 否 | 要分析的手數 |
| `maxVisits` | 否 | 覆蓋設定的搜索次數 |

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
      "prior": 0.0892,
      "order": 0,
      "pv": ["D16", "Q10", "R14"]
    }
  ],
  "rootInfo": {
    "visits": 5000,
    "winrate": 0.5234,
    "scoreLead": 2.1
  }
}
```

### 回應欄位

#### moveInfos

| 欄位 | 說明 |
|------|------|
| `move` | 著點座標 |
| `visits` | 搜索訪問次數 |
| `winrate` | 勝率 |
| `scoreMean` | 預期目數差 |
| `prior` | 神經網路先驗機率 |
| `order` | 排名（0 = 最佳） |
| `pv` | 主要變化序列 |

#### rootInfo

| 欄位 | 說明 |
|------|------|
| `visits` | 總搜索訪問次數 |
| `winrate` | 當前局面勝率 |
| `scoreLead` | 當前領先目數 |

### 進階選項

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

`ownership` 陣列：每個值在 -1（白方）到 +1（黑方）之間。

#### 取得 Policy 分佈

```json
{
  "includePolicy": true
}
```

#### 限制回報數量

```json
{
  "maxMoves": 5
}
```

---

## 座標系統

```
   A B C D E F G H J K L M N O P Q R S T
19 . . . . . . . . . . . . . . . . . . . 19
18 . . . . . . . . . . . . . . . . . . . 18
...
 1 . . . . . . . . . . . . . . . . . . .  1
   A B C D E F G H J K L M N O P Q R S T
```

**注意**：沒有 I 這個字母（避免與數字 1 混淆）。

### 座標轉換

```python
def coord_to_gtp(x, y, board_size=19):
    letters = 'ABCDEFGHJKLMNOPQRST'
    return f"{letters[x]}{board_size - y}"

def gtp_to_coord(gtp_coord, board_size=19):
    letters = 'ABCDEFGHJKLMNOPQRST'
    x = letters.index(gtp_coord[0].upper())
    y = board_size - int(gtp_coord[1:])
    return (x, y)
```

---

## 常見使用模式

### 對弈模式

```
boardsize 19
komi 7.5
play black Q16
genmove white
play black Q4
genmove white
...
```

### 批次分析模式

分析一盤棋的所有著法：

```python
for i in range(len(moves)):
    result = engine.analyze(moves[:i+1])
    print(f"手 {i+1}: 勝率 {result['rootInfo']['winrate']:.1%}")
```

### 即時分析模式

```
kata-analyze black 1000 50
```

每 0.5 秒輸出一次分析結果。

---

## 效能調優

### 搜索設定

```ini
maxVisits = 1000      # 增加準確度
maxTime = 10          # 每手最多 10 秒
```

### 多執行緒

```ini
numSearchThreads = 8
numNNServerThreadsPerModel = 2
nnMaxBatchSize = 16
```

---

## 延伸閱讀

- [整合到你的專案](../integration) — Python/Node.js API
- [KataGo 的關鍵創新](../../how-it-works/katago-innovations) — 了解技術細節
