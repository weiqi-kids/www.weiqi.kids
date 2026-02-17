---
sidebar_position: 2
title: 자주 사용하는 명령어
---

# KataGo 자주 사용하는 명령어

본문에서는 KataGo의 두 가지 주요 조작 모드인 GTP 프로토콜과 Analysis Engine, 그리고 자주 사용하는 명령어의 상세 설명을 소개합니다.

## GTP 프로토콜 소개

GTP(Go Text Protocol)는 바둑 프로그램 간 통신의 표준 프로토콜입니다. 대부분의 바둑 GUI(Sabaki, Lizzie 등)가 GTP로 AI 엔진과 통신합니다.

### GTP 모드 시작

```bash
katago gtp -model /path/to/model.bin.gz -config /path/to/config.cfg
```

### GTP 프로토콜 기본 형식

```
[id] command_name [arguments]
```

- `id`: 선택적 명령 번호, 응답 추적용
- `command_name`: 명령 이름
- `arguments`: 명령 인자

응답 형식:
```
=[id] response_data     # 성공
?[id] error_message     # 실패
```

### 기본 예제

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

## 자주 쓰는 GTP 명령어

### 프로그램 정보

| 명령어 | 설명 | 예제 |
|------|------|------|
| `name` | 프로그램 이름 가져오기 | `name` → `= KataGo` |
| `version` | 버전 번호 가져오기 | `version` → `= 1.15.3` |
| `list_commands` | 지원하는 모든 명령어 나열 | `list_commands` |
| `protocol_version` | GTP 프로토콜 버전 | `protocol_version` → `= 2` |

### 바둑판 설정

```
# 바둑판 크기 설정(9, 13, 19)
boardsize 19

# 덤 설정
komi 7.5

# 바둑판 초기화
clear_board

# 규칙 설정(KataGo 확장)
kata-set-rules chinese    # 중국 규칙
kata-set-rules japanese   # 일본 규칙
kata-set-rules tromp-taylor
```

### 착수 관련

```
# 착수
play black Q16    # 흑이 Q16에 착수
play white D4     # 백이 D4에 착수
play black pass   # 흑 패스

# AI에게 한 수 두게 하기
genmove black     # 흑 수 생성
genmove white     # 백 수 생성

# 취소
undo              # 한 수 취소

# 수 제한 설정
kata-set-param maxVisits 1000    # 최대 탐색 횟수 설정
```

### 국면 조회

```
# 바둑판 표시
showboard

# 현재 차례 가져오기
kata-get-player

# 분석 결과 가져오기
kata-analyze black 100    # 흑 분석, 100회 탐색
```

### 규칙 관련

```
# 현재 규칙 가져오기
kata-get-rules

# 규칙 설정
kata-set-rules chinese

# 접바둑 설정
fixed_handicap 4     # 표준 4점 접바둑 위치
place_free_handicap 4  # 자유 접바둑
```

## KataGo 확장 명령어

KataGo는 표준 GTP 외에 많은 확장 명령어를 제공합니다:

### kata-analyze

현재 국면 실시간 분석:

```
kata-analyze [player] [visits] [interval]
```

파라미터:
- `player`: 어느 쪽 분석(black/white)
- `visits`: 탐색 횟수
- `interval`: 리포트 간격(centiseconds, 1/100초)

예제:
```
kata-analyze black 1000 100
```

출력:
```
info move Q3 visits 523 winrate 0.5432 scoreMean 2.31 scoreSelfplay 2.45 prior 0.1234 order 0 pv Q3 R4 Q5 ...
info move D4 visits 312 winrate 0.5123 scoreMean 1.82 scoreSelfplay 1.95 prior 0.0987 order 1 pv D4 C6 E3 ...
...
```

출력 필드 설명:

| 필드 | 설명 |
|------|------|
| `move` | 착점 |
| `visits` | 탐색 방문 횟수 |
| `winrate` | 승률(0-1) |
| `scoreMean` | 예상 집수 차 |
| `scoreSelfplay` | 자가대국 예상 집수 |
| `prior` | 신경망의 사전 확률 |
| `order` | 순위 |
| `pv` | 주요 변화(Principal Variation) |

### kata-raw-nn

원시 신경망 출력 가져오기:

```
kata-raw-nn [symmetry]
```

출력 포함:
- Policy 확률 분포
- Value 예측
- 영역 예측 등

### kata-debug-print

상세 탐색 정보 표시, 디버깅용:

```
kata-debug-print move Q16
```

### 기력 조정

```
# 최대 방문 횟수 설정
kata-set-param maxVisits 100      # 약함
kata-set-param maxVisits 10000    # 강함

# 사고 시간 설정
kata-time-settings main 60 0      # 각 60초
kata-time-settings byoyomi 30 5   # 초읽기 30초 5회
```

## Analysis Engine 사용

Analysis Engine은 KataGo가 제공하는 또 다른 조작 모드로, JSON 형식 통신을 사용하여 프로그래밍에 더 적합합니다.

### Analysis Engine 시작

```bash
katago analysis -model /path/to/model.bin.gz -config /path/to/config.cfg
```

### 기본 사용 흐름

```
당신의 프로그램 ──JSON요청──> KataGo Analysis Engine ──JSON응답──> 당신의 프로그램
```

### 요청 형식

각 요청은 한 줄을 차지하는 JSON 객체:

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

### 요청 필드 설명

| 필드 | 필수 | 설명 |
|------|------|------|
| `id` | 예 | 쿼리 식별자, 응답 매칭용 |
| `moves` | 아니오 | 수순 `[["B","Q16"],["W","D4"]]` |
| `initialStones` | 아니오 | 초기 돌 `[["B","Q16"],["W","D4"]]` |
| `rules` | 예 | 규칙 이름 |
| `komi` | 예 | 덤 |
| `boardXSize` | 예 | 바둑판 너비 |
| `boardYSize` | 예 | 바둑판 높이 |
| `analyzeTurns` | 아니오 | 분석할 수순(0-indexed) |
| `maxVisits` | 아니오 | 설정 파일의 maxVisits 덮어쓰기 |

### 응답 형식

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

### 응답 필드 설명

#### moveInfos 필드

| 필드 | 설명 |
|------|------|
| `move` | 착점 좌표 |
| `visits` | 해당 착점의 탐색 방문 횟수 |
| `winrate` | 승률(0-1, 현재 차례 기준) |
| `scoreMean` | 예상 최종 집수 차 |
| `scoreStdev` | 집수 표준편차 |
| `scoreLead` | 현재 리드 집수 |
| `prior` | 신경망 사전 확률 |
| `order` | 순위(0 = 최선) |
| `pv` | 주요 변화 시퀀스 |

#### rootInfo 필드

| 필드 | 설명 |
|------|------|
| `visits` | 총 탐색 방문 횟수 |
| `winrate` | 현재 국면 승률 |
| `scoreLead` | 현재 리드 집수 |
| `scoreSelfplay` | 자가대국 예상 집수 |

#### ownership 필드

1차원 배열, 길이 boardXSize × boardYSize, 각 값은 -1에서 1 사이:
- -1: 백 영역 예측
- +1: 흑 영역 예측
- 0: 미정/경계

### 고급 쿼리 옵션

#### 영역 맵 가져오기

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

#### Policy 분포 가져오기

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

#### 리포트 수 제한

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

#### 특정 수 분석

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

### 완전 예제: Python 통합

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

        # 쿼리 전송
        self.process.stdin.write(json.dumps(query) + '\n')
        self.process.stdin.flush()

        # 응답 읽기
        response_line = self.process.stdout.readline()
        return json.loads(response_line)

    def close(self):
        self.process.terminate()


# 사용 예제
engine = KataGoEngine(
    '/usr/local/bin/katago',
    '/path/to/model.bin.gz',
    '/path/to/config.cfg'
)

# 국면 분석
result = engine.analyze([
    ['B', 'Q16'],
    ['W', 'D4'],
    ['B', 'Q4'],
    ['W', 'D16']
])

# 최선수 출력
best_move = result['moveInfos'][0]
print(f"최선수: {best_move['move']}")
print(f"승률: {best_move['winrate']:.1%}")
print(f"리드 집수: {best_move['scoreLead']:.1f}")

engine.close()
```

### 완전 예제: Node.js 통합

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

// 사용 예제
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

  console.log('최선수:', result.moveInfos[0].move);
  console.log('승률:', (result.moveInfos[0].winrate * 100).toFixed(1) + '%');

  engine.close();
}

main();
```

## 좌표 시스템

KataGo는 표준 바둑 좌표 시스템을 사용합니다:

### 문자 좌표

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

주의: I 문자가 없습니다(숫자 1과 혼동 방지).

### 좌표 변환

```python
def coord_to_gtp(x, y, board_size=19):
    """(x, y) 좌표를 GTP 형식으로 변환"""
    letters = 'ABCDEFGHJKLMNOPQRST'
    return f"{letters[x]}{board_size - y}"

def gtp_to_coord(gtp_coord, board_size=19):
    """GTP 좌표를 (x, y)로 변환"""
    letters = 'ABCDEFGHJKLMNOPQRST'
    x = letters.index(gtp_coord[0].upper())
    y = board_size - int(gtp_coord[1:])
    return (x, y)
```

## 흔한 사용 패턴

### 대국 모드

```bash
# GTP 모드 시작
katago gtp -model model.bin.gz -config gtp.cfg

# GTP 명령어 시퀀스
boardsize 19
komi 7.5
play black Q16
genmove white
play black Q4
genmove white
...
```

### 배치 분석 모드

```python
# 한 대국의 모든 수 분석
sgf_moves = parse_sgf('game.sgf')

for i in range(len(sgf_moves)):
    result = engine.analyze(sgf_moves[:i+1])
    winrate = result['rootInfo']['winrate']
    print(f"수 {i+1}: 승률 {winrate:.1%}")
```

### 실시간 분석 모드

`kata-analyze`로 실시간 분석:

```
kata-analyze black 1000 50
```

1000회 방문에 도달할 때까지 0.5초마다 분석 결과 출력.

## 성능 튜닝

### 탐색 설정

```ini
# 탐색량 늘려 정확도 향상
maxVisits = 1000

# 또는 시간 제어 사용
maxTime = 10  # 매 수 최대 10초 사고
```

### 멀티스레드 설정

```ini
# CPU 스레드 수
numSearchThreads = 8

# GPU 배치 처리
numNNServerThreadsPerModel = 2
nnMaxBatchSize = 16
```

### 메모리 설정

```ini
# 메모리 사용량 줄이기
nnCacheSizePowerOfTwo = 20  # 기본 23
```

## 다음 단계

명령어 사용을 이해했다면 KataGo 구현을 심층 연구하고 싶으시면 [소스코드 아키텍처](./architecture.md)를 계속 읽으세요.

