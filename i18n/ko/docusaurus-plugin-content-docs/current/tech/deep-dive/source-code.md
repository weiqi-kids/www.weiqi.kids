---
sidebar_position: 2
title: KataGo 소스 코드 가이드
description: KataGo 코드 구조, 핵심 모듈 및 아키텍처 설계
---

# KataGo 소스 코드 가이드

이 문서는 KataGo의 코드 구조를 안내하며, 깊이 연구하거나 코드에 기여하고자 하는 엔지니어에게 적합합니다.

---

## 소스 코드 가져오기

```bash
git clone https://github.com/lightvector/KataGo.git
cd KataGo
```

---

## 디렉토리 구조

```
KataGo/
├── cpp/                    # C++ 핵심 엔진
│   ├── main.cpp            # 메인 프로그램 진입점
│   ├── command/            # 명령어 처리
│   ├── core/               # 핵심 유틸리티
│   ├── game/               # 바둑 규칙
│   ├── search/             # MCTS 탐색
│   ├── neuralnet/          # 신경망 추론
│   ├── dataio/             # 데이터 I/O
│   └── tests/              # 단위 테스트
│
├── python/                 # Python 학습 코드
│   ├── train.py            # 학습 메인 프로그램
│   ├── model.py            # 네트워크 아키텍처 정의
│   ├── data/               # 데이터 처리
│   └── configs/            # 학습 설정
│
└── docs/                   # 문서
```

---

## 핵심 모듈 분석

### 1. game/ — 바둑 규칙

바둑 규칙의 완전한 구현.

#### board.h / board.cpp

```cpp
// 바둑판 상태 표현
class Board {
public:
    static constexpr int MAX_BOARD_SIZE = 19;

    // 바둑판 상태
    Color colors[MAX_ARR_SIZE];  // 각 위치의 색상
    Chain chains[MAX_ARR_SIZE];  // 돌 그룹 정보

    // 핵심 연산
    bool playMove(Loc loc, Player pla);  // 한 수 착수
    bool isLegal(Loc loc, Player pla);   // 합법성 판단
    void calculateArea(Color* area);      // 영역 계산
};
```

**애니메이션 대응**:
- A2 격자 모델: 바둑판의 데이터 구조
- A6 연결 영역: 돌 그룹(Chain) 표현
- A7 활로 계산: liberty 추적

#### rules.h / rules.cpp

```cpp
// 다중 규칙 지원
struct Rules {
    enum KoRule { SIMPLE_KO, POSITIONAL_KO, SITUATIONAL_KO };
    enum ScoringRule { TERRITORY_SCORING, AREA_SCORING };
    enum TaxRule { NO_TAX, TAX_SEKI, TAX_ALL };

    KoRule koRule;
    ScoringRule scoringRule;
    TaxRule taxRule;
    float komi;

    // 규칙 이름 매핑
    static Rules parseRules(const std::string& name);
};
```

지원하는 규칙:
- `chinese`: 중국 규칙 (계자)
- `japanese`: 일본 규칙 (계목)
- `korean`: 한국 규칙
- `aga`: 미국 규칙
- `tromp-taylor`: Tromp-Taylor 규칙

---

### 2. search/ — MCTS 탐색

몬테카를로 트리 탐색 구현.

#### search.h / search.cpp

```cpp
class Search {
public:
    // 핵심 탐색
    void runWholeSearch(Player pla);

    // MCTS 단계
    void selectNode();           // 노드 선택
    void expandNode();           // 노드 확장
    void evaluateNode();         // 신경망 평가
    void backpropValue();        // 역전파 업데이트

    // 결과 가져오기
    Loc getChosenMove();
    std::vector<MoveInfo> getSortedMoveInfos();
};
```

**애니메이션 대응**:
- C5 MCTS 4단계: select → expand → evaluate → backprop에 대응
- E4 PUCT 공식: `selectNode()`에서 구현

#### searchparams.h

```cpp
struct SearchParams {
    // 탐색 제어
    int64_t maxVisits;          // 최대 방문 횟수
    double maxTime;             // 최대 시간

    // PUCT 파라미터
    double cpuctExploration;    // 탐색 상수
    double cpuctBase;

    // 가상 손실
    int virtualLoss;

    // 루트 노드 노이즈
    double rootNoiseEnabled;
    double rootDirichletAlpha;
};
```

---

### 3. neuralnet/ — 신경망 추론

신경망 추론 엔진.

#### nninputs.h / nninputs.cpp

```cpp
// 신경망 입력 특성
class NNInputs {
public:
    // 특성 평면
    static constexpr int NUM_FEATURES = 22;

    // 특성 채우기
    static void fillFeatures(
        const Board& board,
        const BoardHistory& hist,
        float* features
    );
};
```

입력 특성 포함:
- 흑돌 위치, 백돌 위치
- 활로 수 (1, 2, 3+)
- 히스토리 착수
- 규칙 인코딩

**애니메이션 대응**:
- A10 히스토리 스택: 다중 프레임 입력
- A11 합법 착수 마스크: 금수 필터링

#### nneval.h / nneval.cpp

```cpp
// 신경망 평가 결과
struct NNOutput {
    // Policy 출력 (362개 위치, pass 포함)
    float policyProbs[NNPos::MAX_NN_POLICY_SIZE];

    // Value 출력
    float winProb;       // 승률
    float lossProb;      // 패률
    float noResultProb;  // 무승부율

    // 보조 출력
    float scoreMean;     // 집수 예측
    float scoreStdev;    // 집수 표준편차
    float lead;          // 리드 집수

    // 영역 예측
    float ownership[NNPos::MAX_BOARD_AREA];
};
```

**애니메이션 대응**:
- E1 정책 네트워크: policyProbs
- E2 가치 네트워크: winProb, scoreMean
- E3 듀얼 헤드 네트워크: 다중 출력 헤드 설계

---

### 4. command/ — 명령어 처리

다양한 실행 모드 구현.

#### gtp.cpp

GTP(Go Text Protocol) 모드 구현:

```cpp
void MainCmds::gtp(const std::vector<std::string>& args) {
    // 명령어 파싱 및 실행
    while(true) {
        std::string line;
        std::getline(std::cin, line);

        if(line == "name") {
            respond("KataGo");
        }
        else if(line.find("play") == 0) {
            // 착수 명령 처리
        }
        else if(line.find("genmove") == 0) {
            // 탐색 실행 및 최선의 착수 반환
        }
        // ... 기타 명령어
    }
}
```

#### analysis.cpp

Analysis Engine 구현:

```cpp
void MainCmds::analysis(const std::vector<std::string>& args) {
    while(true) {
        // JSON 요청 읽기
        std::string line;
        std::getline(std::cin, line);
        json query = json::parse(line);

        // 바둑판 상태 설정
        Board board = setupBoard(query);

        // 분석 실행
        Search search(...);
        search.runWholeSearch();

        // JSON 응답 출력
        json response = formatResponse(search);
        std::cout << response.dump() << std::endl;
    }
}
```

---

## Python 학습 코드

### model.py — 네트워크 아키텍처

```python
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 초기 합성곱
        self.initial_conv = nn.Conv2d(
            in_channels=config.input_features,
            out_channels=config.trunk_channels,
            kernel_size=3, padding=1
        )

        # 잔차 타워
        self.trunk = nn.ModuleList([
            ResidualBlock(config.trunk_channels)
            for _ in range(config.num_blocks)
        ])

        # 출력 헤드
        self.policy_head = PolicyHead(config)
        self.value_head = ValueHead(config)
        self.ownership_head = OwnershipHead(config)

    def forward(self, x):
        # 초기 합성곱
        x = self.initial_conv(x)

        # 잔차 타워
        for block in self.trunk:
            x = block(x)

        # 다중 헤드 출력
        policy = self.policy_head(x)
        value = self.value_head(x)
        ownership = self.ownership_head(x)

        return policy, value, ownership
```

**애니메이션 대응**:
- D9 합성곱 연산: Conv2d
- D12 잔차 연결: ResidualBlock
- E11 잔차 타워: trunk 구조

### train.py — 학습 루프

```python
def train_step(model, optimizer, batch):
    # 순전파
    policy_pred, value_pred, ownership_pred = model(batch.inputs)

    # 손실 계산
    policy_loss = cross_entropy(policy_pred, batch.policy_target)
    value_loss = mse_loss(value_pred, batch.value_target)
    ownership_loss = mse_loss(ownership_pred, batch.ownership_target)

    total_loss = policy_loss + value_loss + ownership_loss

    # 역전파
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()
```

**애니메이션 대응**:
- D3 순전파: model(batch.inputs)
- D13 역전파: total_loss.backward()
- K3 Adam: optimizer.step()

---

## 핵심 알고리즘 구현

### PUCT 선택 공식

```cpp
// search.cpp
double Search::getPUCTScore(const SearchNode* node, int moveIdx) {
    double Q = node->getChildValue(moveIdx);
    double P = node->getChildPolicy(moveIdx);
    double N_parent = node->visits;
    double N_child = node->getChildVisits(moveIdx);

    double exploration = params.cpuctExploration;
    double cpuct = exploration * sqrt(N_parent) / (1.0 + N_child);

    return Q + cpuct * P;
}
```

### 가상 손실

```cpp
// 다중 스레드가 같은 노드를 선택하는 것을 방지
void Search::applyVirtualLoss(SearchNode* node) {
    node->virtualLoss += params.virtualLoss;
}

void Search::removeVirtualLoss(SearchNode* node) {
    node->virtualLoss -= params.virtualLoss;
}
```

**애니메이션 대응**:
- C9 가상 손실: 병렬 탐색 기법

---

## 컴파일과 디버깅

### 컴파일 (Debug 모드)

```bash
cd cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=OPENCL -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

### 단위 테스트

```bash
./katago runtests
```

### 디버깅 팁

```cpp
// 상세 로그 활성화
#define SEARCH_DEBUG 1

// 탐색 중 브레이크포인트 추가
if(node->visits > 1000) {
    // 탐색 상태 검사용 브레이크포인트 설정
}
```

---

## 추가 읽기

- [KataGo 학습 메커니즘 분석](../training) — 전체 학습 프로세스
- [오픈소스 커뮤니티 참여](../contributing) — 기여 가이드
- [개념 빠른 참조표](/docs/animations/) — 109개 개념 대조
