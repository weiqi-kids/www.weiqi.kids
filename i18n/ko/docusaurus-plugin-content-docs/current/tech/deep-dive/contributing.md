---
sidebar_position: 4
title: 오픈소스 커뮤니티 참여
description: KataGo 오픈소스 커뮤니티에 참여하여 연산 능력 또는 코드 기여하기
---

# 오픈소스 커뮤니티 참여

KataGo는 활발한 오픈소스 프로젝트로, 다양한 방법으로 기여에 참여할 수 있습니다.

---

## 기여 방법 개요

| 방법 | 난이도 | 요구사항 |
|------|------|------|
| **연산 능력 기여** | 낮음 | GPU가 있는 컴퓨터 |
| **이슈 보고** | 낮음 | GitHub 계정 |
| **문서 개선** | 중간 | 기술 내용에 익숙함 |
| **코드 기여** | 높음 | C++/Python 개발 능력 |

---

## 연산 능력 기여: 분산 학습

### KataGo Training 소개

KataGo Training은 전 세계 분산 학습 네트워크입니다:

- 자원봉사자가 GPU 연산 능력을 기여하여 자가 대국 실행
- 자가 대국 데이터를 중앙 서버에 업로드
- 서버가 정기적으로 새 모델 학습
- 새 모델을 자원봉사자에게 배포하여 대국 계속

공식 웹사이트: https://katagotraining.org/

### 참여 단계

#### 1. 계정 생성

https://katagotraining.org/에서 계정 등록.

#### 2. KataGo 다운로드

```bash
# 최신 버전 다운로드
wget https://github.com/lightvector/KataGo/releases/download/v1.15.3/katago-v1.15.3-cuda11.1-linux-x64.zip
unzip katago-v1.15.3-cuda11.1-linux-x64.zip
```

#### 3. contribute 모드 설정

```bash
# 첫 실행 시 설정 안내
./katago contribute -username YOUR_USERNAME -password YOUR_PASSWORD
```

시스템이 자동으로:
- 최신 모델 다운로드
- 자가 대국 실행
- 대국 데이터 업로드

#### 4. 백그라운드 실행

```bash
# screen 또는 tmux로 백그라운드 실행
screen -S katago
./katago contribute -username YOUR_USERNAME -password YOUR_PASSWORD
# Ctrl+A, D로 screen 나가기
```

### 기여 통계

https://katagotraining.org/contributions/에서 확인 가능:
- 기여 순위
- 총 기여 대국 수
- 최근 학습된 모델

---

## 이슈 보고

### 어디서 보고

- **GitHub Issues**: https://github.com/lightvector/KataGo/issues
- **Discord**: https://discord.gg/bqkZAz3

### 좋은 이슈 보고서 포함 내용

1. **KataGo 버전**: `katago version`
2. **운영체제**: Windows/Linux/macOS
3. **하드웨어**: GPU 모델, 메모리
4. **전체 오류 메시지**: 전체 로그 복사
5. **재현 단계**: 문제를 어떻게 트리거하는지

### 예시

```markdown
## 문제 설명
benchmark 실행 시 메모리 부족 오류 발생

## 환경
- KataGo 버전: 1.15.3
- 운영체제: Ubuntu 22.04
- GPU: RTX 3060 12GB
- 모델: kata-b40c256.bin.gz

## 오류 메시지
```
CUDA error: out of memory
```

## 재현 단계
1. `katago benchmark -model kata-b40c256.bin.gz` 실행
2. 약 30초 대기
3. 오류 발생
```

---

## 문서 개선

### 문서 위치

- **README**: `README.md`
- **GTP 문서**: `docs/GTP_Extensions.md`
- **Analysis 문서**: `docs/Analysis_Engine.md`
- **학습 문서**: `python/README.md`

### 기여 프로세스

1. 프로젝트 Fork
2. 새 브랜치 생성
3. 문서 수정
4. Pull Request 제출

```bash
git clone https://github.com/YOUR_USERNAME/KataGo.git
cd KataGo
git checkout -b improve-docs
# 문서 편집
git add .
git commit -m "Improve documentation for Analysis Engine"
git push origin improve-docs
# GitHub에서 Pull Request 생성
```

---

## 코드 기여

### 개발 환경 설정

```bash
# 프로젝트 복제
git clone https://github.com/lightvector/KataGo.git
cd KataGo

# 컴파일 (Debug 모드)
cd cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=OPENCL -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# 테스트 실행
./katago runtests
```

### 코딩 스타일

KataGo는 다음 코딩 스타일을 사용합니다:

**C++**:
- 2 스페이스 들여쓰기
- 중괄호 같은 줄
- 변수명은 camelCase
- 클래스명은 PascalCase

```cpp
class ExampleClass {
public:
  void exampleMethod() {
    int localVariable = 0;
    if(condition) {
      doSomething();
    }
  }
};
```

**Python**:
- PEP 8 준수
- 4 스페이스 들여쓰기

### 기여 영역

| 영역 | 파일 위치 | 필요 기술 |
|------|---------|---------|
| 핵심 엔진 | `cpp/` | C++, CUDA/OpenCL |
| 학습 프로그램 | `python/` | Python, PyTorch |
| GTP 프로토콜 | `cpp/command/gtp.cpp` | C++ |
| Analysis API | `cpp/command/analysis.cpp` | C++, JSON |
| 테스트 | `cpp/tests/` | C++ |

### Pull Request 프로세스

1. **Issue 생성**: 먼저 하고 싶은 변경사항 논의
2. **Fork & Clone**: 자신의 브랜치 생성
3. **개발 및 테스트**: 모든 테스트 통과 확인
4. **PR 제출**: 변경 내용 상세히 기술
5. **Code Review**: 관리자 피드백에 대응
6. **병합**: 관리자가 코드 병합

### PR 예시

```markdown
## 변경 설명
New Zealand 규칙 지원 추가

## 변경 내용
- rules.cpp에 NEW_ZEALAND 규칙 추가
- GTP 명령어 `kata-set-rules nz` 지원 업데이트
- 단위 테스트 추가

## 테스트 결과
- 모든 기존 테스트 통과
- 새 테스트 통과

## 관련 Issue
Fixes #123
```

---

## 커뮤니티 리소스

### 공식 링크

| 리소스 | 링크 |
|------|------|
| GitHub | https://github.com/lightvector/KataGo |
| Discord | https://discord.gg/bqkZAz3 |
| 학습 네트워크 | https://katagotraining.org/ |

### 토론 공간

- **Discord**: 실시간 토론, 기술 Q&A
- **GitHub Discussions**: 장문 토론, 기능 제안
- **Reddit r/baduk**: 일반 바둑 AI 토론

### 관련 프로젝트

| 프로젝트 | 설명 | 링크 |
|------|------|------|
| KaTrain | 교육 분석 도구 | github.com/sanderland/katrain |
| Lizzie | 분석 인터페이스 | github.com/featurecat/lizzie |
| Sabaki | 기보 편집기 | sabaki.yichuanshen.de |
| BadukAI | 온라인 분석 | baduk.ai |

---

## 인정과 보상

### 기여자 명단

모든 기여자는 다음에 등록됩니다:
- GitHub Contributors 페이지
- KataGo Training 기여 순위표

### 학습 수확

오픈소스 프로젝트 참여의 수확:
- 산업급 AI 시스템 아키텍처 학습
- 전 세계 개발자와 교류
- 오픈소스 기여 기록 축적
- 바둑 AI 기술 심층 이해

---

## 추가 읽기

- [소스 코드 가이드](../source-code) — 코드 구조 이해
- [KataGo 학습 메커니즘 분석](../training) — 로컬 학습 실험
- [바둑 AI 한 편으로 이해하기](../../how-it-works/) — 기술 원리
