---
sidebar_position: 2
title: KataGo 실전 입문
---

# KataGo 실전 입문 가이드

이 섹션에서는 설치부터 실제 사용까지 KataGo의 모든 실용적인 조작 지식을 다룹니다. KataGo를 자신의 애플리케이션에 통합하든, 소스 코드를 심층 연구하든, 여기가 출발점입니다.

## 왜 KataGo인가?

수많은 바둑 AI 중에서 KataGo는 현재 최선의 선택이며, 이유는 다음과 같습니다:

| 장점 | 설명 |
|------|------|
| **가장 강한 기력** | 공개 테스트에서 지속적으로 최고 수준 유지 |
| **가장 완전한 기능** | 집수 예측, 영역 분석, 다중 규칙 지원 |
| **완전 오픈소스** | MIT 라이선스, 자유롭게 사용 및 수정 가능 |
| **지속적 업데이트** | 활발한 개발과 커뮤니티 지원 |
| **완벽한 문서** | 공식 문서 상세, 커뮤니티 자료 풍부 |
| **다중 플랫폼 지원** | Linux, macOS, Windows 모두 실행 가능 |

## 이 섹션 내용

### [설치 및 설정](./setup.md)

처음부터 KataGo 환경 구축:

- 시스템 요구사항 및 하드웨어 권장
- 각 플랫폼 설치 단계(macOS / Linux / Windows)
- 모델 다운로드 및 선택 가이드
- 설정 파일 상세 설명

### [자주 사용하는 명령어](./commands.md)

KataGo 사용법 숙달:

- GTP(Go Text Protocol) 프로토콜 소개
- 자주 쓰는 GTP 명령어 및 예제
- Analysis Engine 사용법
- JSON API 완전 설명

### [소스코드 아키텍처](./architecture.md)

KataGo의 구현 세부사항 심층 이해:

- 프로젝트 디렉토리 구조 개요
- 신경망 아키텍처 해석
- 탐색 엔진 구현 세부사항
- 훈련 프로세스 개요

## 빠른 시작

KataGo를 빠르게 시험해보고 싶다면 가장 간단한 방법:

### macOS(Homebrew 사용)

```bash
# 설치
brew install katago

# 모델 다운로드(테스트용 작은 모델 선택)
curl -L -o kata-b18c384.bin.gz \
  https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz

# GTP 모드 실행
katago gtp -model kata-b18c384.bin.gz -config gtp_example.cfg
```

### Linux(사전 컴파일 버전)

```bash
# 사전 컴파일 버전 다운로드
wget https://github.com/lightvector/KataGo/releases/download/v1.15.3/katago-v1.15.3-opencl-linux-x64.zip

# 압축 해제
unzip katago-v1.15.3-opencl-linux-x64.zip

# 모델 다운로드
wget https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz

# 실행
./katago gtp -model kata-b18c384nbt-*.bin.gz -config default_gtp.cfg
```

### 설치 확인

성공적으로 시작하면 GTP 프롬프트가 나타납니다. 다음 명령어를 입력해 보세요:

```
name
= KataGo

version
= 1.15.3

boardsize 19
=

genmove black
= Q16
```

## 사용 시나리오 가이드

요구에 따라 권장 읽기 순서와 중점:

### 시나리오 1: 바둑 앱에 통합

자신의 바둑 애플리케이션에서 KataGo를 AI 엔진으로 사용하고 싶을 때.

**중점 읽기**:
1. [설치 및 설정](./setup.md) - 배포 요구사항 이해
2. [자주 사용하는 명령어](./commands.md) - 특히 Analysis Engine 부분

**핵심 지식**:
- GTP 모드 대신 Analysis Engine 모드 사용
- JSON API로 KataGo와 통신
- 하드웨어에 따라 탐색 파라미터 조정

### 시나리오 2: 대국 서버 구축

사용자가 AI와 대국할 수 있는 서버를 구축하고 싶을 때.

**중점 읽기**:
1. [설치 및 설정](./setup.md) - GPU 설정 부분
2. [자주 사용하는 명령어](./commands.md) - GTP 프로토콜 부분

**핵심 지식**:
- GTP 모드로 대국
- 다중 인스턴스 배포 전략
- 기력 조정 방법

### 시나리오 3: AI 알고리즘 연구

KataGo 구현을 심층 연구하고 수정이나 실험을 원할 때.

**중점 읽기**:
1. [소스코드 아키텍처](./architecture.md) - 전문 정독
2. 배경 지식 섹션의 모든 논문 해독

**핵심 지식**:
- C++ 코드 구조
- 신경망 아키텍처 세부사항
- MCTS 구현 방식

### 시나리오 4: 자체 모델 훈련

처음부터 KataGo 모델을 훈련하거나 파인튜닝하고 싶을 때.

**중점 읽기**:
1. [소스코드 아키텍처](./architecture.md) - 훈련 프로세스 부분
2. [KataGo 논문 해독](../background-info/katago-paper.md)

**핵심 지식**:
- 훈련 데이터 형식
- 훈련 스크립트 사용
- 하이퍼파라미터 설정

## 하드웨어 권장

KataGo는 다양한 하드웨어에서 실행 가능하지만 성능 차이가 큽니다:

| 하드웨어 구성 | 예상 성능 | 적용 시나리오 |
|---------|---------|---------|
| **고급 GPU**(RTX 4090)| ~2000 playouts/sec | 최상급 분석, 빠른 탐색 |
| **중급 GPU**(RTX 3060)| ~500 playouts/sec | 일반 분석, 대국 |
| **입문 GPU**(GTX 1650)| ~100 playouts/sec | 기본 사용 |
| **Apple Silicon**(M1/M2)| ~200-400 playouts/sec | macOS 개발 |
| **순수 CPU** | ~10-30 playouts/sec | 학습, 테스트 |

:::tip
느린 하드웨어에서도 KataGo는 가치 있는 분석을 제공합니다. 탐색량 감소가 정확도를 낮추지만 교육과 학습에는 보통 충분합니다.
:::

## 자주 묻는 질문

### KataGo와 Leela Zero의 차이점?

| 측면 | KataGo | Leela Zero |
|------|--------|------------|
| 기력 | 더 강함 | 더 약함 |
| 기능 | 풍부(집수, 영역) | 기본 |
| 다중 규칙 | 지원 | 미지원 |
| 개발 상태 | 활발 | 유지보수 모드 |
| 훈련 효율 | 높음 | 낮음 |

### GPU가 필요한가?

필수는 아니지만 강력 권장:
- **GPU 있음**: 빠른 분석 가능, 고품질 결과 획득
- **GPU 없음**: Eigen 백엔드 사용 가능, 하지만 느림

### 모델 파일 차이?

| 모델 크기 | 파일 크기 | 기력 | 속도 |
|---------|---------|------|------|
| b10c128 | ~20 MB | 중간 | 가장 빠름 |
| b18c384 | ~140 MB | 강함 | 빠름 |
| b40c256 | ~250 MB | 매우 강함 | 중간 |
| b60c320 | ~500 MB | 가장 강함 | 느림 |

보통 b18c384 또는 b40c256을 권장하며, 기력과 속도 사이 균형을 맞춥니다.

## 관련 자료

- [KataGo GitHub](https://github.com/lightvector/KataGo)
- [KataGo 훈련 웹사이트](https://katagotraining.org/)
- [KataGo Discord 커뮤니티](https://discord.gg/bqkZAz3)
- [Lizzie](https://github.com/featurecat/lizzie) - KataGo와 함께 쓰는 GUI

준비되셨나요? [설치 및 설정](./setup.md)부터 시작합시다!

