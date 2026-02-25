---
sidebar_position: 4
title: オープンソースコミュニティへの参加
description: KataGoオープンソースコミュニティに参加し、計算能力やコードを貢献する
---

# オープンソースコミュニティへの参加

KataGoは活発なオープンソースプロジェクトで、貢献する方法は複数あります。

---

## 貢献方法の概要

| 方法 | 難易度 | 要件 |
|------|------|------|
| **計算能力を貢献** | 低 | GPU搭載のコンピュータ |
| **問題を報告** | 低 | GitHubアカウント |
| **ドキュメントを改善** | 中 | 技術内容に精通 |
| **コードを貢献** | 高 | C++/Python開発能力 |

---

## 計算能力の貢献：分散訓練

### KataGo Trainingの紹介

KataGo Trainingはグローバル分散訓練ネットワークです：

- ボランティアがGPU計算能力を提供して自己対局を実行
- 自己対局データを中央サーバーにアップロード
- サーバーが定期的に新モデルを訓練
- 新モデルをボランティアに配布して対局を継続

公式サイト：https://katagotraining.org/

### 参加手順

#### 1. アカウントを作成

https://katagotraining.org/ でアカウントを登録。

#### 2. KataGoをダウンロード

```bash
# 最新版をダウンロード
wget https://github.com/lightvector/KataGo/releases/download/v1.15.3/katago-v1.15.3-cuda11.1-linux-x64.zip
unzip katago-v1.15.3-cuda11.1-linux-x64.zip
```

#### 3. contributeモードを設定

```bash
# 初回実行時に設定をガイド
./katago contribute -username YOUR_USERNAME -password YOUR_PASSWORD
```

システムは自動的に：
- 最新モデルをダウンロード
- 自己対局を実行
- 対局データをアップロード

#### 4. バックグラウンドで実行

```bash
# screenまたはtmuxでバックグラウンド実行
screen -S katago
./katago contribute -username YOUR_USERNAME -password YOUR_PASSWORD
# Ctrl+A, Dでscreenから離脱
```

### 貢献統計

https://katagotraining.org/contributions/ で以下を確認できます：
- あなたの貢献ランキング
- 総貢献対局数
- 最近訓練されたモデル

---

## 問題の報告

### どこで報告するか

- **GitHub Issues**：https://github.com/lightvector/KataGo/issues
- **Discord**：https://discord.gg/bqkZAz3

### 良い問題報告に含まれるもの

1. **KataGoバージョン**：`katago version`
2. **オペレーティングシステム**：Windows/Linux/macOS
3. **ハードウェア**：GPUモデル、メモリ
4. **完全なエラーメッセージ**：完全なログをコピー
5. **再現手順**：どうすればこの問題が発生するか

### 例

```markdown
## 問題の説明
benchmark実行時にメモリ不足エラーが発生

## 環境
- KataGoバージョン：1.15.3
- オペレーティングシステム：Ubuntu 22.04
- GPU：RTX 3060 12GB
- モデル：kata-b40c256.bin.gz

## エラーメッセージ
```
CUDA error: out of memory
```

## 再現手順
1. `katago benchmark -model kata-b40c256.bin.gz`を実行
2. 約30秒待つ
3. エラーが発生
```

---

## ドキュメントの改善

### ドキュメントの場所

- **README**：`README.md`
- **GTPドキュメント**：`docs/GTP_Extensions.md`
- **Analysisドキュメント**：`docs/Analysis_Engine.md`
- **訓練ドキュメント**：`python/README.md`

### 貢献フロー

1. プロジェクトをFork
2. 新しいブランチを作成
3. ドキュメントを修正
4. Pull Requestを提出

```bash
git clone https://github.com/YOUR_USERNAME/KataGo.git
cd KataGo
git checkout -b improve-docs
# ドキュメントを編集
git add .
git commit -m "Improve documentation for Analysis Engine"
git push origin improve-docs
# GitHubでPull Requestを作成
```

---

## コードの貢献

### 開発環境のセットアップ

```bash
# プロジェクトをクローン
git clone https://github.com/lightvector/KataGo.git
cd KataGo

# コンパイル（Debugモード）
cd cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=OPENCL -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# テストを実行
./katago runtests
```

### コーディングスタイル

KataGoは以下のコーディングスタイルを使用しています：

**C++**：
- 2スペースインデント
- 中括弧は同じ行
- 変数名はcamelCase
- クラス名はPascalCase

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

**Python**：
- PEP 8に準拠
- 4スペースインデント

### 貢献領域

| 領域 | ファイル位置 | 必要スキル |
|------|---------|---------|
| コアエンジン | `cpp/` | C++, CUDA/OpenCL |
| 訓練プログラム | `python/` | Python, PyTorch |
| GTPプロトコル | `cpp/command/gtp.cpp` | C++ |
| Analysis API | `cpp/command/analysis.cpp` | C++, JSON |
| テスト | `cpp/tests/` | C++ |

### Pull Requestフロー

1. **Issueを作成**：まずやりたい変更について議論
2. **Fork & Clone**：自分のブランチを作成
3. **開発とテスト**：すべてのテストが通ることを確認
4. **PRを提出**：変更内容を詳細に説明
5. **Code Review**：メンテナーのフィードバックに対応
6. **マージ**：メンテナーがあなたのコードをマージ

### PRの例

```markdown
## 変更の説明
New Zealandルールのサポートを追加

## 変更内容
- rules.cppにNEW_ZEALANDルールを追加
- GTPコマンドで`kata-set-rules nz`をサポート
- ユニットテストを追加

## テスト結果
- 既存のテストすべて合格
- 新しいテスト合格

## 関連Issue
Fixes #123
```

---

## コミュニティリソース

### 公式リンク

| リソース | リンク |
|------|------|
| GitHub | https://github.com/lightvector/KataGo |
| Discord | https://discord.gg/bqkZAz3 |
| 訓練ネットワーク | https://katagotraining.org/ |

### ディスカッション

- **Discord**：リアルタイム議論、技術Q&A
- **GitHub Discussions**：長い議論、機能提案
- **Reddit r/baduk**：一般的な囲碁AI議論

### 関連プロジェクト

| プロジェクト | 説明 | リンク |
|------|------|------|
| KaTrain | 教育分析ツール | github.com/sanderland/katrain |
| Lizzie | 分析インターフェース | github.com/featurecat/lizzie |
| Sabaki | 棋譜エディタ | sabaki.yichuanshen.de |
| BadukAI | オンライン分析 | baduk.ai |

---

## 認知と報酬

### 貢献者リスト

すべての貢献者は以下に記載されます：
- GitHub Contributorsページ
- KataGo Training貢献ランキング

### 学習の成果

オープンソースプロジェクトに参加することで得られるもの：
- 工業レベルのAIシステムアーキテクチャを学ぶ
- 世界中の開発者と交流
- オープンソース貢献記録を蓄積
- 囲碁AI技術の深い理解

---

## 関連記事

- [ソースコード解説](../source-code) — コード構造を理解
- [KataGo訓練メカニズム解析](../training) — ローカル訓練実験
- [囲碁AIを一記事で理解](../../how-it-works/) — 技術原理
