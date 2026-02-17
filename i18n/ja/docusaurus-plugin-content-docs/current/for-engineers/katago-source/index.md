---
sidebar_position: 2
title: KataGo実践入門
---

# KataGo実践入門ガイド

本章ではインストールから実際の使用まで、KataGoに関するすべての実用的な知識を網羅します。KataGoを自分のアプリケーションに統合したい方も、ソースコードを深く研究したい方も、ここが出発点となります。

## なぜKataGoを選ぶか？

多くの囲碁AIの中で、KataGoは現在最良の選択肢です。理由は以下の通りです：

| 優位性 | 説明 |
|------|------|
| **最強の棋力** | 公開テストで常に最高水準を維持 |
| **最も豊富な機能** | 目数予測、領地分析、複数ルールサポート |
| **完全オープンソース** | MITライセンス、自由に使用・修正可能 |
| **継続的な更新** | 活発な開発とコミュニティサポート |
| **完備したドキュメント** | 公式ドキュメントが詳細で、コミュニティリソースも豊富 |
| **マルチプラットフォームサポート** | Linux、macOS、Windowsで実行可能 |

## 本章の内容

### [インストールと設定](./setup.md)

ゼロからKataGo環境を構築：

- システム要件とハードウェア推奨
- 各プラットフォームのインストール手順（macOS / Linux / Windows）
- モデルのダウンロードと選択ガイド
- 設定ファイルの詳細説明

### [よく使うコマンド](./commands.md)

KataGoの使用方法をマスター：

- GTP（Go Text Protocol）プロトコル紹介
- よく使うGTPコマンドと例
- Analysis Engineの使用方法
- JSON API完全ガイド

### [ソースコードアーキテクチャ](./architecture.md)

KataGoの実装詳細を深く理解：

- プロジェクトディレクトリ構造の概要
- ニューラルネットワークアーキテクチャ解析
- 探索エンジンの実装詳細
- 訓練フローの概要

## クイックスタート

KataGoを素早く試したいだけなら、以下が最も簡単な方法です：

### macOS（Homebrewを使用）

```bash
# インストール
brew install katago

# モデルをダウンロード（テスト用に小さいモデルを選択）
curl -L -o kata-b18c384.bin.gz \
  https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz

# GTPモードで実行
katago gtp -model kata-b18c384.bin.gz -config gtp_example.cfg
```

### Linux（プリコンパイル版）

```bash
# プリコンパイル版をダウンロード
wget https://github.com/lightvector/KataGo/releases/download/v1.15.3/katago-v1.15.3-opencl-linux-x64.zip

# 解凍
unzip katago-v1.15.3-opencl-linux-x64.zip

# モデルをダウンロード
wget https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz

# 実行
./katago gtp -model kata-b18c384nbt-*.bin.gz -config default_gtp.cfg
```

### インストール確認

起動に成功すると、GTPプロンプトが表示されます。以下のコマンドを試してください：

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

## 使用シナリオガイド

ニーズに応じて、以下の読む順序と重点をお勧めします：

### シナリオ1：囲碁アプリへの統合

自分の囲碁アプリケーションでKataGoをAIエンジンとして使用したい場合。

**重点的に読む内容**：
1. [インストールと設定](./setup.md) - デプロイ要件を理解
2. [よく使うコマンド](./commands.md) - 特にAnalysis Engine部分

**キーポイント**：
- GTPモードではなくAnalysis Engineモードを使用
- JSON APIでKataGoと通信
- ハードウェアに応じて探索パラメータを調整

### シナリオ2：対局サーバーの構築

ユーザーがAIと対局できるサーバーを構築したい場合。

**重点的に読む内容**：
1. [インストールと設定](./setup.md) - GPU設定部分
2. [よく使うコマンド](./commands.md) - GTPプロトコル部分

**キーポイント**：
- GTPモードで対局
- 複数インスタンスのデプロイ戦略
- 棋力調整方法

### シナリオ3：AIアルゴリズムの研究

KataGoの実装を深く研究し、修正や実験をしたい場合。

**重点的に読む内容**：
1. [ソースコードアーキテクチャ](./architecture.md) - 全文精読
2. 背景知識章のすべての論文解説

**キーポイント**：
- C++コード構造
- ニューラルネットワークアーキテクチャの詳細
- MCTS実装方式

### シナリオ4：自分のモデルを訓練

ゼロからKataGoモデルを訓練またはファインチューニングしたい場合。

**重点的に読む内容**：
1. [ソースコードアーキテクチャ](./architecture.md) - 訓練フロー部分
2. [KataGo論文解説](../background-info/katago-paper.md)

**キーポイント**：
- 訓練データフォーマット
- 訓練スクリプトの使用
- ハイパーパラメータ設定

## ハードウェア推奨

KataGoは様々なハードウェアで実行できますが、性能差は大きいです：

| ハードウェア構成 | 予想性能 | 適用シーン |
|---------|---------|---------|
| **ハイエンドGPU**（RTX 4090）| ~2000 playouts/sec | トップレベル分析、高速探索 |
| **ミドルレンジGPU**（RTX 3060）| ~500 playouts/sec | 一般分析、対局 |
| **エントリーGPU**（GTX 1650）| ~100 playouts/sec | 基本使用 |
| **Apple Silicon**（M1/M2）| ~200-400 playouts/sec | macOS開発 |
| **純CPU** | ~10-30 playouts/sec | 学習、テスト |

:::tip
遅いハードウェアでも、KataGoは価値ある分析を提供できます。探索量の減少は精度を下げますが、教育や学習には通常十分です。
:::

## よくある質問

### KataGoとLeela Zeroの違いは？

| 側面 | KataGo | Leela Zero |
|------|--------|------------|
| 棋力 | より強い | 比較的弱い |
| 機能 | 豊富（目数、領地） | 基本のみ |
| 複数ルール | サポート | 非サポート |
| 開発状態 | 活発 | メンテナンスモード |
| 訓練効率 | 高い | 比較的低い |

### GPUは必要？

必須ではありませんが、強く推奨します：
- **GPUあり**：高速分析が可能、高品質な結果を得られる
- **GPUなし**：Eigenバックエンドを使用可能だが、速度は遅い

### モデルファイルの違いは？

| モデルサイズ | ファイルサイズ | 棋力 | 速度 |
|---------|---------|------|------|
| b10c128 | ~20 MB | 中程度 | 最速 |
| b18c384 | ~140 MB | 強い | 速い |
| b40c256 | ~250 MB | とても強い | 中 |
| b60c320 | ~500 MB | 最強 | 遅い |

通常はb18c384またはb40c256を推奨します。棋力と速度のバランスが取れています。

## 関連リソース

- [KataGo GitHub](https://github.com/lightvector/KataGo)
- [KataGo訓練サイト](https://katagotraining.org/)
- [KataGo Discordコミュニティ](https://discord.gg/bqkZAz3)
- [Lizzie](https://github.com/featurecat/lizzie) - KataGoと組み合わせて使用するGUI

準備はできましたか？[インストールと設定](./setup.md)から始めましょう！

