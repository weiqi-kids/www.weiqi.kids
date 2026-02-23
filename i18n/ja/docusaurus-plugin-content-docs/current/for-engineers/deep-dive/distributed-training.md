---
sidebar_position: 7
title: 分散訓練アーキテクチャ
description: KataGo分散訓練システムのアーキテクチャ、Self-play Worker、モデル公開フロー
---

# 分散訓練アーキテクチャ

本記事では、KataGoの分散訓練システムアーキテクチャを紹介し、グローバルコミュニティの計算能力を通じてモデルを継続的に改善する方法を説明します。

---

## システムアーキテクチャ概要

```
┌─────────────────────────────────────────────────────────────┐
│                    KataGo 分散訓練システム                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│   │  Worker 1   │    │  Worker 2   │    │  Worker N   │   │
│   │  (Self-play)│    │  (Self-play)│    │  (Self-play)│   │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘   │
│          │                  │                  │           │
│          └────────────┬─────┴──────────────────┘           │
│                       │                                     │
│                       ▼                                     │
│            ┌─────────────────────┐                         │
│            │    訓練サーバー       │                         │
│            │  (Data Collection)  │                         │
│            └──────────┬──────────┘                         │
│                       │                                     │
│                       ▼                                     │
│            ┌─────────────────────┐                         │
│            │    訓練フロー         │                         │
│            │  (Model Training)   │                         │
│            └──────────┬──────────┘                         │
│                       │                                     │
│                       ▼                                     │
│            ┌─────────────────────┐                         │
│            │    新モデル公開       │                         │
│            │  (Model Release)    │                         │
│            └─────────────────────┘                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Self-play Worker

### ワークフロー

各Workerは以下のループを実行します：

```python
def self_play_worker():
    while True:
        # 1. 最新モデルをダウンロード
        model = download_latest_model()

        # 2. 自己対局を実行
        games = []
        for _ in range(batch_size):
            game = play_game(model)
            games.append(game)

        # 3. 対局データをアップロード
        upload_games(games)

        # 4. 新モデルをチェック
        if new_model_available():
            model = download_latest_model()
```

### 対局生成

```python
def play_game(model):
    """1局の自己対局を実行"""
    game = Game()
    positions = []

    while not game.is_terminal():
        # MCTS探索
        mcts = MCTS(model, num_simulations=800)
        policy = mcts.get_policy(game.state)

        # ディリクレノイズを追加（探索を増やす）
        if game.move_count < 30:
            policy = add_dirichlet_noise(policy)

        # policyに基づいて手を選択
        if game.move_count < 30:
            # 最初の30手は温度サンプリング
            action = sample_with_temperature(policy, temp=1.0)
        else:
            # その後は貪欲選択
            action = np.argmax(policy)

        # 訓練データを記録
        positions.append({
            'state': game.state.copy(),
            'policy': policy,
            'player': game.current_player
        })

        game.play(action)

    # 勝敗をマーク
    winner = game.get_winner()
    for pos in positions:
        pos['value'] = 1.0 if pos['player'] == winner else -1.0

    return positions
```

### データフォーマット

```json
{
  "version": 1,
  "rules": "chinese",
  "komi": 7.5,
  "board_size": 19,
  "positions": [
    {
      "move_number": 0,
      "board": "...",
      "policy": [0.01, 0.02, ...],
      "value": 1.0,
      "score": 2.5
    }
  ]
}
```

---

## データ収集サーバー

### 機能

1. **対局データの受信**：Workerから対局を収集
2. **データ検証**：フォーマットをチェック、異常をフィルタリング
3. **データ保存**：訓練データセットに書き込み
4. **統計監視**：対局数、Workerステータスを追跡

### データ検証

```python
def validate_game(game_data):
    """対局データを検証"""
    checks = [
        len(game_data['positions']) > 10,  # 最小手数
        len(game_data['positions']) < 500,  # 最大手数
        all(is_valid_policy(p['policy']) for p in game_data['positions']),
        game_data['rules'] in SUPPORTED_RULES,
    ]
    return all(checks)
```

### データ保存構造

```
training_data/
├── run_001/
│   ├── games_00001.npz
│   ├── games_00002.npz
│   └── ...
├── run_002/
│   └── ...
└── current/
    └── latest_games.npz
```

---

## 訓練フロー

### 訓練ループ

```python
def training_loop():
    model = load_model()
    optimizer = Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        # 最新の対局データをロード
        dataset = load_recent_games(num_games=100000)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        for batch in dataloader:
            states = batch['states']
            target_policies = batch['policies']
            target_values = batch['values']

            # 順伝播
            pred_policies, pred_values = model(states)

            # 損失を計算
            policy_loss = cross_entropy(pred_policies, target_policies)
            value_loss = mse_loss(pred_values, target_values)
            loss = policy_loss + value_loss

            # 逆伝播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 定期的に評価
        if epoch % 100 == 0:
            evaluate_model(model)
```

### 損失関数

KataGoは複数の損失項を使用します：

```python
def compute_loss(predictions, targets):
    # Policy損失（交差エントロピー）
    policy_loss = F.cross_entropy(
        predictions['policy'],
        targets['policy']
    )

    # Value損失（MSE）
    value_loss = F.mse_loss(
        predictions['value'],
        targets['value']
    )

    # Score損失（MSE）
    score_loss = F.mse_loss(
        predictions['score'],
        targets['score']
    )

    # Ownership損失（MSE）
    ownership_loss = F.mse_loss(
        predictions['ownership'],
        targets['ownership']
    )

    # 重み付き合計
    total_loss = (
        1.0 * policy_loss +
        1.0 * value_loss +
        0.5 * score_loss +
        0.5 * ownership_loss
    )

    return total_loss
```

---

## モデル評価と公開

### Elo評価

新モデルは旧モデルと対戦して棋力を評価する必要があります：

```python
def evaluate_new_model(new_model, baseline_model, num_games=400):
    """新モデルのEloを評価"""
    wins = 0
    losses = 0
    draws = 0

    for _ in range(num_games // 2):
        # 新モデルが黒番
        result = play_game(new_model, baseline_model)
        if result == 'black_wins':
            wins += 1
        elif result == 'white_wins':
            losses += 1
        else:
            draws += 1

        # 新モデルが白番
        result = play_game(baseline_model, new_model)
        if result == 'white_wins':
            wins += 1
        elif result == 'black_wins':
            losses += 1
        else:
            draws += 1

    # Elo差を計算
    win_rate = (wins + 0.5 * draws) / num_games
    elo_diff = 400 * math.log10(win_rate / (1 - win_rate))

    return elo_diff
```

### 公開条件

```python
def should_release_model(new_model, current_best):
    """新モデルを公開すべきかどうかを決定"""
    elo_diff = evaluate_new_model(new_model, current_best)

    # 条件：Elo向上が閾値を超える
    if elo_diff > 20:
        return True

    # または：一定の訓練ステップに達した
    if training_steps % 10000 == 0:
        return True

    return False
```

### モデルバージョン命名

```
kata1-b18c384nbt-s{steps}-d{data}.bin.gz

例：
kata1-b18c384nbt-s9996604416-d4316597426.bin.gz
├── kata1: 訓練シリーズ
├── b18c384nbt: アーキテクチャ（18残差ブロック、384チャネル）
├── s9996604416: 訓練ステップ数
└── d4316597426: 訓練データ量
```

---

## KataGo Training参加ガイド

### システム要件

| 項目 | 最低要件 | 推奨要件 |
|------|---------|---------|
| GPU | GTX 1060 | RTX 3060以上 |
| VRAM | 4 GB | 8 GB以上 |
| ネットワーク | 10 Mbps | 50 Mbps以上 |
| 稼働時間 | 継続稼働 | 24/7 |

### Workerのインストール

```bash
# Workerをダウンロード
wget https://katagotraining.org/download/worker

# 設定
./katago contribute -config contribute.cfg

# 貢献を開始
./katago contribute
```

### 設定ファイル

```ini
# contribute.cfg

# サーバー設定
serverUrl = https://katagotraining.org/

# ユーザー名（統計用）
username = your_username

# GPU設定
numNNServerThreadsPerModel = 1
nnMaxBatchSize = 16

# 対局設定
gamesPerBatch = 25
```

### 貢献の監視

```bash
# 統計を確認
https://katagotraining.org/contributions/

# ローカルログ
tail -f katago_contribute.log
```

---

## 訓練統計

### KataGo訓練マイルストーン

| 時期 | 対局数 | Elo |
|------|--------|-----|
| 2019.06 | 10M | 初期 |
| 2020.01 | 100M | +500 |
| 2021.01 | 500M | +800 |
| 2022.01 | 1B | +1000 |
| 2024.01 | 5B以上 | +1200 |

### コミュニティ貢献者

- 数百人のグローバル貢献者
- 累計数千GPU年の計算能力
- 24/7継続稼働

---

## 高度なトピック

### カリキュラム学習（Curriculum Learning）

訓練難易度を段階的に上げる：

```python
def get_training_config(training_step):
    if training_step < 100000:
        return {'board_size': 9, 'visits': 200}
    elif training_step < 500000:
        return {'board_size': 13, 'visits': 400}
    else:
        return {'board_size': 19, 'visits': 800}
```

### データ拡張

盤面の対称性を利用してデータ量を増やす：

```python
def augment_position(state, policy):
    """8種類の対称変換"""
    augmented = []

    for rotation in [0, 90, 180, 270]:
        for flip in [False, True]:
            aug_state = transform(state, rotation, flip)
            aug_policy = transform_policy(policy, rotation, flip)
            augmented.append((aug_state, aug_policy))

    return augmented
```

---

## 関連記事

- [KataGo訓練メカニズム解析](../training) — 訓練フローの詳細
- [オープンソースコミュニティへの参加](../contributing) — コードの貢献方法
- [評価とベンチマーク](../evaluation) — モデル評価方法
