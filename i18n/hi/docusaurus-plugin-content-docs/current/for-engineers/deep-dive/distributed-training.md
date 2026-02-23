---
sidebar_position: 7
title: वितरित प्रशिक्षण आर्किटेक्चर
description: KataGo वितरित प्रशिक्षण प्रणाली का आर्किटेक्चर, Self-play Worker और मॉडल रिलीज़ प्रक्रिया
---

# वितरित प्रशिक्षण आर्किटेक्चर

यह लेख KataGo की वितरित प्रशिक्षण प्रणाली के आर्किटेक्चर का परिचय देता है, बताता है कि वैश्विक समुदाय की कम्प्यूटिंग पावर के माध्यम से मॉडल को निरंतर कैसे सुधारा जाता है।

---

## सिस्टम आर्किटेक्चर अवलोकन

```
┌─────────────────────────────────────────────────────────────┐
│                    KataGo वितरित प्रशिक्षण प्रणाली                     │
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
│            │    प्रशिक्षण सर्वर       │                         │
│            │  (Data Collection)  │                         │
│            └──────────┬──────────┘                         │
│                       │                                     │
│                       ▼                                     │
│            ┌─────────────────────┐                         │
│            │    प्रशिक्षण प्रक्रिया         │                         │
│            │  (Model Training)   │                         │
│            └──────────┬──────────┘                         │
│                       │                                     │
│                       ▼                                     │
│            ┌─────────────────────┐                         │
│            │    नया मॉडल रिलीज़       │                         │
│            │  (Model Release)    │                         │
│            └─────────────────────┘                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Self-play Worker

### कार्य प्रवाह

प्रत्येक Worker निम्न लूप निष्पादित करता है:

```python
def self_play_worker():
    while True:
        # 1. नवीनतम मॉडल डाउनलोड करें
        model = download_latest_model()

        # 2. सेल्फ-प्ले निष्पादित करें
        games = []
        for _ in range(batch_size):
            game = play_game(model)
            games.append(game)

        # 3. गेम डेटा अपलोड करें
        upload_games(games)

        # 4. नया मॉडल जांचें
        if new_model_available():
            model = download_latest_model()
```

### गेम जनरेशन

```python
def play_game(model):
    """एक सेल्फ-प्ले गेम खेलें"""
    game = Game()
    positions = []

    while not game.is_terminal():
        # MCTS सर्च
        mcts = MCTS(model, num_simulations=800)
        policy = mcts.get_policy(game.state)

        # Dirichlet नॉइज़ जोड़ें (अन्वेषण बढ़ाएं)
        if game.move_count < 30:
            policy = add_dirichlet_noise(policy)

        # policy के अनुसार चाल चुनें
        if game.move_count < 30:
            # पहली 30 चालों में तापमान सैंपलिंग
            action = sample_with_temperature(policy, temp=1.0)
        else:
            # बाद में लालची चयन
            action = np.argmax(policy)

        # प्रशिक्षण डेटा रिकॉर्ड करें
        positions.append({
            'state': game.state.copy(),
            'policy': policy,
            'player': game.current_player
        })

        game.play(action)

    # जीत/हार चिह्नित करें
    winner = game.get_winner()
    for pos in positions:
        pos['value'] = 1.0 if pos['player'] == winner else -1.0

    return positions
```

### डेटा प्रारूप

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

## डेटा संग्रह सर्वर

### कार्य

1. **गेम डेटा प्राप्त करें**: Workers से गेम संग्रह
2. **डेटा सत्यापन**: प्रारूप जांच, असामान्य फ़िल्टर
3. **डेटा स्टोरेज**: प्रशिक्षण डेटासेट में लिखें
4. **सांख्यिकी मॉनिटरिंग**: गेम संख्या, Worker स्थिति ट्रैक करें

### डेटा सत्यापन

```python
def validate_game(game_data):
    """गेम डेटा सत्यापित करें"""
    checks = [
        len(game_data['positions']) > 10,  # न्यूनतम चालें
        len(game_data['positions']) < 500,  # अधिकतम चालें
        all(is_valid_policy(p['policy']) for p in game_data['positions']),
        game_data['rules'] in SUPPORTED_RULES,
    ]
    return all(checks)
```

### डेटा स्टोरेज संरचना

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

## प्रशिक्षण प्रक्रिया

### प्रशिक्षण लूप

```python
def training_loop():
    model = load_model()
    optimizer = Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        # नवीनतम गेम डेटा लोड करें
        dataset = load_recent_games(num_games=100000)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

        for batch in dataloader:
            states = batch['states']
            target_policies = batch['policies']
            target_values = batch['values']

            # फॉरवर्ड प्रोपेगेशन
            pred_policies, pred_values = model(states)

            # लॉस गणना
            policy_loss = cross_entropy(pred_policies, target_policies)
            value_loss = mse_loss(pred_values, target_values)
            loss = policy_loss + value_loss

            # बैकप्रोपेगेशन
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # नियमित मूल्यांकन
        if epoch % 100 == 0:
            evaluate_model(model)
```

### लॉस फंक्शन

KataGo कई लॉस टर्म्स का उपयोग करता है:

```python
def compute_loss(predictions, targets):
    # Policy लॉस (क्रॉस-एंट्रॉपी)
    policy_loss = F.cross_entropy(
        predictions['policy'],
        targets['policy']
    )

    # Value लॉस (MSE)
    value_loss = F.mse_loss(
        predictions['value'],
        targets['value']
    )

    # Score लॉस (MSE)
    score_loss = F.mse_loss(
        predictions['score'],
        targets['score']
    )

    # Ownership लॉस (MSE)
    ownership_loss = F.mse_loss(
        predictions['ownership'],
        targets['ownership']
    )

    # भारित योग
    total_loss = (
        1.0 * policy_loss +
        1.0 * value_loss +
        0.5 * score_loss +
        0.5 * ownership_loss
    )

    return total_loss
```

---

## मॉडल मूल्यांकन और रिलीज़

### Elo मूल्यांकन

नए मॉडल को पुराने मॉडल के खिलाफ खेलकर खेल शक्ति का मूल्यांकन करना होगा:

```python
def evaluate_new_model(new_model, baseline_model, num_games=400):
    """नए मॉडल का Elo मूल्यांकन"""
    wins = 0
    losses = 0
    draws = 0

    for _ in range(num_games // 2):
        # नया मॉडल काला खेलता है
        result = play_game(new_model, baseline_model)
        if result == 'black_wins':
            wins += 1
        elif result == 'white_wins':
            losses += 1
        else:
            draws += 1

        # नया मॉडल सफेद खेलता है
        result = play_game(baseline_model, new_model)
        if result == 'white_wins':
            wins += 1
        elif result == 'black_wins':
            losses += 1
        else:
            draws += 1

    # Elo अंतर गणना
    win_rate = (wins + 0.5 * draws) / num_games
    elo_diff = 400 * math.log10(win_rate / (1 - win_rate))

    return elo_diff
```

### रिलीज़ शर्तें

```python
def should_release_model(new_model, current_best):
    """नया मॉडल रिलीज़ करना है या नहीं"""
    elo_diff = evaluate_new_model(new_model, current_best)

    # शर्त: Elo वृद्धि थ्रेशोल्ड से अधिक
    if elo_diff > 20:
        return True

    # या: निश्चित प्रशिक्षण चरणों तक पहुंचा
    if training_steps % 10000 == 0:
        return True

    return False
```

### मॉडल संस्करण नामकरण

```
kata1-b18c384nbt-s{steps}-d{data}.bin.gz

उदाहरण:
kata1-b18c384nbt-s9996604416-d4316597426.bin.gz
├── kata1: प्रशिक्षण श्रृंखला
├── b18c384nbt: आर्किटेक्चर (18 रेसिड्यूअल ब्लॉक, 384 चैनल)
├── s9996604416: प्रशिक्षण चरण
└── d4316597426: प्रशिक्षण डेटा मात्रा
```

---

## KataGo Training भागीदारी गाइड

### सिस्टम आवश्यकताएँ

| आइटम | न्यूनतम आवश्यकता | सुझाई आवश्यकता |
|------|---------|---------|
| GPU | GTX 1060 | RTX 3060+ |
| VRAM | 4 GB | 8 GB+ |
| नेटवर्क | 10 Mbps | 50 Mbps+ |
| रनटाइम | निरंतर चलना | 24/7 |

### Worker इंस्टॉल करें

```bash
# Worker डाउनलोड करें
wget https://katagotraining.org/download/worker

# कॉन्फ़िगरेशन
./katago contribute -config contribute.cfg

# योगदान शुरू करें
./katago contribute
```

### कॉन्फ़िगरेशन फ़ाइल

```ini
# contribute.cfg

# सर्वर सेटिंग
serverUrl = https://katagotraining.org/

# यूज़रनेम (सांख्यिकी के लिए)
username = your_username

# GPU सेटिंग
numNNServerThreadsPerModel = 1
nnMaxBatchSize = 16

# गेम सेटिंग
gamesPerBatch = 25
```

### योगदान मॉनिटर करें

```bash
# सांख्यिकी देखें
https://katagotraining.org/contributions/

# लोकल लॉग
tail -f katago_contribute.log
```

---

## प्रशिक्षण सांख्यिकी

### KataGo प्रशिक्षण माइलस्टोन

| समय | गेम संख्या | Elo |
|------|--------|-----|
| 2019.06 | 10M | प्रारंभिक |
| 2020.01 | 100M | +500 |
| 2021.01 | 500M | +800 |
| 2022.01 | 1B | +1000 |
| 2024.01 | 5B+ | +1200 |

### समुदाय योगदानकर्ता

- सैकड़ों वैश्विक योगदानकर्ता
- कुल हजारों GPU वर्षों की कम्प्यूटिंग पावर
- 24/7 निरंतर चलना

---

## उन्नत विषय

### करिकुलम लर्निंग

धीरे-धीरे प्रशिक्षण कठिनाई बढ़ाएं:

```python
def get_training_config(training_step):
    if training_step < 100000:
        return {'board_size': 9, 'visits': 200}
    elif training_step < 500000:
        return {'board_size': 13, 'visits': 400}
    else:
        return {'board_size': 19, 'visits': 800}
```

### डेटा ऑगमेंटेशन

बोर्ड सममिति से डेटा मात्रा बढ़ाएं:

```python
def augment_position(state, policy):
    """8 सममिति परिवर्तन"""
    augmented = []

    for rotation in [0, 90, 180, 270]:
        for flip in [False, True]:
            aug_state = transform(state, rotation, flip)
            aug_policy = transform_policy(policy, rotation, flip)
            augmented.append((aug_state, aug_policy))

    return augmented
```

---

## आगे पढ़ें

- [KataGo प्रशिक्षण तंत्र विश्लेषण](../training) — प्रशिक्षण प्रक्रिया विस्तृत व्याख्या
- [ओपन सोर्स समुदाय में भागीदारी](../contributing) — कोड योगदान कैसे करें
- [मूल्यांकन और बेंचमार्क टेस्टिंग](../evaluation) — मॉडल मूल्यांकन विधियाँ
