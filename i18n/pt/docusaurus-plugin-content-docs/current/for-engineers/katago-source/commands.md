---
sidebar_position: 2
title: Comandos Comuns
---

# Comandos Comuns do KataGo

Este artigo apresenta os dois modos principais de operacao do KataGo: protocolo GTP e Analysis Engine, alem de explicacao detalhada de comandos comuns.

## Introducao ao Protocolo GTP

GTP (Go Text Protocol) e o protocolo padrao de comunicacao entre programas de Go. A maioria das GUIs de Go (como Sabaki, Lizzie) usa GTP para se comunicar com motores de IA.

### Iniciar Modo GTP

```bash
katago gtp -model /path/to/model.bin.gz -config /path/to/config.cfg
```

### Formato Basico do Protocolo GTP

```
[id] command_name [arguments]
```

- `id`: Numero de comando opcional, usado para rastrear respostas
- `command_name`: Nome do comando
- `arguments`: Argumentos do comando

Formato de resposta:
```
=[id] response_data     # Sucesso
?[id] error_message     # Falha
```

### Exemplo Basico

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

## Comandos GTP Comuns

### Informacao do Programa

| Comando | Descricao | Exemplo |
|------|------|------|
| `name` | Obter nome do programa | `name` → `= KataGo` |
| `version` | Obter numero de versao | `version` → `= 1.15.3` |
| `list_commands` | Listar todos os comandos suportados | `list_commands` |
| `protocol_version` | Versao do protocolo GTP | `protocol_version` → `= 2` |

### Configuracao do Tabuleiro

```
# Definir tamanho do tabuleiro (9, 13, 19)
boardsize 19

# Definir komi
komi 7.5

# Limpar tabuleiro
clear_board

# Definir regras (extensao KataGo)
kata-set-rules chinese    # Regras chinesas
kata-set-rules japanese   # Regras japonesas
kata-set-rules tromp-taylor
```

### Relacionados a Jogar

```
# Fazer jogada
play black Q16    # Preto joga em Q16
play white D4     # Branco joga em D4
play black pass   # Preto passa

# Deixar IA fazer uma jogada
genmove black     # Gerar jogada para preto
genmove white     # Gerar jogada para branco

# Desfazer
undo              # Desfazer uma jogada

# Definir limite de jogadas
kata-set-param maxVisits 1000    # Definir numero maximo de buscas
```

### Consulta de Posicao

```
# Mostrar tabuleiro
showboard

# Obter jogador atual
kata-get-player

# Obter resultado de analise
kata-analyze black 100    # Analisar preto, buscar 100 vezes
```

### Relacionados a Regras

```
# Obter regras atuais
kata-get-rules

# Definir regras
kata-set-rules chinese

# Definir handicap
fixed_handicap 4     # Posicoes padrao de 4 pedras de handicap
place_free_handicap 4  # Handicap livre
```

## Comandos de Extensao KataGo

KataGo fornece muitos comandos de extensao alem do GTP padrao:

### kata-analyze

Analise em tempo real da posicao atual:

```
kata-analyze [player] [visits] [interval]
```

Parametros:
- `player`: Analisar qual lado (black/white)
- `visits`: Numero de buscas
- `interval`: Intervalo de relatorio (centissegundos, 1/100 segundo)

Exemplo:
```
kata-analyze black 1000 100
```

Saida:
```
info move Q3 visits 523 winrate 0.5432 scoreMean 2.31 scoreSelfplay 2.45 prior 0.1234 order 0 pv Q3 R4 Q5 ...
info move D4 visits 312 winrate 0.5123 scoreMean 1.82 scoreSelfplay 1.95 prior 0.0987 order 1 pv D4 C6 E3 ...
...
```

Descricao dos campos de saida:

| Campo | Descricao |
|------|------|
| `move` | Posicao da jogada |
| `visits` | Numero de visitas de busca |
| `winrate` | Taxa de vitoria (0-1) |
| `scoreMean` | Diferenca de pontos esperada |
| `scoreSelfplay` | Pontos esperados em self-play |
| `prior` | Probabilidade prior da rede neural |
| `order` | Ordem de ranking |
| `pv` | Principal Variation (variacao principal) |

### kata-raw-nn

Obter saida bruta da rede neural:

```
kata-raw-nn [symmetry]
```

Saida inclui:
- Distribuicao de probabilidade Policy
- Predicao Value
- Predicao de territorio, etc.

### kata-debug-print

Mostrar informacao detalhada de busca, para depuracao:

```
kata-debug-print move Q16
```

### Ajuste de Forca

```
# Definir numero maximo de visitas
kata-set-param maxVisits 100      # Mais fraco
kata-set-param maxVisits 10000    # Mais forte

# Definir tempo de pensamento
kata-time-settings main 60 0      # 60 segundos por lado
kata-time-settings byoyomi 30 5   # Byoyomi 30 segundos 5 periodos
```

## Uso do Analysis Engine

Analysis Engine e outro modo de operacao fornecido pelo KataGo, usando formato JSON para comunicacao, mais adequado para uso programatico.

### Iniciar Analysis Engine

```bash
katago analysis -model /path/to/model.bin.gz -config /path/to/config.cfg
```

### Fluxo de Uso Basico

```
Seu programa ──Requisicao JSON──> KataGo Analysis Engine ──Resposta JSON──> Seu programa
```

### Formato de Requisicao

Cada requisicao e um objeto JSON, deve ocupar uma linha:

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

### Descricao dos Campos de Requisicao

| Campo | Obrigatorio | Descricao |
|------|------|------|
| `id` | Sim | ID da consulta, usado para corresponder resposta |
| `moves` | Nao | Sequencia de jogadas `[["B","Q16"],["W","D4"]]` |
| `initialStones` | Nao | Pedras iniciais `[["B","Q16"],["W","D4"]]` |
| `rules` | Sim | Nome das regras |
| `komi` | Sim | Komi |
| `boardXSize` | Sim | Largura do tabuleiro |
| `boardYSize` | Sim | Altura do tabuleiro |
| `analyzeTurns` | Nao | Numero de jogadas a analisar (0-indexed) |
| `maxVisits` | Nao | Sobrescreve maxVisits do arquivo de configuracao |

### Formato de Resposta

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

### Descricao dos Campos de Resposta

#### Campos moveInfos

| Campo | Descricao |
|------|------|
| `move` | Coordenada da jogada |
| `visits` | Numero de visitas de busca para esta jogada |
| `winrate` | Taxa de vitoria (0-1, para jogador atual) |
| `scoreMean` | Diferenca de pontos final esperada |
| `scoreStdev` | Desvio padrao de pontos |
| `scoreLead` | Pontos de lideranca atual |
| `prior` | Probabilidade prior da rede neural |
| `order` | Ranking (0 = melhor) |
| `pv` | Sequencia de variacao principal |

#### Campos rootInfo

| Campo | Descricao |
|------|------|
| `visits` | Numero total de visitas de busca |
| `winrate` | Taxa de vitoria da posicao atual |
| `scoreLead` | Pontos de lideranca atual |
| `scoreSelfplay` | Pontos esperados em self-play |

#### Campo ownership

Array unidimensional, comprimento boardXSize x boardYSize, cada valor entre -1 e 1:
- -1: Previsto como territorio branco
- +1: Previsto como territorio preto
- 0: Indefinido/fronteira

### Opcoes de Consulta Avancadas

#### Obter Mapa de Territorio

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

#### Obter Distribuicao Policy

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

#### Limitar Numero de Jogadas Reportadas

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

#### Analisar Jogadas Especificas

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

### Exemplo Completo: Integracao Python

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

        # Enviar consulta
        self.process.stdin.write(json.dumps(query) + '\n')
        self.process.stdin.flush()

        # Ler resposta
        response_line = self.process.stdout.readline()
        return json.loads(response_line)

    def close(self):
        self.process.terminate()


# Exemplo de uso
engine = KataGoEngine(
    '/usr/local/bin/katago',
    '/path/to/model.bin.gz',
    '/path/to/config.cfg'
)

# Analisar uma posicao
result = engine.analyze([
    ['B', 'Q16'],
    ['W', 'D4'],
    ['B', 'Q4'],
    ['W', 'D16']
])

# Imprimir melhor jogada
best_move = result['moveInfos'][0]
print(f"Melhor jogada: {best_move['move']}")
print(f"Taxa de vitoria: {best_move['winrate']:.1%}")
print(f"Pontos de lideranca: {best_move['scoreLead']:.1f}")

engine.close()
```

### Exemplo Completo: Integracao Node.js

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

// Exemplo de uso
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

  console.log('Melhor jogada:', result.moveInfos[0].move);
  console.log('Taxa de vitoria:', (result.moveInfos[0].winrate * 100).toFixed(1) + '%');

  engine.close();
}

main();
```

## Sistema de Coordenadas

KataGo usa o sistema de coordenadas padrao de Go:

### Coordenadas por Letras

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

Nota: Nao ha letra I (para evitar confusao com numero 1).

### Conversao de Coordenadas

```python
def coord_to_gtp(x, y, board_size=19):
    """Converter coordenadas (x, y) para formato GTP"""
    letters = 'ABCDEFGHJKLMNOPQRST'
    return f"{letters[x]}{board_size - y}"

def gtp_to_coord(gtp_coord, board_size=19):
    """Converter coordenadas GTP para (x, y)"""
    letters = 'ABCDEFGHJKLMNOPQRST'
    x = letters.index(gtp_coord[0].upper())
    y = board_size - int(gtp_coord[1:])
    return (x, y)
```

## Padroes de Uso Comuns

### Modo de Jogo

```bash
# Iniciar modo GTP
katago gtp -model model.bin.gz -config gtp.cfg

# Sequencia de comandos GTP
boardsize 19
komi 7.5
play black Q16
genmove white
play black Q4
genmove white
...
```

### Modo de Analise em Lote

```python
# Analisar todas as jogadas de um jogo
sgf_moves = parse_sgf('game.sgf')

for i in range(len(sgf_moves)):
    result = engine.analyze(sgf_moves[:i+1])
    winrate = result['rootInfo']['winrate']
    print(f"Jogada {i+1}: Taxa de vitoria {winrate:.1%}")
```

### Modo de Analise em Tempo Real

Use `kata-analyze` para analise em tempo real:

```
kata-analyze black 1000 50
```

Mostrara resultados de analise a cada 0.5 segundos, ate atingir 1000 visitas.

## Otimizacao de Desempenho

### Configuracao de Busca

```ini
# Aumentar volume de busca para melhorar precisao
maxVisits = 1000

# Ou usar controle de tempo
maxTime = 10  # Pensar no maximo 10 segundos por jogada
```

### Configuracao Multi-thread

```ini
# Numero de threads CPU
numSearchThreads = 8

# Processamento em lote GPU
numNNServerThreadsPerModel = 2
nnMaxBatchSize = 16
```

### Configuracao de Memoria

```ini
# Reduzir uso de memoria
nnCacheSizePowerOfTwo = 20  # Padrao 23
```

## Proximos Passos

Apos entender o uso de comandos, se voce quiser pesquisar profundamente a implementacao do KataGo, continue lendo [Arquitetura do Codigo-fonte](./architecture.md).

