---
sidebar_position: 2
title: Comandos comunes
---

# Comandos comunes de KataGo

Este articulo presenta los dos modos principales de operacion de KataGo: Protocolo GTP y Analysis Engine, asi como explicaciones detalladas de comandos comunes.

## Introduccion al protocolo GTP

GTP (Go Text Protocol) es el protocolo estandar de comunicacion entre programas de Go. La mayoria de las GUI de Go (como Sabaki, Lizzie) usan GTP para comunicarse con motores de IA.

### Iniciar modo GTP

```bash
katago gtp -model /ruta/al/modelo.bin.gz -config /ruta/al/config.cfg
```

### Formato basico del protocolo GTP

```
[id] nombre_comando [argumentos]
```

- `id`: Numero de comando opcional, usado para rastrear respuestas
- `nombre_comando`: Nombre del comando
- `argumentos`: Parametros del comando

Formato de respuesta:
```
=[id] datos_respuesta     # Exito
?[id] mensaje_error       # Fallo
```

### Ejemplo basico

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

## Comandos GTP comunes

### Informacion del programa

| Comando | Descripcion | Ejemplo |
|------|------|------|
| `name` | Obtener nombre del programa | `name` -> `= KataGo` |
| `version` | Obtener numero de version | `version` -> `= 1.15.3` |
| `list_commands` | Listar todos los comandos soportados | `list_commands` |
| `protocol_version` | Version del protocolo GTP | `protocol_version` -> `= 2` |

### Configuracion del tablero

```
# Establecer tamano del tablero (9, 13, 19)
boardsize 19

# Establecer komi
komi 7.5

# Limpiar tablero
clear_board

# Establecer reglas (extension KataGo)
kata-set-rules chinese    # Reglas chinas
kata-set-rules japanese   # Reglas japonesas
kata-set-rules tromp-taylor
```

### Relacionado con juego

```
# Jugar
play black Q16    # Negro juega en Q16
play white D4     # Blanco juega en D4
play black pass   # Negro pasa

# Hacer que IA juegue una jugada
genmove black     # Generar jugada de negro
genmove white     # Generar jugada de blanco

# Deshacer
undo              # Deshacer una jugada

# Establecer limite de jugadas
kata-set-param maxVisits 1000    # Establecer maximo de busquedas
```

### Consulta de posicion

```
# Mostrar tablero
showboard

# Obtener lado actual para jugar
kata-get-player

# Obtener resultado de analisis
kata-analyze black 100    # Analizar negro, buscar 100 veces
```

### Relacionado con reglas

```
# Obtener reglas actuales
kata-get-rules

# Establecer reglas
kata-set-rules chinese

# Establecer handicap
fixed_handicap 4     # Posiciones estandar de 4 handicap
place_free_handicap 4  # Handicap libre
```

## Comandos de extension KataGo

KataGo proporciona muchos comandos de extension ademas del GTP estandar:

### kata-analyze

Analizar la posicion actual en tiempo real:

```
kata-analyze [jugador] [visitas] [intervalo]
```

Parametros:
- `jugador`: Que lado analizar (black/white)
- `visitas`: Numero de busquedas
- `intervalo`: Intervalo de reporte (centisegundos, 1/100 segundo)

Ejemplo:
```
kata-analyze black 1000 100
```

Salida:
```
info move Q3 visits 523 winrate 0.5432 scoreMean 2.31 scoreSelfplay 2.45 prior 0.1234 order 0 pv Q3 R4 Q5 ...
info move D4 visits 312 winrate 0.5123 scoreMean 1.82 scoreSelfplay 1.95 prior 0.0987 order 1 pv D4 C6 E3 ...
...
```

Explicacion de campos de salida:

| Campo | Descripcion |
|------|------|
| `move` | Punto de jugada |
| `visits` | Numero de visitas de busqueda |
| `winrate` | Tasa de victoria (0-1) |
| `scoreMean` | Diferencia de puntos esperada |
| `scoreSelfplay` | Puntos esperados en self-play |
| `prior` | Probabilidad a priori de red neuronal |
| `order` | Orden de ranking |
| `pv` | Variacion principal (Principal Variation) |

### kata-raw-nn

Obtener salida cruda de red neuronal:

```
kata-raw-nn [simetria]
```

Salida incluye:
- Distribucion de probabilidad de Policy
- Prediccion de Value
- Prediccion de territorio, etc.

### kata-debug-print

Mostrar informacion detallada de busqueda, para depuracion:

```
kata-debug-print move Q16
```

### Ajuste de fuerza

```
# Establecer numero maximo de visitas
kata-set-param maxVisits 100      # Mas debil
kata-set-param maxVisits 10000    # Mas fuerte

# Establecer tiempo de pensamiento
kata-time-settings main 60 0      # 60 segundos por lado
kata-time-settings byoyomi 30 5   # 30 segundos byoyomi 5 periodos
```

## Uso de Analysis Engine

Analysis Engine es otro modo de operacion proporcionado por KataGo, usa formato JSON para comunicacion, mas adecuado para uso programatico.

### Iniciar Analysis Engine

```bash
katago analysis -model /ruta/al/modelo.bin.gz -config /ruta/al/config.cfg
```

### Flujo de uso basico

```
Tu programa --Solicitud JSON--> KataGo Analysis Engine --Respuesta JSON--> Tu programa
```

### Formato de solicitud

Cada solicitud es un objeto JSON, debe ocupar una linea:

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

### Explicacion de campos de solicitud

| Campo | Requerido | Descripcion |
|------|------|------|
| `id` | Si | Identificador de consulta, usado para corresponder respuestas |
| `moves` | No | Secuencia de jugadas `[["B","Q16"],["W","D4"]]` |
| `initialStones` | No | Piedras iniciales `[["B","Q16"],["W","D4"]]` |
| `rules` | Si | Nombre de reglas |
| `komi` | Si | Komi |
| `boardXSize` | Si | Ancho del tablero |
| `boardYSize` | Si | Alto del tablero |
| `analyzeTurns` | No | Turnos a analizar (indexados desde 0) |
| `maxVisits` | No | Sobrescribir maxVisits del archivo de configuracion |

### Formato de respuesta

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

### Explicacion de campos de respuesta

#### Campos de moveInfos

| Campo | Descripcion |
|------|------|
| `move` | Coordenada del punto |
| `visits` | Numero de visitas de busqueda para esa jugada |
| `winrate` | Tasa de victoria (0-1, para el lado actual) |
| `scoreMean` | Diferencia de puntos esperada final |
| `scoreStdev` | Desviacion estandar de puntos |
| `scoreLead` | Puntos liderando actualmente |
| `prior` | Probabilidad a priori de red neuronal |
| `order` | Ranking (0 = mejor) |
| `pv` | Secuencia de variacion principal |

#### Campos de rootInfo

| Campo | Descripcion |
|------|------|
| `visits` | Total de visitas de busqueda |
| `winrate` | Tasa de victoria de posicion actual |
| `scoreLead` | Puntos liderando actualmente |
| `scoreSelfplay` | Puntos esperados en self-play |

#### Campo ownership

Array unidimensional, longitud boardXSize x boardYSize, cada valor entre -1 y 1:
- -1: Predicho como territorio blanco
- +1: Predicho como territorio negro
- 0: Indefinido/borde

### Opciones avanzadas de consulta

#### Obtener mapa de territorio

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

#### Obtener distribucion de Policy

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

#### Limitar numero de jugadas reportadas

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

#### Analizar jugadas especificas

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

### Ejemplo completo: Integracion con Python

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

        # Leer respuesta
        response_line = self.process.stdout.readline()
        return json.loads(response_line)

    def close(self):
        self.process.terminate()


# Ejemplo de uso
engine = KataGoEngine(
    '/usr/local/bin/katago',
    '/ruta/al/modelo.bin.gz',
    '/ruta/al/config.cfg'
)

# Analizar una posicion
result = engine.analyze([
    ['B', 'Q16'],
    ['W', 'D4'],
    ['B', 'Q4'],
    ['W', 'D16']
])

# Imprimir mejor jugada
best_move = result['moveInfos'][0]
print(f"Mejor jugada: {best_move['move']}")
print(f"Tasa de victoria: {best_move['winrate']:.1%}")
print(f"Puntos liderando: {best_move['scoreLead']:.1f}")

engine.close()
```

### Ejemplo completo: Integracion con Node.js

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
        console.error('Error de parseo:', e);
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

// Ejemplo de uso
async function main() {
  const engine = new KataGoEngine(
    '/usr/local/bin/katago',
    '/ruta/al/modelo.bin.gz',
    '/ruta/al/config.cfg'
  );

  const result = await engine.analyze([
    ['B', 'Q16'],
    ['W', 'D4'],
    ['B', 'Q4']
  ]);

  console.log('Mejor jugada:', result.moveInfos[0].move);
  console.log('Tasa de victoria:', (result.moveInfos[0].winrate * 100).toFixed(1) + '%');

  engine.close();
}

main();
```

## Sistema de coordenadas

KataGo usa el sistema de coordenadas estandar de Go:

### Coordenadas de letras

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

Nota: No hay letra I (para evitar confusion con el numero 1).

### Conversion de coordenadas

```python
def coord_to_gtp(x, y, board_size=19):
    """Convertir coordenadas (x, y) a formato GTP"""
    letters = 'ABCDEFGHJKLMNOPQRST'
    return f"{letters[x]}{board_size - y}"

def gtp_to_coord(gtp_coord, board_size=19):
    """Convertir coordenadas GTP a (x, y)"""
    letters = 'ABCDEFGHJKLMNOPQRST'
    x = letters.index(gtp_coord[0].upper())
    y = board_size - int(gtp_coord[1:])
    return (x, y)
```

## Patrones de uso comunes

### Modo de juego

```bash
# Iniciar modo GTP
katago gtp -model model.bin.gz -config gtp.cfg

# Secuencia de comandos GTP
boardsize 19
komi 7.5
play black Q16
genmove white
play black Q4
genmove white
...
```

### Modo de analisis por lotes

```python
# Analizar todas las jugadas de una partida
sgf_moves = parse_sgf('game.sgf')

for i in range(len(sgf_moves)):
    result = engine.analyze(sgf_moves[:i+1])
    winrate = result['rootInfo']['winrate']
    print(f"Turno {i+1}: Tasa de victoria {winrate:.1%}")
```

### Modo de analisis en tiempo real

Usar `kata-analyze` para analisis en tiempo real:

```
kata-analyze black 1000 50
```

Mostrara resultado de analisis cada 0.5 segundos hasta alcanzar 1000 visitas.

## Optimizacion de rendimiento

### Configuracion de busqueda

```ini
# Aumentar cantidad de busqueda mejora precision
maxVisits = 1000

# O usar control de tiempo
maxTime = 10  # Maximo 10 segundos por jugada
```

### Configuracion multi-hilo

```ini
# Numero de hilos CPU
numSearchThreads = 8

# Procesamiento por lotes GPU
numNNServerThreadsPerModel = 2
nnMaxBatchSize = 16
```

### Configuracion de memoria

```ini
# Reducir uso de memoria
nnCacheSizePowerOfTwo = 20  # Por defecto 23
```

## Siguientes pasos

Despues de conocer el uso de comandos, si quieres investigar profundamente la implementacion de KataGo, continua leyendo [Arquitectura del codigo fuente](./architecture.md).

