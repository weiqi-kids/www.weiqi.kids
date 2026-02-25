---
sidebar_position: 2
title: Guia del codigo fuente de KataGo
description: Estructura del codigo de KataGo, modulos principales y diseno de arquitectura
---

# Guia del codigo fuente de KataGo

Este articulo te guia a traves de la estructura del codigo de KataGo, ideal para ingenieros que desean investigar a fondo o contribuir codigo.

---

## Obtener el codigo fuente

```bash
git clone https://github.com/lightvector/KataGo.git
cd KataGo
```

---

## Estructura de directorios

```
KataGo/
├── cpp/                    # Motor principal en C++
│   ├── main.cpp            # Punto de entrada principal
│   ├── command/            # Procesamiento de comandos
│   ├── core/               # Utilidades principales
│   ├── game/               # Reglas de Go
│   ├── search/             # Busqueda MCTS
│   ├── neuralnet/          # Inferencia de red neuronal
│   ├── dataio/             # I/O de datos
│   └── tests/              # Pruebas unitarias
│
├── python/                 # Codigo de entrenamiento Python
│   ├── train.py            # Programa principal de entrenamiento
│   ├── model.py            # Definicion de arquitectura de red
│   ├── data/               # Procesamiento de datos
│   └── configs/            # Configuraciones de entrenamiento
│
└── docs/                   # Documentacion
```

---

## Analisis de modulos principales

### 1. game/ — Reglas de Go

Implementacion completa de las reglas de Go.

#### board.h / board.cpp

```cpp
// Representacion del estado del tablero
class Board {
public:
    static constexpr int MAX_BOARD_SIZE = 19;

    // Estado del tablero
    Color colors[MAX_ARR_SIZE];  // Color de cada posicion
    Chain chains[MAX_ARR_SIZE];  // Informacion de cadenas

    // Operaciones principales
    bool playMove(Loc loc, Player pla);  // Jugar un movimiento
    bool isLegal(Loc loc, Player pla);   // Verificar legalidad
    void calculateArea(Color* area);      // Calcular territorio
};
```

**Correspondencia con animaciones**:
- A2 Modelo de red: Estructura de datos del tablero
- A6 Region conectada: Representacion de cadenas (Chain)
- A7 Calculo de libertades: Seguimiento de libertades

#### rules.h / rules.cpp

```cpp
// Soporte multi-reglas
struct Rules {
    enum KoRule { SIMPLE_KO, POSITIONAL_KO, SITUATIONAL_KO };
    enum ScoringRule { TERRITORY_SCORING, AREA_SCORING };
    enum TaxRule { NO_TAX, TAX_SEKI, TAX_ALL };

    KoRule koRule;
    ScoringRule scoringRule;
    TaxRule taxRule;
    float komi;

    // Mapeo de nombres de reglas
    static Rules parseRules(const std::string& name);
};
```

Reglas soportadas:
- `chinese`: Reglas chinas (conteo de area)
- `japanese`: Reglas japonesas (conteo de territorio)
- `korean`: Reglas coreanas
- `aga`: Reglas americanas
- `tromp-taylor`: Reglas Tromp-Taylor

---

### 2. search/ — Busqueda MCTS

Implementacion de Monte Carlo Tree Search.

#### search.h / search.cpp

```cpp
class Search {
public:
    // Busqueda principal
    void runWholeSearch(Player pla);

    // Pasos de MCTS
    void selectNode();           // Seleccionar nodo
    void expandNode();           // Expandir nodo
    void evaluateNode();         // Evaluacion con red neuronal
    void backpropValue();        // Retropropagar actualizacion

    // Obtener resultados
    Loc getChosenMove();
    std::vector<MoveInfo> getSortedMoveInfos();
};
```

**Correspondencia con animaciones**:
- C5 Cuatro pasos de MCTS: Corresponde a select → expand → evaluate → backprop
- E4 Formula PUCT: Implementada en `selectNode()`

#### searchparams.h

```cpp
struct SearchParams {
    // Control de busqueda
    int64_t maxVisits;          // Visitas maximas
    double maxTime;             // Tiempo maximo

    // Parametros PUCT
    double cpuctExploration;    // Constante de exploracion
    double cpuctBase;

    // Perdida virtual
    int virtualLoss;

    // Ruido en nodo raiz
    double rootNoiseEnabled;
    double rootDirichletAlpha;
};
```

---

### 3. neuralnet/ — Inferencia de red neuronal

Motor de inferencia de red neuronal.

#### nninputs.h / nninputs.cpp

```cpp
// Caracteristicas de entrada de red neuronal
class NNInputs {
public:
    // Planos de caracteristicas
    static constexpr int NUM_FEATURES = 22;

    // Rellenar caracteristicas
    static void fillFeatures(
        const Board& board,
        const BoardHistory& hist,
        float* features
    );
};
```

Las caracteristicas de entrada incluyen:
- Posicion de piedras negras, posicion de piedras blancas
- Numero de libertades (1, 2, 3+)
- Movimientos historicos
- Codificacion de reglas

**Correspondencia con animaciones**:
- A10 Apilamiento historico: Entrada multi-frame
- A11 Mascara de movimientos legales: Filtrado de movimientos prohibidos

#### nneval.h / nneval.cpp

```cpp
// Resultado de evaluacion de red neuronal
struct NNOutput {
    // Salida de Policy (362 posiciones, incluye pass)
    float policyProbs[NNPos::MAX_NN_POLICY_SIZE];

    // Salida de Value
    float winProb;       // Probabilidad de victoria
    float lossProb;      // Probabilidad de derrota
    float noResultProb;  // Probabilidad de empate

    // Salidas auxiliares
    float scoreMean;     // Prediccion de puntos
    float scoreStdev;    // Desviacion estandar de puntos
    float lead;          // Puntos de ventaja

    // Prediccion de territorio
    float ownership[NNPos::MAX_BOARD_AREA];
};
```

**Correspondencia con animaciones**:
- E1 Red de politica: policyProbs
- E2 Red de valor: winProb, scoreMean
- E3 Red de doble cabeza: Diseno de salida multiple

---

### 4. command/ — Procesamiento de comandos

Implementacion de diferentes modos de ejecucion.

#### gtp.cpp

Implementacion del modo GTP (Go Text Protocol):

```cpp
void MainCmds::gtp(const std::vector<std::string>& args) {
    // Analisis y ejecucion de comandos
    while(true) {
        std::string line;
        std::getline(std::cin, line);

        if(line == "name") {
            respond("KataGo");
        }
        else if(line.find("play") == 0) {
            // Procesar comando de jugar
        }
        else if(line.find("genmove") == 0) {
            // Ejecutar busqueda y devolver mejor movimiento
        }
        // ... otros comandos
    }
}
```

#### analysis.cpp

Implementacion del Analysis Engine:

```cpp
void MainCmds::analysis(const std::vector<std::string>& args) {
    while(true) {
        // Leer solicitud JSON
        std::string line;
        std::getline(std::cin, line);
        json query = json::parse(line);

        // Configurar estado del tablero
        Board board = setupBoard(query);

        // Ejecutar analisis
        Search search(...);
        search.runWholeSearch();

        // Salida de respuesta JSON
        json response = formatResponse(search);
        std::cout << response.dump() << std::endl;
    }
}
```

---

## Codigo de entrenamiento Python

### model.py — Arquitectura de red

```python
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Convolucion inicial
        self.initial_conv = nn.Conv2d(
            in_channels=config.input_features,
            out_channels=config.trunk_channels,
            kernel_size=3, padding=1
        )

        # Torre residual
        self.trunk = nn.ModuleList([
            ResidualBlock(config.trunk_channels)
            for _ in range(config.num_blocks)
        ])

        # Cabezas de salida
        self.policy_head = PolicyHead(config)
        self.value_head = ValueHead(config)
        self.ownership_head = OwnershipHead(config)

    def forward(self, x):
        # Convolucion inicial
        x = self.initial_conv(x)

        # Torre residual
        for block in self.trunk:
            x = block(x)

        # Salida multiple
        policy = self.policy_head(x)
        value = self.value_head(x)
        ownership = self.ownership_head(x)

        return policy, value, ownership
```

**Correspondencia con animaciones**:
- D9 Operacion de convolucion: Conv2d
- D12 Conexion residual: ResidualBlock
- E11 Torre residual: estructura trunk

### train.py — Ciclo de entrenamiento

```python
def train_step(model, optimizer, batch):
    # Propagacion hacia adelante
    policy_pred, value_pred, ownership_pred = model(batch.inputs)

    # Calcular perdida
    policy_loss = cross_entropy(policy_pred, batch.policy_target)
    value_loss = mse_loss(value_pred, batch.value_target)
    ownership_loss = mse_loss(ownership_pred, batch.ownership_target)

    total_loss = policy_loss + value_loss + ownership_loss

    # Retropropagacion
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()
```

**Correspondencia con animaciones**:
- D3 Propagacion hacia adelante: model(batch.inputs)
- D13 Retropropagacion: total_loss.backward()
- K3 Adam: optimizer.step()

---

## Implementacion de algoritmos clave

### Formula de seleccion PUCT

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

### Perdida virtual

```cpp
// Evitar que multiples hilos seleccionen el mismo nodo
void Search::applyVirtualLoss(SearchNode* node) {
    node->virtualLoss += params.virtualLoss;
}

void Search::removeVirtualLoss(SearchNode* node) {
    node->virtualLoss -= params.virtualLoss;
}
```

**Correspondencia con animaciones**:
- C9 Perdida virtual: Tecnica de busqueda paralela

---

## Compilacion y depuracion

### Compilacion (modo Debug)

```bash
cd cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=OPENCL -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

### Pruebas unitarias

```bash
./katago runtests
```

### Tecnicas de depuracion

```cpp
// Habilitar logs detallados
#define SEARCH_DEBUG 1

// Agregar punto de interrupcion en la busqueda
if(node->visits > 1000) {
    // Establecer breakpoint para inspeccionar estado de busqueda
}
```

---

## Lectura adicional

- [Analisis del mecanismo de entrenamiento de KataGo](../training) — Proceso completo de entrenamiento
- [Participar en la comunidad de codigo abierto](../contributing) — Guia de contribucion
- [Hoja de referencia de conceptos](../../how-it-works/concepts/) — Referencia de 109 conceptos
