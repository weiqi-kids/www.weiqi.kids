---
sidebar_position: 2
title: Guia do Codigo-fonte do KataGo
description: Estrutura de codigo do KataGo, modulos principais e design de arquitetura
---

# Guia do Codigo-fonte do KataGo

Este artigo ajuda voce a entender a estrutura de codigo do KataGo, adequado para engenheiros que desejam pesquisar a fundo ou contribuir com codigo.

---

## Obter o Codigo-fonte

```bash
git clone https://github.com/lightvector/KataGo.git
cd KataGo
```

---

## Estrutura de Diretorios

```
KataGo/
├── cpp/                    # Engine principal em C++
│   ├── main.cpp            # Ponto de entrada do programa
│   ├── command/            # Tratamento de comandos
│   ├── core/               # Utilitarios principais
│   ├── game/               # Regras de Go
│   ├── search/             # Busca MCTS
│   ├── neuralnet/          # Inferencia de rede neural
│   ├── dataio/             # I/O de dados
│   └── tests/              # Testes unitarios
│
├── python/                 # Codigo de treinamento Python
│   ├── train.py            # Programa principal de treinamento
│   ├── model.py            # Definicao da arquitetura de rede
│   ├── data/               # Processamento de dados
│   └── configs/            # Configuracoes de treinamento
│
└── docs/                   # Documentacao
```

---

## Analise dos Modulos Principais

### 1. game/ — Regras de Go

Implementacao completa das regras de Go.

#### board.h / board.cpp

```cpp
// Representacao do estado do tabuleiro
class Board {
public:
    static constexpr int MAX_BOARD_SIZE = 19;

    // Estado do tabuleiro
    Color colors[MAX_ARR_SIZE];  // Cor de cada posicao
    Chain chains[MAX_ARR_SIZE];  // Informacao de grupos

    // Operacoes principais
    bool playMove(Loc loc, Player pla);  // Jogar uma pedra
    bool isLegal(Loc loc, Player pla);   // Verificar legalidade
    void calculateArea(Color* area);      // Calcular territorio
};
```

**Correspondencia com animacoes**:
- A2 Modelo de rede: Estrutura de dados do tabuleiro
- A6 Regiao conectada: Representacao de grupos (Chain)
- A7 Calculo de liberdades: Rastreamento de liberdades

#### rules.h / rules.cpp

```cpp
// Suporte a multiplas regras
struct Rules {
    enum KoRule { SIMPLE_KO, POSITIONAL_KO, SITUATIONAL_KO };
    enum ScoringRule { TERRITORY_SCORING, AREA_SCORING };
    enum TaxRule { NO_TAX, TAX_SEKI, TAX_ALL };

    KoRule koRule;
    ScoringRule scoringRule;
    TaxRule taxRule;
    float komi;

    // Mapeamento de nomes de regras
    static Rules parseRules(const std::string& name);
};
```

Regras suportadas:
- `chinese`: Regras Chinesas (contagem de area)
- `japanese`: Regras Japonesas (contagem de territorio)
- `korean`: Regras Coreanas
- `aga`: Regras Americanas
- `tromp-taylor`: Regras Tromp-Taylor

---

### 2. search/ — Busca MCTS

Implementacao da Busca em Arvore de Monte Carlo.

#### search.h / search.cpp

```cpp
class Search {
public:
    // Busca principal
    void runWholeSearch(Player pla);

    // Etapas do MCTS
    void selectNode();           // Selecionar no
    void expandNode();           // Expandir no
    void evaluateNode();         // Avaliacao pela rede neural
    void backpropValue();        // Atualizar retropropagacao

    // Obter resultados
    Loc getChosenMove();
    std::vector<MoveInfo> getSortedMoveInfos();
};
```

**Correspondencia com animacoes**:
- C5 Quatro etapas do MCTS: Corresponde a select → expand → evaluate → backprop
- E4 Formula PUCT: Implementada em `selectNode()`

#### searchparams.h

```cpp
struct SearchParams {
    // Controle de busca
    int64_t maxVisits;          // Maximo de visitas
    double maxTime;             // Tempo maximo

    // Parametros PUCT
    double cpuctExploration;    // Constante de exploracao
    double cpuctBase;

    // Perda virtual
    int virtualLoss;

    // Ruido no no raiz
    double rootNoiseEnabled;
    double rootDirichletAlpha;
};
```

---

### 3. neuralnet/ — Inferencia de Rede Neural

Engine de inferencia de rede neural.

#### nninputs.h / nninputs.cpp

```cpp
// Recursos de entrada da rede neural
class NNInputs {
public:
    // Planos de recursos
    static constexpr int NUM_FEATURES = 22;

    // Preencher recursos
    static void fillFeatures(
        const Board& board,
        const BoardHistory& hist,
        float* features
    );
};
```

Recursos de entrada incluem:
- Posicao de pedras pretas, posicao de pedras brancas
- Numero de liberdades (1, 2, 3+)
- Jogadas historicas
- Codificacao de regras

**Correspondencia com animacoes**:
- A10 Empilhamento historico: Entrada multi-frame
- A11 Mascara de jogadas legais: Filtragem de jogadas proibidas

#### nneval.h / nneval.cpp

```cpp
// Resultado de avaliacao da rede neural
struct NNOutput {
    // Saida de Policy (362 posicoes, incluindo pass)
    float policyProbs[NNPos::MAX_NN_POLICY_SIZE];

    // Saida de Value
    float winProb;       // Taxa de vitoria
    float lossProb;      // Taxa de derrota
    float noResultProb;  // Taxa de empate

    // Saidas auxiliares
    float scoreMean;     // Previsao de pontos
    float scoreStdev;    // Desvio padrao de pontos
    float lead;          // Pontos de lideranca

    // Previsao de territorio
    float ownership[NNPos::MAX_BOARD_AREA];
};
```

**Correspondencia com animacoes**:
- E1 Rede de politica: policyProbs
- E2 Rede de valor: winProb, scoreMean
- E3 Rede de duas cabecas: Design de multiplas saidas

---

### 4. command/ — Tratamento de Comandos

Implementacao de diferentes modos de execucao.

#### gtp.cpp

Implementacao do modo GTP (Go Text Protocol):

```cpp
void MainCmds::gtp(const std::vector<std::string>& args) {
    // Parse e execucao de comandos
    while(true) {
        std::string line;
        std::getline(std::cin, line);

        if(line == "name") {
            respond("KataGo");
        }
        else if(line.find("play") == 0) {
            // Tratar comando de jogada
        }
        else if(line.find("genmove") == 0) {
            // Executar busca e retornar melhor jogada
        }
        // ... outros comandos
    }
}
```

#### analysis.cpp

Implementacao do Analysis Engine:

```cpp
void MainCmds::analysis(const std::vector<std::string>& args) {
    while(true) {
        // Ler requisicao JSON
        std::string line;
        std::getline(std::cin, line);
        json query = json::parse(line);

        // Configurar estado do tabuleiro
        Board board = setupBoard(query);

        // Executar analise
        Search search(...);
        search.runWholeSearch();

        // Gerar resposta JSON
        json response = formatResponse(search);
        std::cout << response.dump() << std::endl;
    }
}
```

---

## Codigo de Treinamento Python

### model.py — Arquitetura de Rede

```python
class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Convolucao inicial
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

        # Cabecas de saida
        self.policy_head = PolicyHead(config)
        self.value_head = ValueHead(config)
        self.ownership_head = OwnershipHead(config)

    def forward(self, x):
        # Convolucao inicial
        x = self.initial_conv(x)

        # Torre residual
        for block in self.trunk:
            x = block(x)

        # Multiplas saidas
        policy = self.policy_head(x)
        value = self.value_head(x)
        ownership = self.ownership_head(x)

        return policy, value, ownership
```

**Correspondencia com animacoes**:
- D9 Operacao de convolucao: Conv2d
- D12 Conexao residual: ResidualBlock
- E11 Torre residual: Estrutura do trunk

### train.py — Loop de Treinamento

```python
def train_step(model, optimizer, batch):
    # Forward pass
    policy_pred, value_pred, ownership_pred = model(batch.inputs)

    # Calcular perda
    policy_loss = cross_entropy(policy_pred, batch.policy_target)
    value_loss = mse_loss(value_pred, batch.value_target)
    ownership_loss = mse_loss(ownership_pred, batch.ownership_target)

    total_loss = policy_loss + value_loss + ownership_loss

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()
```

**Correspondencia com animacoes**:
- D3 Forward pass: model(batch.inputs)
- D13 Backward pass: total_loss.backward()
- K3 Adam: optimizer.step()

---

## Implementacoes de Algoritmos-Chave

### Formula de Selecao PUCT

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

### Perda Virtual

```cpp
// Evita que multiplas threads selecionem o mesmo no
void Search::applyVirtualLoss(SearchNode* node) {
    node->virtualLoss += params.virtualLoss;
}

void Search::removeVirtualLoss(SearchNode* node) {
    node->virtualLoss -= params.virtualLoss;
}
```

**Correspondencia com animacoes**:
- C9 Perda virtual: Tecnica de busca paralela

---

## Compilacao e Depuracao

### Compilacao (Modo Debug)

```bash
cd cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=OPENCL -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

### Testes Unitarios

```bash
./katago runtests
```

### Dicas de Depuracao

```cpp
// Habilitar logs detalhados
#define SEARCH_DEBUG 1

// Adicionar breakpoint na busca
if(node->visits > 1000) {
    // Configurar breakpoint para verificar estado da busca
}
```

---

## Leitura Adicional

- [Analise do Mecanismo de Treinamento do KataGo](../training) — Fluxo completo de treinamento
- [Participando da Comunidade Open Source](../contributing) — Guia de contribuicao
- [Tabela de Referencia de Conceitos](/docs/animations/) — Comparacao de 109 conceitos
