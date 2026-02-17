---
sidebar_position: 1
title: Conhecimento de Fundo
---

# Visao Geral do Conhecimento de Fundo

Antes de mergulhar na pratica do KataGo, entender a historia do desenvolvimento e a tecnologia central da IA de Go e muito importante. Este capitulo levara voce a entender a evolucao tecnologica desde AlphaGo ate a IA de Go moderna.

## Por que Entender o Conhecimento de Fundo?

O desenvolvimento da IA de Go e um dos avancos mais empolgantes no campo da inteligencia artificial. A partida de 2016 onde AlphaGo derrotou Lee Sedol nao foi apenas um marco na historia do Go, mas tambem marcou o tremendo sucesso da combinacao de aprendizado profundo e aprendizado por reforco.

Entender esse conhecimento de fundo pode ajuda-lo a:

- **Tomar melhores decisoes tecnicas**: Entender os pros e contras de varios metodos, escolher a solucao adequada para seu projeto
- **Depurar mais efetivamente**: Entender os principios subjacentes facilita diagnosticar problemas
- **Acompanhar os ultimos desenvolvimentos**: Dominar o conhecimento fundamental facilita entender novos artigos e novas tecnologias
- **Contribuir para projetos de codigo aberto**: Participar do desenvolvimento de projetos como KataGo requer entendimento profundo de sua filosofia de design

## Conteudo deste Capitulo

### [Analise do Artigo AlphaGo](./alphago.md)

Analise detalhada do artigo classico da DeepMind, incluindo:

- Significado historico e impacto do AlphaGo
- Design de Policy Network e Value Network
- Principios e implementacao da Busca por Arvore de Monte Carlo (MCTS)
- Inovacao do metodo de treinamento Self-play
- Evolucao de AlphaGo para AlphaGo Zero e depois para AlphaZero

### [Analise do Artigo KataGo](./katago-paper.md)

Entenda as inovacoes tecnicas da IA de Go de codigo aberto mais forte atualmente:

- Melhorias do KataGo em relacao ao AlphaGo
- Metodos de treinamento mais eficientes e utilizacao de recursos
- Implementacao tecnica para suportar multiplas regras de Go
- Design para prever simultaneamente taxa de vitoria e contagem de pontos
- Por que KataGo consegue alcancar forca maior com menos recursos

### [Introducao a Outras IAs de Go](./zen.md)

Visao abrangente do ecossistema de IA de Go:

- IAs comerciais: Zen, JueYi (Tencent), Golaxy
- IAs de codigo aberto: Leela Zero, ELF OpenGo, SAI
- Comparacao de caracteristicas tecnicas e cenarios de aplicacao de cada IA

## Linha do Tempo do Desenvolvimento Tecnologico

| Tempo | Evento | Importancia |
|------|------|--------|
| Outubro 2015 | AlphaGo derrota Fan Hui | Primeira IA a derrotar jogador profissional |
| Marco 2016 | AlphaGo derrota Lee Sedol | Partida homem-maquina que chocou o mundo |
| Maio 2017 | AlphaGo derrota Ke Jie | Estabelece IA acima do nivel humano |
| Outubro 2017 | AlphaGo Zero publicado | Puro self-play, sem necessidade de registros humanos |
| Dezembro 2017 | AlphaZero publicado | Design generalizado, conquistando Go, xadrez e shogi simultaneamente |
| 2018 | Leela Zero atinge nivel super-humano | Vitoria da comunidade de codigo aberto |
| 2019 | KataGo publicado | Metodo de treinamento mais eficiente |
| 2020-presente | KataGo continua melhorando | Torna-se a IA de Go de codigo aberto mais forte |

## Previa dos Conceitos Centrais

Antes de ler os capitulos detalhados, aqui esta uma breve introducao a alguns conceitos centrais:

### Papel das Redes Neurais no Go

```
Estado do Tabuleiro → Rede Neural → { Policy (probabilidade de jogada), Value (avaliacao de taxa de vitoria) }
```

A rede neural recebe o estado atual do tabuleiro como entrada e produz dois tipos de informacao:
- **Policy**: Probabilidade de jogada para cada posicao, guiando a direcao da busca
- **Value**: Estimativa de taxa de vitoria da posicao atual, usada para avaliar a posicao

### Busca por Arvore de Monte Carlo (MCTS)

MCTS e um algoritmo de busca que combina redes neurais para determinar a melhor jogada:

1. **Selection (Selecao)**: Do no raiz, selecionar o caminho mais promissor
2. **Expansion (Expansao)**: No no folha, expandir novas jogadas possiveis
3. **Evaluation (Avaliacao)**: Usar rede neural para avaliar valor da posicao
4. **Backpropagation (Retropropagacao)**: Retornar resultados da avaliacao para atualizar nos no caminho

### Self-play (Auto-jogo)

IA joga contra si mesma para gerar dados de treinamento:

```
Modelo inicial → Self-play → Coletar registros → Treinar novo modelo → Modelo mais forte → Repetir
```

Este ciclo permite que a IA se aprimore continuamente sem depender de registros humanos.

## Ordem de Leitura Sugerida

1. **Primeiro leia Analise do Artigo AlphaGo**: Estabelecer framework teorico fundamental
2. **Depois leia Analise do Artigo KataGo**: Entender melhorias e otimizacoes mais recentes
3. **Por ultimo leia Introducao a Outras IAs de Go**: Ampliar visao, entender diferentes implementacoes

Pronto? Vamos comecar com a [Analise do Artigo AlphaGo](./alphago.md)!

