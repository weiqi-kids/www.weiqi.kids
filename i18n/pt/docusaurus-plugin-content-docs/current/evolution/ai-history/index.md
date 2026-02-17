---
sidebar_position: 2
title: Historia da IA no Go
---

# Historia da IA no Go

Por muito tempo, o Go foi considerado o jogo mais dificil de ser conquistado pela inteligencia artificial. Com 19x19 = 361 intersecoes no tabuleiro, cada ponto pode receber uma pedra, o numero de variacoes supera o total de atomos no universo (aproximadamente 10^170 jogos possiveis). Os metodos tradicionais de busca exaustiva simplesmente nao funcionam no Go.

No entanto, entre 2015 e 2017, a serie AlphaGo da DeepMind mudou tudo isso completamente. Esta revolucao nao apenas afetou o Go, mas impulsionou o desenvolvimento de todo o campo da inteligencia artificial.

## Por que o Go e tao Dificil?

### Espaco de Busca Gigantesco

Tomando o xadrez como exemplo, em media cada jogada tem cerca de 35 movimentos legais, e uma partida tem cerca de 80 jogadas. No Go, em media cada jogada tem cerca de 250 movimentos legais, e uma partida tem cerca de 150 jogadas. Isso significa que o espaco de busca do Go e centenas de ordens de magnitude maior que o do xadrez.

### Posicoes Dificeis de Avaliar

No xadrez, cada peca tem um valor claro (rainha 9 pontos, torre 5 pontos, etc.), permitindo avaliar posicoes com formulas simples. Mas no Go, o valor de uma pedra depende de sua relacao com as pedras ao redor, nao existindo metodo simples de avaliacao.

Um grupo esta vivo ou morto? Quanto vale uma area de influencia? Estas questoes, mesmo para especialistas humanos, frequentemente requerem calculos e julgamentos profundos.

### O Dilema dos Primeiros Programas de Go

Antes do AlphaGo, os programas de Go mais fortes tinham nivel de apenas 5-6 dan amador, muito distante dos jogadores profissionais. Estes programas usavam principalmente o metodo "Monte Carlo Tree Search" (MCTS), avaliando posicoes atraves de muitas simulacoes aleatorias.

Mas este metodo tinha limitacoes obvias: simulacoes aleatorias nao conseguiam capturar o pensamento estrategico do Go, e os programas frequentemente cometiam erros que pareciam muito tolos aos humanos.

## Duas Eras da IA no Go

### [Era AlphaGo (2015-2017)](/docs/evolution/ai-history/alphago-era)

Esta era comecou com a vitoria do AlphaGo sobre Fan Hui e terminou com a publicacao do artigo AlphaZero. Em apenas dois anos, a DeepMind realizou o salto de derrotar jogadores profissionais a superar os limites humanos.

Marcos principais:
- 2015.10: Derrota Fan Hui (primeira vitoria sobre jogador profissional)
- 2016.03: Derrota Lee Sedol (4:1)
- 2017.01: Master vence 60 partidas online consecutivas
- 2017.05: Derrota Ke Jie (3:0)
- 2017.10: AlphaZero e publicado

### [Era KataGo (2019-presente)](/docs/evolution/ai-history/katago-era)

Apos a aposentadoria do AlphaGo, a comunidade de codigo aberto assumiu a tocha. KataGo, Leela Zero e outras IAs de codigo aberto permitiram que todos usassem motores de Go de nivel superior, mudando completamente a forma de aprender e treinar Go.

Caracteristicas desta era:
- Democratizacao das ferramentas de IA
- Jogadores profissionais usando amplamente IA para treinar
- Estilo de jogo humano se tornando mais "IA-like"
- Elevacao geral do nivel do Go

## O Impacto Cognitivo da IA

### Redefinicao da "Jogada Correta"

Antes da IA, a humanidade estabeleceu atraves de milhares de anos uma teoria do Go considerada "correta". No entanto, muitas jogadas da IA contradizem o conhecimento tradicional humano:

- **San-san**: A nocao tradicional considerava o san-san direto na abertura uma "jogada vulgar", mas a IA frequentemente joga assim
- **Kata-tsuki**: O kata-tsuki antes considerado "mau movimento" foi provado pela IA ser a melhor escolha em certas posicoes
- **Ataque proximo**: A IA gosta de luta corpo a corpo, diferente da nocao tradicional humana de "comecar o ataque de longe"

### Limitacoes e Potencial Humano

O surgimento da IA fez os humanos reconhecerem suas limitacoes, mas ao mesmo tempo revelou o potencial humano.

Com a ajuda da IA, a velocidade de crescimento dos jovens jogadores acelerou muito. O nivel que antes levava dez anos para atingir, agora pode levar apenas tres a cinco anos. O nivel geral do Go esta aumentando.

### O Futuro do Go

Alguns temiam que a IA tornasse o Go sem sentido -- se nunca conseguiremos vencer a IA, por que jogar?

Mas os fatos provaram que essa preocupacao era desnecessaria. A IA nao acabou com o Go, mas abriu uma nova era. Partidas entre humanos ainda estao cheias de criatividade, emocao e imprevisibilidade -- estas sao as essencias que tornam o Go interessante.

---

A seguir, vamos entender em detalhes o desenvolvimento especifico destas duas eras.

- **[Era AlphaGo](/docs/evolution/ai-history/alphago-era)** - De derrotar jogadores profissionais a superar limites humanos
- **[Era KataGo](/docs/evolution/ai-history/katago-era)** - IA de codigo aberto e o novo ecossistema do Go

