---
sidebar_position: 1
title: Regras do Go
---

# Regras do Go

As regras do Go sao muito simples, mas as variacoes que surgem delas sao infinitas. Esta e a magia do Go.

## Conceitos Basicos

### Tabuleiro e Pedras

- **Tabuleiro**: O tabuleiro padrao tem 19 linhas (19x19), iniciantes frequentemente usam 9 ou 13 linhas
- **Pedras**: Duas cores, preto e branco; preto joga primeiro
- **Posicao das jogadas**: Pedras sao colocadas nas intersecoes das linhas, nao dentro dos quadrados

### Objetivo do Jogo

O objetivo do Go e **cercar territorio**. Ao final do jogo, quem cercou mais espaco vazio vence.

---

## O Conceito de Liberdades

**Liberdades** e o conceito mais importante do Go. Liberdades sao as intersecoes vazias ao redor das pedras, a "linha de vida" das pedras.

### Liberdades de Uma Pedra

Uma pedra em diferentes posicoes tem diferentes quantidades de liberdades:

| Posicao | Numero de Liberdades |
|:---:|:---:|
| Centro | 4 liberdades |
| Lateral | 3 liberdades |
| Canto | 2 liberdades |

### Pedras Conectadas

Quando pedras da mesma cor sao adjacentes (conectadas acima, abaixo, esquerda ou direita), elas se tornam uma unidade e compartilham todas as liberdades.

:::info Conceito Importante
Pedras conectadas vivem ou morrem juntas. Se este grupo for capturado, todas as pedras conectadas serao removidas juntas.
:::

---

## Captura

Quando todas as liberdades de um grupo de pedras sao bloqueadas pelo oponente (liberdades = 0), este grupo e "capturado" e removido do tabuleiro.

### Passos da Captura

1. As pedras do oponente tem apenas uma liberdade restante
2. Voce joga e bloqueia a ultima liberdade
3. As pedras do oponente sao capturadas (removidas do tabuleiro)

### Atari (Xeque)

Quando um grupo de pedras tem apenas uma liberdade restante, este estado e chamado **atari** (ou xeque). Neste ponto, o oponente deve encontrar uma forma de escapar ou sacrificar as pedras.

---

## Pontos Proibidos

Algumas posicoes nao podem ser jogadas; estas sao chamadas **pontos proibidos**.

### Identificando Pontos Proibidos

Uma posicao e um ponto proibido se, simultaneamente:

1. Apos jogar la, suas proprias pedras ficam sem liberdades
2. E voce nao consegue capturar nenhuma pedra do oponente

:::tip Regra Simples
Se jogar uma pedra permite capturar pedras do oponente, nao e um ponto proibido.
:::

### Suicidio

Jogar uma pedra que deixa suas proprias pedras sem liberdades, e nao consegue capturar pedras do oponente, e chamado "suicidio". As regras do Go proibem suicidio.

---

## Regra do Ko

**Ko** e uma forma especial no Go que criaria uma situacao de loop infinito.

### O que e Ko

Quando ambos os lados podem capturar uma pedra mutuamente, e apos capturar o oponente poderia capturar imediatamente de volta, forma-se um ko.

### Regra do Ko

**Nao pode recapturar imediatamente**. Apos ser capturado no ko, voce deve jogar em outro lugar primeiro (chamado "encontrar uma ameaca de ko"), e so entao pode recapturar.

### Processo de Luta de Ko

1. Jogador A captura o ko
2. Jogador B nao pode recapturar imediatamente, deve jogar em outro lugar primeiro
3. Jogador A responde a esta jogada
4. Jogador B recaptura o ko
5. Isto se repete ate um lado desistir

:::note Por que esta regra existe
Sem a regra do ko, ambos os lados poderiam capturar infinitamente, e o jogo nunca terminaria.
:::

---

## Olhos e Grupos Vivos

**Olhos** sao um dos conceitos mais importantes no Go. Entender olhos significa entender vida e morte.

### O que e um Olho

Um olho e um ponto vazio completamente cercado por suas proprias pedras. O oponente nao pode jogar no ponto do olho (seria um ponto proibido).

### Condicao para Vida

**Para um grupo viver, deve ter dois ou mais olhos verdadeiros.**

Por que precisa de dois olhos?

- Se tem apenas um olho, o oponente pode gradualmente reduzir liberdades de fora
- Eventualmente este olho sera a ultima liberdade, e o oponente pode jogar la para capturar todo o grupo
- Com dois olhos, o oponente nao pode ocupar ambas as posicoes de olho simultaneamente, e o grupo nunca pode ser capturado

### Olhos Verdadeiros vs Falsos

- **Olho verdadeiro**: Um olho completo que o oponente nao pode destruir
- **Olho falso**: Parece um olho, mas tem defeitos e pode ser destruido pelo oponente

Distinguir olhos verdadeiros de falsos requer observar as posicoes diagonais do olho - este e conhecimento avancado de vida e morte.

---

## Determinacao do Vencedor

Quando o jogo termina, e necessario contar o territorio de ambos os lados para determinar o vencedor. Ha dois metodos principais de contagem.

### Contagem de Territorio (Regras Japonesas/Coreanas)

Conta a quantidade de pontos vazios cercados por cada lado.

**Metodo de calculo**:
- Pontos vazios (territorios) cercados pelo proprio lado
- Nao conta as pedras no tabuleiro

**Determinacao do vencedor**: Quem tem mais territorios vence.

### Contagem de Area (Regras Chinesas)

Conta o total de pedras e territorios de cada lado.

**Metodo de calculo**:
- Numero de pedras vivas no tabuleiro
- Mais os pontos vazios cercados

**Determinacao do vencedor**:
- O tabuleiro padrao tem 361 intersecoes
- Quem excede 180,5 pontos vence

### Komi (Compensacao)

Como preto joga primeiro e tem vantagem, branco recebe pontos de compensacao chamados "komi".

| Regra | Komi |
|:---:|:---:|
| Regras Chinesas | Preto da 3,75 pedras (7,5 pontos) |
| Regras Japonesas | Preto da 6,5 pontos |
| Regras Coreanas | Preto da 6,5 pontos |

:::tip Sugestao para Iniciantes
Quando esta comecando a aprender, nao precisa se preocupar muito com os detalhes de determinacao do vencedor. Primeiro entenda bem os conceitos de "liberdades" e "olhos" - esta e a base mais importante.
:::

---

## Fim do Jogo

### Quando Termina

Quando ambos os lados concordam que nao ha mais lugares para jogar, o jogo termina. Na pratica, e quando ambos passam consecutivamente.

### Processamento do Fim de Jogo

1. Confirmar todas as pedras mortas
2. Remover pedras mortas do tabuleiro
3. Contar o territorio de ambos os lados
4. Determinar o vencedor conforme as regras

### Resignacao

Durante o jogo, se um lado acredita que nao ha mais possibilidade de vitoria, pode resignar a qualquer momento. Resignar e uma forma comum e educada de terminar.

