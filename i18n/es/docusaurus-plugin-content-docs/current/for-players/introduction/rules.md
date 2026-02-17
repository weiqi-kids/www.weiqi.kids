---
sidebar_position: 1
title: Reglas del Go
---

# Reglas del Go

Las reglas del Go son muy simples, pero las variaciones que generan son infinitas. Esto es precisamente lo que hace fascinante al Go.

## Conceptos basicos

### Tablero y piedras

- **Tablero**: El tablero estandar es de 19 lineas (19x19), los principiantes suelen usar tableros de 9 o 13 lineas
- **Piedras**: Dos colores, negro y blanco, las negras juegan primero
- **Posicion de las piedras**: Las piedras se colocan en las intersecciones de las lineas, no dentro de los cuadrados

### Objetivo del juego

El objetivo del Go es **rodear territorio**. Al final de la partida, el lado que haya rodeado mas territorio gana.

---

## El concepto de libertades

**Libertades** es el concepto mas importante en Go. Las libertades son los puntos vacios adyacentes alrededor de una piedra, son la "linea de vida" de las piedras.

### Libertades de una sola piedra

Una piedra en diferentes posiciones tiene diferente numero de libertades:

| Posicion | Numero de libertades |
|:---:|:---:|
| Centro | 4 libertades |
| Lado | 3 libertades |
| Esquina | 2 libertades |

### Piedras conectadas

Cuando piedras del mismo color estan adyacentes (conectadas arriba, abajo, izquierda o derecha), se convierten en una unidad y comparten todas sus libertades.

:::info Concepto importante
Las piedras conectadas viven y mueren juntas. Si este grupo es capturado, todas las piedras conectadas seran removidas.
:::

---

## Capturas

Cuando todas las libertades de un grupo son bloqueadas por el oponente (libertades = 0), ese grupo es "capturado" y removido del tablero.

### Pasos de la captura

1. Las piedras del oponente solo tienen una libertad
2. Juegas para bloquear la ultima libertad
3. Las piedras del oponente son capturadas (removidas del tablero)

### Atari

Cuando un grupo solo tiene una libertad, esta situacion se llama **atari**. En este momento el oponente debe intentar escapar o abandonar las piedras.

---

## Puntos prohibidos (suicidio)

Algunas posiciones no se pueden jugar, estas se llaman **puntos prohibidos** o **puntos de suicidio**.

### Identificar puntos prohibidos

Una posicion es un punto prohibido si cumple simultaneamente:

1. Despues de jugar ahi, tu propia piedra no tiene libertades
2. Y no puedes capturar ninguna piedra del oponente

:::tip Regla simple
Si jugar ahi te permite capturar piedras del oponente, no es un punto prohibido.
:::

### Suicidio

Jugar un movimiento que deja tu propia piedra sin libertades, y no puede capturar piedras del oponente, se llama "suicidio". Las reglas del Go prohiben el suicidio.

---

## Regla del Ko

**Ko** es una forma especial en Go que causa una situacion de ciclo infinito.

### Que es Ko

Cuando ambos lados pueden capturarse mutuamente una piedra, y despues de capturar el oponente puede inmediatamente recapturar, se forma un ko.

### Regla del Ko

**No se puede recapturar inmediatamente**. Despues de que te capturen en el ko, debes jugar en otro lugar primero (llamado "buscar amenaza de ko"), y solo entonces puedes recapturar.

### Proceso del Ko

1. El jugador A captura en el ko
2. El jugador B no puede recapturar inmediatamente, debe jugar en otro lugar primero
3. El jugador A responde a ese movimiento
4. El jugador B recaptura el ko
5. Esto continua hasta que un lado abandona

:::note Por que existe esta regla
Si no existiera la regla del ko, ambos lados podrian capturarse infinitamente y la partida nunca terminaria.
:::

---

## Ojos y grupos vivos

**Ojos** es uno de los conceptos mas importantes en Go. Entender los ojos es entender la vida y muerte.

### Que es un ojo

Un ojo es un punto vacio completamente rodeado por tus propias piedras. El oponente no puede jugar en la posicion del ojo (seria un punto prohibido).

### Condicion para vivir

**Un grupo necesita dos o mas ojos verdaderos para vivir.**

Por que se necesitan dos ojos?

- Si solo hay un ojo, el oponente puede reducir libertades desde afuera gradualmente
- Finalmente ese ojo es la ultima libertad, el oponente puede jugar ahi y capturar todo el grupo
- Con dos ojos, el oponente no puede ocupar ambas posiciones de ojo simultaneamente, asi que el grupo nunca puede ser capturado

### Ojos verdaderos y falsos

- **Ojo verdadero**: Un ojo completo que el oponente no puede destruir
- **Ojo falso**: Parece un ojo pero tiene defectos, puede ser destruido por el oponente

Distinguir ojos verdaderos de falsos requiere mirar las posiciones diagonales del ojo, esto es conocimiento avanzado de vida y muerte.

---

## Determinacion del ganador

Al final de la partida, se necesita calcular el territorio de ambos lados para determinar el ganador. Hay dos metodos principales de calculo.

### Conteo por territorio (reglas japonesas/coreanas)

Cuenta el numero de puntos vacios que cada lado ha rodeado.

**Metodo de calculo**:
- Puntos vacios rodeados por tu lado (territorio)
- No se cuentan las piedras en el tablero

**Determinacion del ganador**: El lado con mas territorio gana.

### Conteo por area (reglas chinas)

Cuenta el total de piedras y territorio de tu lado.

**Metodo de calculo**:
- Numero de piedras vivas de tu lado en el tablero (piedras)
- Mas los puntos vacios rodeados por tu lado (territorio)

**Determinacion del ganador**:
- El tablero estandar tiene 361 intersecciones
- El lado que exceda 180.5 puntos gana

### Komi (compensacion)

Como las negras juegan primero y tienen ventaja, las blancas reciben puntos de compensacion, llamados "komi".

| Reglas | Komi |
|:---:|:---:|
| Reglas chinas | Negro da 3.75 piedras (7.5 puntos) |
| Reglas japonesas | Negro da 6.5 puntos |
| Reglas coreanas | Negro da 6.5 puntos |

:::tip Consejo para principiantes
Cuando recien empiezas a aprender, no te preocupes demasiado por los detalles de determinacion del ganador. Primero entiende los conceptos de "libertades" y "ojos", esa es la base mas importante.
:::

---

## Fin de la partida

### Cuando termina

Cuando ambos lados consideran que no hay lugares donde jugar, la partida termina. En realidad, es cuando ambos lados pasan consecutivamente.

### Proceso de finalizacion

1. Confirmar todas las piedras muertas
2. Remover las piedras muertas del tablero
3. Calcular el territorio de ambos lados
4. Determinar el ganador segun las reglas

### Rendirse

Durante la partida, si un lado considera que ya no hay posibilidad de ganar, puede rendirse en cualquier momento. Rendirse es una forma comun y cortés de terminar.

