---
sidebar_position: 11
title: Guia de articulos clave
description: Analisis de los puntos clave de los articulos hito de IA de Go como AlphaGo, AlphaZero y KataGo
---

# Guia de articulos clave

Este articulo resume los articulos mas importantes en la historia del desarrollo de IA de Go, proporcionando resumenes y puntos tecnicos clave para una comprension rapida.

---

## Vision general de articulos

### Linea temporal

```
2006  Coulom - MCTS aplicado por primera vez a Go
2016  Silver et al. - AlphaGo (Nature)
2017  Silver et al. - AlphaGo Zero (Nature)
2017  Silver et al. - AlphaZero
2019  Wu - KataGo
2020+ Diversas mejoras y aplicaciones
```

### Recomendaciones de lectura

| Objetivo | Articulo recomendado |
|----------|---------------------|
| Entender lo basico | AlphaGo (2016) |
| Entender el auto-juego | AlphaGo Zero (2017) |
| Entender el metodo general | AlphaZero (2017) |
| Referencia de implementacion | KataGo (2019) |

---

## 1. El nacimiento de MCTS (2006)

### Informacion del articulo

```
Titulo: Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search
Autor: Remi Coulom
Publicacion: Computers and Games 2006
```

### Contribucion principal

Primera aplicacion sistematica de metodos Monte Carlo a Go:

```
Antes: Simulacion puramente aleatoria, sin estructura de arbol
Despues: Construir arbol de busqueda + seleccion UCB + retropropagacion de estadisticas
```

### Conceptos clave

#### Formula UCB1

```
Puntuacion de seleccion = Tasa de victoria promedio + C * sqrt(ln(N) / n)

Donde:
- N: Numero de visitas del nodo padre
- n: Numero de visitas del nodo hijo
- C: Constante de exploracion
```

#### Cuatro pasos de MCTS

```
1. Selection: Seleccionar nodo usando UCB
2. Expansion: Expandir nuevo nodo
3. Simulation: Simular aleatoriamente hasta el final
4. Backpropagation: Retropropagar victoria/derrota
```

### Impacto

- Llevo la IA de Go al nivel de dan amateur
- Se convirtio en la base de todas las IAs de Go posteriores
- El concepto UCB influyo en el desarrollo de PUCT

---

## 2. AlphaGo (2016)

### Informacion del articulo

```
Titulo: Mastering the game of Go with deep neural networks and tree search
Autores: Silver, D., Huang, A., Maddison, C.J., et al.
Publicacion: Nature, 2016
DOI: 10.1038/nature16961
```

### Contribucion principal

**Primera combinacion de deep learning y MCTS**, derrotando al campeon mundial humano.

### Arquitectura del sistema

```
+---------------------------------------------+
|              Arquitectura AlphaGo           |
+---------------------------------------------+
|                                             |
|   Policy Network (SL)                       |
|   +-- Entrada: Estado del tablero           |
|   |   (48 planos de caracteristicas)        |
|   +-- Arquitectura: CNN de 13 capas         |
|   +-- Salida: Probabilidad de 361 posiciones|
|   +-- Entrenamiento: 30 millones de         |
|       partidas humanas                      |
|                                             |
|   Policy Network (RL)                       |
|   +-- Inicializado desde SL Policy          |
|   +-- Aprendizaje por refuerzo con          |
|       auto-juego                            |
|                                             |
|   Value Network                             |
|   +-- Entrada: Estado del tablero           |
|   +-- Salida: Valor de tasa de victoria     |
|       unico                                 |
|   +-- Entrenamiento: Posiciones generadas   |
|       por auto-juego                        |
|                                             |
|   MCTS                                      |
|   +-- Usa Policy Network para guiar         |
|       busqueda                              |
|   +-- Usa Value Network + Rollout para      |
|       evaluacion                            |
|                                             |
+---------------------------------------------+
```

### Puntos tecnicos

#### 1. Policy Network con aprendizaje supervisado

```python
# Caracteristicas de entrada (48 planos)
- Posicion de piedras propias
- Posicion de piedras del oponente
- Numero de libertades
- Estado despues de captura
- Posiciones de movimientos legales
- Posiciones de los ultimos movimientos
...
```

#### 2. Mejora con aprendizaje por refuerzo

```
SL Policy -> Auto-juego -> RL Policy

RL Policy tiene ~80% tasa de victoria contra SL Policy
```

#### 3. Entrenamiento de Value Network

```
Clave para prevenir sobreajuste:
- Tomar solo una posicion de cada partida
- Evitar posiciones similares repetidas
```

#### 4. Integracion con MCTS

```
Evaluacion de nodo hoja = 0.5 * Value Network + 0.5 * Rollout

Rollout usa Policy Network rapido (menor precision pero mas velocidad)
```

### Datos clave

| Item | Valor |
|------|-------|
| Precision SL Policy | 57% |
| Tasa de victoria RL Policy vs SL Policy | 80% |
| GPUs de entrenamiento | 176 |
| GPUs de partida | 48 TPU |

---

## 3. AlphaGo Zero (2017)

### Informacion del articulo

```
Titulo: Mastering the game of Go without human knowledge
Autores: Silver, D., Schrittwieser, J., Simonyan, K., et al.
Publicacion: Nature, 2017
DOI: 10.1038/nature24270
```

### Contribucion principal

**Sin necesidad de partidas humanas**, aprendizaje desde cero.

### Diferencias con AlphaGo

| Aspecto | AlphaGo | AlphaGo Zero |
|---------|---------|--------------|
| Partidas humanas | Necesarias | **No necesarias** |
| Numero de redes | 4 | **1 de doble cabeza** |
| Planos de entrada | 48 | **17** |
| Rollout | Usado | **No usado** |
| Red residual | No | **Si** |
| Tiempo de entrenamiento | Meses | **3 dias** |

### Innovaciones clave

#### 1. Red unica de doble cabeza

```
              Entrada (17 planos)
                   |
              +----+----+
              | Torre   |
              | residual|
              | (19 o   |
              |  39     |
              | capas)  |
              +----+----+
           +-------+-------+
           |               |
        Policy          Value
        (361)            (1)
```

#### 2. Caracteristicas de entrada simplificadas

```python
# Solo 17 planos de caracteristicas
features = [
    current_player_stones,      # Piedras propias
    opponent_stones,            # Piedras del oponente
    history_1_player,           # Estado historico 1
    history_1_opponent,
    ...                         # Estados historicos 2-7
    color_to_play               # Turno
]
```

#### 3. Evaluacion solo con Value Network

```
Ya no usa Rollout
Evaluacion de nodo hoja = Salida de Value Network

Mas simple, mas rapido
```

#### 4. Proceso de entrenamiento

```
Inicializar red aleatoria
    |
    v
+-----------------------------+
|  Auto-juego genera partidas | <--+
+-------------+---------------+    |
              |                    |
              v                    |
+-----------------------------+    |
|  Entrenar red neuronal      |    |
|  - Policy: Minimizar        |    |
|    entropia cruzada         |    |
|  - Value: Minimizar MSE     |    |
+-------------+---------------+    |
              |                    |
              v                    |
+-----------------------------+    |
|  Evaluar nueva red          |    |
|  Si es mejor, reemplazar    |----+
+-----------------------------+
```

### Curva de aprendizaje

```
Tiempo de entrenamiento    Elo
-------------------------
3 horas      Principiante
24 horas     Supera a AlphaGo Lee
72 horas     Supera a AlphaGo Master
```

---

## 4. AlphaZero (2017)

### Informacion del articulo

```
Titulo: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
Autores: Silver, D., Hubert, T., Schrittwieser, J., et al.
Publicacion: arXiv:1712.01815 (posteriormente en Science, 2018)
```

### Contribucion principal

**Generalizacion**: El mismo algoritmo aplicado a Go, ajedrez y shogi.

### Arquitectura general

```
Codificacion de entrada (especifica del juego) -> Red residual (general) -> Salida de doble cabeza (general)
```

### Adaptacion entre juegos

| Juego | Planos de entrada | Espacio de acciones | Tiempo de entrenamiento |
|-------|-------------------|---------------------|-------------------------|
| Go | 17 | 362 | 40 dias |
| Ajedrez | 119 | 4672 | 9 horas |
| Shogi | 362 | 11259 | 12 horas |

### Mejoras de MCTS

#### Formula PUCT

```
Puntuacion de seleccion = Q(s,a) + c(s) * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

c(s) = log((1 + N(s) + c_base) / c_base) + c_init
```

#### Ruido de exploracion

```python
# Agregar ruido de Dirichlet en el nodo raiz
P(s,a) = (1 - epsilon) * p_a + epsilon * eta_a

eta ~ Dir(alpha)
alpha = 0.03 (Go), 0.3 (ajedrez), 0.15 (shogi)
```

---

## 5. KataGo (2019)

### Informacion del articulo

```
Titulo: Accelerating Self-Play Learning in Go
Autor: David J. Wu
Publicacion: arXiv:1902.10565
```

### Contribucion principal

**50x de mejora en eficiencia**, permitiendo a desarrolladores individuales entrenar una potente IA de Go.

### Innovaciones clave

#### 1. Objetivos de entrenamiento auxiliares

```
Perdida total = Policy Loss + Value Loss +
         Score Loss + Ownership Loss + ...

Los objetivos auxiliares hacen que la red converja mas rapido
```

#### 2. Caracteristicas globales

```python
# Capa de pooling global
global_features = global_avg_pool(conv_features)
# Combinar con caracteristicas locales
combined = concat(conv_features, broadcast(global_features))
```

#### 3. Aleatorizacion de Playout Cap

```
Tradicional: Buscar N veces fijas
KataGo: N muestreado aleatoriamente de una distribucion

Hace que la red aprenda a funcionar bien en varias profundidades de busqueda
```

#### 4. Tamano de tablero progresivo

```python
if training_step < 1000000:
    board_size = random.choice([9, 13, 19])
else:
    board_size = 19
```

### Comparacion de eficiencia

| Metrica | AlphaZero | KataGo |
|---------|-----------|--------|
| Dias-GPU para nivel superhumano | 5000 | **100** |
| Mejora de eficiencia | Base | **50x** |

---

## 6. Articulos de extension

### MuZero (2020)

```
Titulo: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model
Contribucion: Aprende el modelo de dinamica del entorno, no necesita reglas del juego
```

### EfficientZero (2021)

```
Titulo: Mastering Atari Games with Limited Data
Contribucion: Gran mejora en eficiencia de muestras
```

### Gumbel AlphaZero (2022)

```
Titulo: Policy Improvement by Planning with Gumbel
Contribucion: Metodo mejorado de mejora de politica
```

---

## Recomendaciones de lectura de articulos

### Orden para principiantes

```
1. AlphaGo (2016) - Entender arquitectura basica
2. AlphaGo Zero (2017) - Entender auto-juego
3. KataGo (2019) - Entender detalles de implementacion
```

### Orden avanzado

```
4. AlphaZero (2017) - Generalizacion
5. MuZero (2020) - Aprender modelo del mundo
6. Articulo original de MCTS - Entender fundamentos
```

### Tecnicas de lectura

1. **Ver primero resumen y conclusiones**: Captar rapidamente la contribucion principal
2. **Ver las figuras**: Entender la arquitectura general
3. **Ver seccion de metodos**: Entender detalles tecnicos
4. **Ver apendices**: Encontrar detalles de implementacion e hiperparametros

---

## Enlaces de recursos

### PDFs de articulos

| Articulo | Enlace |
|----------|--------|
| AlphaGo | [Nature](https://www.nature.com/articles/nature16961) |
| AlphaGo Zero | [Nature](https://www.nature.com/articles/nature24270) |
| AlphaZero | [Science](https://www.science.org/doi/10.1126/science.aar6404) |
| KataGo | [arXiv](https://arxiv.org/abs/1902.10565) |

### Implementaciones de codigo abierto

| Proyecto | Enlace |
|----------|--------|
| KataGo | [GitHub](https://github.com/lightvector/KataGo) |
| Leela Zero | [GitHub](https://github.com/leela-zero/leela-zero) |
| MiniGo | [GitHub](https://github.com/tensorflow/minigo) |

---

## Lectura adicional

- [Arquitectura de redes neuronales en detalle](../neural-network) — Entender a fondo el diseno de redes
- [Detalles de implementacion de MCTS](../mcts-implementation) — Implementacion del algoritmo de busqueda
- [Analisis del mecanismo de entrenamiento de KataGo](../training) — Detalles del proceso de entrenamiento
