---
sidebar_position: 1
title: Para quienes desean profundizar
description: "Guía de temas avanzados: redes neuronales, MCTS, entrenamiento, optimización, despliegue"
---

# Para quienes desean profundizar

Este capítulo está diseñado para ingenieros que desean investigar a fondo la IA de Go, cubriendo implementación técnica, fundamentos teóricos y aplicaciones prácticas.

---

## Índice de artículos

### Tecnología central

| Artículo | Descripción |
|----------|-------------|
| [Arquitectura de redes neuronales en detalle](./neural-network) | Red residual de KataGo, características de entrada, diseño de salida múltiple |
| [Detalles de implementación de MCTS](./mcts-implementation) | Selección PUCT, pérdida virtual, evaluación por lotes, paralelización |
| [Análisis del mecanismo de entrenamiento de KataGo](./training) | Auto-juego, función de pérdida, ciclo de entrenamiento |

### Optimización de rendimiento

| Artículo | Descripción |
|----------|-------------|
| [Backend GPU y optimización](./gpu-optimization) | Comparación y ajuste de backends CUDA, OpenCL, Metal |
| [Cuantización y despliegue de modelos](./quantization-deploy) | FP16, INT8, TensorRT, despliegue en diversas plataformas |
| [Evaluación y benchmarking](./evaluation) | Puntuación Elo, pruebas de partidas, método estadístico SPRT |

### Temas avanzados

| Artículo | Descripción |
|----------|-------------|
| [Arquitectura de entrenamiento distribuido](./distributed-training) | Self-play Worker, recolección de datos, publicación de modelos |
| [Reglas personalizadas y variantes](./custom-rules) | Reglas chinas, japonesas, AGA, variantes de tamaño de tablero |
| [Guía de artículos clave](./papers) | Análisis de los puntos clave de AlphaGo, AlphaZero, KataGo |

### Código abierto e implementación

| Artículo | Descripción |
|----------|-------------|
| [Guía del código fuente de KataGo](./source-code) | Estructura de directorios, módulos principales, estilo de código |
| [Participar en la comunidad de código abierto](./contributing) | Formas de contribuir, entrenamiento distribuido, participación comunitaria |
| [Construir una IA de Go desde cero](./build-from-scratch) | Implementar paso a paso una versión simplificada de AlphaGo Zero |

---

## ¿Qué quieres hacer?

| Objetivo | Ruta recomendada |
|----------|------------------|
| Entender el diseño de redes neuronales | [Arquitectura de redes neuronales en detalle](./neural-network) → [Detalles de implementación de MCTS](./mcts-implementation) |
| Optimizar el rendimiento de ejecución | [Backend GPU y optimización](./gpu-optimization) → [Cuantización y despliegue de modelos](./quantization-deploy) |
| Investigar métodos de entrenamiento | [Análisis del mecanismo de entrenamiento de KataGo](./training) → [Arquitectura de entrenamiento distribuido](./distributed-training) |
| Entender los principios de los artículos | [Guía de artículos clave](./papers) → [Arquitectura de redes neuronales en detalle](./neural-network) |
| Programar hands-on | [Construir una IA de Go desde cero](./build-from-scratch) → [Guía del código fuente de KataGo](./source-code) |
| Participar en proyectos de código abierto | [Participar en la comunidad de código abierto](./contributing) → [Guía del código fuente de KataGo](./source-code) |

---

## Índice de conceptos avanzados

Al profundizar en la investigación, encontrarás los siguientes conceptos avanzados:

### Serie F: Escalado (8)

| Número | Concepto de Go | Correspondencia física/matemática |
|--------|----------------|-----------------------------------|
| F1 | Tamaño del tablero vs complejidad | Escalado de complejidad |
| F2 | Tamaño de red vs fuerza de juego | Escalado de capacidad |
| F3 | Tiempo de entrenamiento vs beneficio | Ley de rendimientos decrecientes |
| F4 | Cantidad de datos vs generalización | Complejidad de muestra |
| F5 | Escalado de recursos computacionales | Leyes de escalado |
| F6 | Leyes de escalado neuronal | Relación log-log |
| F7 | Entrenamiento con lotes grandes | Lote crítico |
| F8 | Eficiencia de parámetros | Límites de compresión |

### Serie G: Dimensiones (6)

| Número | Concepto de Go | Correspondencia física/matemática |
|--------|----------------|-----------------------------------|
| G1 | Representación de alta dimensión | Espacio vectorial |
| G2 | Maldición de la dimensionalidad | Dilema de alta dimensión |
| G3 | Hipótesis del manifold | Manifold de baja dimensión |
| G4 | Representación intermedia | Espacio latente |
| G5 | Desacoplamiento de características | Componentes independientes |
| G6 | Direcciones semánticas | Álgebra geométrica |

### Serie H: Aprendizaje por refuerzo (9)

| Número | Concepto de Go | Correspondencia física/matemática |
|--------|----------------|-----------------------------------|
| H1 | MDP | Cadena de Markov |
| H2 | Ecuación de Bellman | Programación dinámica |
| H3 | Iteración de valor | Teorema del punto fijo |
| H4 | Gradiente de política | Optimización estocástica |
| H5 | Replay de experiencia | Muestreo por importancia |
| H6 | Factor de descuento | Preferencia temporal |
| H7 | Aprendizaje TD | Estimación incremental |
| H8 | Función de ventaja | Reducción de varianza con línea base |
| H9 | Recorte PPO | Región de confianza |

### Serie K: Métodos de optimización (6)

| Número | Concepto de Go | Correspondencia física/matemática |
|--------|----------------|-----------------------------------|
| K1 | SGD | Aproximación estocástica |
| K2 | Momentum | Inercia |
| K3 | Adam | Tamaño de paso adaptativo |
| K4 | Decaimiento de tasa de aprendizaje | Recocido |
| K5 | Recorte de gradiente | Límite de saturación |
| K6 | Ruido SGD | Perturbación estocástica |

### Serie L: Generalización y estabilidad (5)

| Número | Concepto de Go | Correspondencia física/matemática |
|--------|----------------|-----------------------------------|
| L1 | Sobreajuste | Sobre-adaptación |
| L2 | Regularización | Optimización con restricciones |
| L3 | Dropout | Activación dispersa |
| L4 | Aumento de datos | Ruptura de simetría |
| L5 | Parada temprana | Parada óptima |

---

## Requisitos de hardware

### Lectura y aprendizaje

Sin requisitos especiales, cualquier computadora es suficiente.

### Entrenamiento de modelos

| Escala | Hardware recomendado | Tiempo de entrenamiento |
|--------|----------------------|-------------------------|
| Mini (b6c96) | GTX 1060 6GB | Varias horas |
| Pequeño (b10c128) | RTX 3060 12GB | 1-2 días |
| Mediano (b18c384) | RTX 4090 24GB | 1-2 semanas |
| Completo (b40c256) | Clúster multi-GPU | Varias semanas |

### Contribución al entrenamiento distribuido

- Cualquier computadora con GPU puede participar
- Se recomienda al menos GTX 1060 o equivalente
- Se requiere conexión de red estable

---

## Comenzar a leer

**Recomendamos empezar aquí:**

- ¿Quieres entender los principios? → [Arquitectura de redes neuronales en detalle](./neural-network)
- ¿Quieres implementar hands-on? → [Construir una IA de Go desde cero](./build-from-scratch)
- ¿Quieres leer artículos académicos? → [Guía de artículos clave](./papers)
