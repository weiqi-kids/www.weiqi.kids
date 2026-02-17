---
sidebar_position: 1
title: Conocimientos previos
---

# Vista general de conocimientos previos

Antes de profundizar en la practica de KataGo, es muy importante entender la historia del desarrollo y la tecnologia central de la IA de Go. Este capitulo te llevara a conocer la evolucion tecnica desde AlphaGo hasta la IA de Go moderna.

## Por que necesitas conocer los antecedentes?

El desarrollo de la IA de Go es uno de los avances mas emocionantes en el campo de la inteligencia artificial. El partido de 2016 donde AlphaGo derroto a Lee Sedol no solo fue un hito en la historia del Go, sino que tambien marco el gran exito de la combinacion del aprendizaje profundo y el aprendizaje por refuerzo.

Entender estos conocimientos previos puede ayudarte a:

- **Tomar mejores decisiones tecnicas**: Entender las ventajas y desventajas de varios metodos, elegir la solucion adecuada para tu proyecto
- **Depurar mas efectivamente**: Entender los principios subyacentes, mas facil diagnosticar problemas
- **Mantenerte al dia con los desarrollos mas recientes**: Dominar los conocimientos basicos, mas facil entender nuevos papers y tecnologias
- **Contribuir a proyectos de codigo abierto**: Participar en el desarrollo de proyectos como KataGo requiere entender profundamente su filosofia de diseno

## Contenido de este capitulo

### [Analisis del paper de AlphaGo](./alphago.md)

Analisis profundo del paper clasico de DeepMind, incluyendo:

- Significado historico e impacto de AlphaGo
- Diseno de Policy Network y Value Network
- Principios e implementacion de Busqueda de Arbol Monte Carlo (MCTS)
- Innovacion del metodo de entrenamiento Self-play
- Evolucion de AlphaGo a AlphaGo Zero a AlphaZero

### [Analisis del paper de KataGo](./katago-paper.md)

Conocer las innovaciones tecnicas de la IA de Go de codigo abierto mas fuerte actualmente:

- Mejoras de KataGo respecto a AlphaGo
- Metodos de entrenamiento mas eficientes y utilizacion de recursos
- Implementacion tecnica del soporte para multiples reglas de Go
- Diseno de prediccion simultanea de tasa de victoria y puntos
- Por que KataGo puede lograr fuerza mas alta con menos recursos

### [Introduccion a otras IAs de Go](./zen.md)

Conocer completamente el ecosistema de IA de Go:

- IAs comerciales: Zen (Tencent), Fine Art (Tencent), Golaxy
- IAs de codigo abierto: Leela Zero, ELF OpenGo, SAI
- Comparacion de caracteristicas tecnicas y escenarios de aplicacion de cada IA

## Linea de tiempo del desarrollo tecnico

| Tiempo | Evento | Importancia |
|------|------|--------|
| Octubre 2015 | AlphaGo derrota a Fan Hui | Primera IA en derrotar a un jugador profesional |
| Marzo 2016 | AlphaGo derrota a Lee Sedol | El partido humano-maquina que conmociono al mundo |
| Mayo 2017 | AlphaGo derrota a Ke Jie | Confirmo que la IA supera el nivel humano mas alto |
| Octubre 2017 | Publicacion de AlphaGo Zero | Self-play puro, sin partidas humanas |
| Diciembre 2017 | Publicacion de AlphaZero | Diseno generalizado, conquistando simultaneamente Go, ajedrez, shogi |
| 2018 | Leela Zero alcanza nivel sobrehumano | Victoria de la comunidad de codigo abierto |
| 2019 | Publicacion de KataGo | Metodos de entrenamiento mas eficientes |
| 2020-presente | KataGo mejora continuamente | Se convierte en la IA de Go de codigo abierto mas fuerte |

## Vista previa de conceptos centrales

Antes de leer los capitulos detallados, aqui hay una breve introduccion a algunos conceptos centrales:

### Rol de las redes neuronales en Go

```
Estado del tablero -> Red neuronal -> { Policy (probabilidad de jugadas), Value (evaluacion de tasa de victoria) }
```

La red neuronal recibe el estado actual del tablero como entrada y produce dos tipos de informacion:
- **Policy**: Probabilidad de jugar en cada posicion, guia la direccion de busqueda
- **Value**: Estimacion de tasa de victoria de la posicion actual, usado para evaluar posiciones

### Busqueda de Arbol Monte Carlo (MCTS)

MCTS es un algoritmo de busqueda que combina redes neuronales para decidir la mejor jugada:

1. **Seleccion**: Desde el nodo raiz, seleccionar el camino mas prometedor
2. **Expansion**: Expandir nuevas jugadas posibles en nodos hoja
3. **Evaluacion**: Usar red neuronal para evaluar el valor de la posicion
4. **Retropropagacion**: Propagar el resultado de la evaluacion hacia atras para actualizar todos los nodos en el camino

### Self-play (Juego contra si mismo)

La IA juega contra si misma para generar datos de entrenamiento:

```
Modelo inicial -> Self-play -> Recoger partidas -> Entrenar nuevo modelo -> Modelo mas fuerte -> Repetir
```

Este ciclo permite que la IA mejore continuamente sin depender de partidas humanas.

## Orden de lectura sugerido

1. **Lee primero el analisis del paper de AlphaGo**: Establecer el marco teorico basico
2. **Luego lee el analisis del paper de KataGo**: Conocer las mejoras y optimizaciones mas recientes
3. **Finalmente lee la introduccion a otras IAs de Go**: Ampliar la vision, conocer diferentes implementaciones

Estas listo? Comencemos con [Analisis del paper de AlphaGo](./alphago.md).

