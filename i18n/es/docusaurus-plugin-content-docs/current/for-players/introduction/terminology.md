---
sidebar_position: 2
title: Terminologia del Go
---

# Terminologia del Go

Aprender la terminologia del Go te ayudara a entender los comentarios de partidas, comprender las explicaciones y comunicarte con otros jugadores.

## Terminologia del tablero

### Posiciones importantes del tablero

| Termino | Posicion | Descripcion |
|------|------|------|
| **Tengen** | Centro exacto del tablero | Interseccion de la linea 10 en un tablero de 19x19 (10,10) |
| **Hoshi (estrella)** | Puntos negros cerca de las cuatro esquinas | Posiciones a 4 lineas de cada borde (4,4) |
| **Komoku** | Punto comun de apertura en la esquina | Posicion (3,4) o (4,3) |
| **San-san** | El punto mas profundo de la esquina | Posicion (3,3), jugada que toma la esquina directamente |
| **Mokuhazushi** | Posicion alta en la esquina | Posicion (3,5) o (5,3) |
| **Takamoku** | Posicion alta de la esquina | Posicion (4,5) o (5,4) |

### Regiones del tablero

| Termino | Descripcion |
|------|------|
| **Esquina** | Las cuatro regiones de esquina del tablero, generalmente dentro de la linea 4 |
| **Lado** | Las cuatro regiones laterales del tablero, entre las esquinas |
| **Centro** | La region central del tablero, tambien llamada "el vientre" |

### Altura de las posiciones

| Termino | Descripcion |
|------|------|
| **Primera linea** | La linea mas externa del borde (linea de muerte) |
| **Segunda linea** | La segunda linea (linea de derrota) |
| **Tercera linea** | La tercera linea (linea de territorio) |
| **Cuarta linea** | La cuarta linea (linea de influencia) |

:::note Proverbio
"En la primera linea buscas vida como pez en arbol, en la segunda linea la cabeza puede no tener ganancia" - La primera y segunda linea son posiciones desfavorables.
:::

---

## Terminologia de jugadas

### Acciones basicas

| Termino | Descripcion |
|------|------|
| **Colocar piedra** | Poner una piedra en el tablero |
| **Capturar** | Remover las piedras del oponente sin libertades |
| **Sacrificar** | Abandonar intencionalmente algunas piedras sin salvarlas |

### Relacionado con ataques

| Termino | Descripcion |
|------|------|
| **Atari** | Dejar las piedras del oponente con solo una libertad |
| **Doble atari** | Poner dos grupos en atari simultaneamente, el oponente solo puede salvar uno |
| **Escalera** | Atari continuo, metodo de captura donde el oponente no puede escapar |
| **Red** | Rodear al oponente holgadamente, si corre sera capturado |

### Formas de mover piedras

| Termino | Descripcion | Forma |
|------|------|------|
| **Extension** | Conectar en linea recta | Extension lineal |
| **Salto** | Saltar un espacio (en la misma linea) | Un espacio vacio en medio |
| **Vuelo** | Moverse diagonalmente, un espacio en medio | Similar al elefante en ajedrez |
| **Gran vuelo** | Moverse diagonalmente, distancia mayor | Mas lejos que el vuelo |
| **Diagonal** | Diagonalmente adyacente | Conexion diagonal |
| **Ikken-tobi** | Saltar un espacio en la misma linea (igual que salto) | Horizontal o vertical |

### Terminologia de contacto

| Termino | Descripcion |
|------|------|
| **Hane** | Jugar en la diagonal de la piedra del oponente bloqueando |
| **Corte** | Cortar la conexion de las piedras del oponente |
| **Conectar** | Conectar tus propias piedras |
| **Bloquear** | Bloquear el camino del oponente |
| **Pinza** | Rodear las piedras del oponente desde ambos lados |
| **Contacto** | Jugar pegado a la piedra del oponente |
| **Adjuntar** | Jugar muy cerca, usualmente junto a la piedra del oponente |
| **Presionar** | Presionar desde arriba, haciendo que el oponente quede bajo |
| **Socavar** | Socavar debajo de la piedra del oponente |
| **Invasion** | Jugar en una posicion clave (frecuentemente en el punto de ojo o vital) |
| **Sondear** | Jugar una piedra amenazando el punto debil del oponente |

---

## Terminologia de formas

### Buenas formas

| Termino | Descripcion |
|------|------|
| **Boca de tigre** | Dos piedras en diagonal, un punto vacio en medio, formando una boca de tigre |
| **Doble** | Forma basica despues de conectar dos piedras |
| **Solido** | Forma sin debilidades, muy estable |
| **Ligero** | Forma flexible, no teme ser atacada |

### Conceptos basicos

| Termino | Descripcion |
|------|------|
| **Ojo** | Punto vacio rodeado por tus propias piedras |
| **Libertad** | Puntos vacios alrededor de piedras o grupos (linea de vida) |
| **Punto de corte** | Punto debil en la conexion, donde puede ser cortado |
| **Conexion** | Piedras conectadas formando una unidad |

### Malas formas

| Termino | Descripcion |
|------|------|
| **Forma torpe** | Forma de baja eficiencia |
| **Forma pesada** | Piedras amontonadas, no eficientes |
| **Triangulo vacio** | Tres piedras formando un triangulo vacio, forma torpe tipica |
| **Forma dividida** | Forma separada, facil de atacar |

---

## Terminologia de partida

### Orden de turnos

| Termino | Descripcion |
|------|------|
| **Sente** | Despues de este movimiento, tu lado continua dirigiendo |
| **Gote** | Despues de este movimiento, necesitas dejar que el oponente mueva primero |
| **Tomar sente** | No terminar localmente, girar a otros puntos grandes |
| **Ganancia en sente** | Beneficio obtenido en sente |

### Fases de la partida

| Termino | Descripcion |
|------|------|
| **Fuseki (apertura)** | Fase de apertura, ambos lados ocupan puntos grandes |
| **Chuban (medio juego)** | Fase de combate mas intenso |
| **Yose (final)** | Fase final, arreglar los bordes |
| **Cierre** | Realizar el yose, cerrar bien los bordes |

### Evaluacion de posicion

| Termino | Descripcion |
|------|------|
| **Ventaja** | Posicion adelante |
| **Desventaja** | Posicion atras |
| **Partida cerrada** | Posicion con muy poca diferencia |
| **Vision global** | Jugada o juicio que afecta la situacion general |

### Resultado de la partida

| Termino | Descripcion |
|------|------|
| **Negro gana por X puntos** | Negro gana por X puntos |
| **Blanco gana por X puntos** | Blanco gana por X puntos |
| **Empate** | Mismo numero de puntos (muy raro) |
| **Victoria en medio juego** | Victoria sin contar puntos (oponente se rinde o abandona) |

---

## Terminologia avanzada

Estos terminos se encuentran frecuentemente en el aprendizaje avanzado:

| Termino | Descripcion |
|------|------|
| **Joseki** | Secuencias fijas de mejores respuestas en la apertura |
| **Tesuji** | Tecnica ingeniosa, un movimiento brillante |
| **Jugada comun** | Jugada ordinaria, no muy eficiente |
| **Jugada lenta** | Jugada no urgente, que da ventaja al oponente |
| **Punto urgente** | Posicion importante que debe jugarse inmediatamente |
| **Punto grande** | Posicion de gran valor |
| **Moyo (influencia)** | Influencia solida, con impacto hacia afuera |
| **Territorio** | Territorio definitivamente rodeado |
| **Marco** | Territorio aun no completado pero con potencial |

:::tip Consejo de aprendizaje
No necesitas memorizar todos los terminos de una vez. Se sugiere primero familiarizarte con los terminos basicos de "jugadas", el resto puedes aprenderlo gradualmente en la practica.
:::

