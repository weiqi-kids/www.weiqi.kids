---
sidebar_position: 2
title: Historia del desarrollo del Go con IA
---

# Historia del desarrollo del Go con IA

Durante mucho tiempo, el Go fue considerado el juego mas dificil de conquistar para la inteligencia artificial. Con 19x19 = 361 intersecciones en el tablero, cada punto puede tener una piedra colocada, el numero de variaciones supera el numero total de atomos en el universo (aproximadamente 10^170 posibles partidas). Los metodos tradicionales de busqueda exhaustiva fallaban completamente frente al Go.

Sin embargo, entre 2015 y 2017, la serie de programas AlphaGo de DeepMind cambio todo esto por completo. Esta revolucion no solo afecto al Go, sino que impulso el desarrollo de todo el campo de la inteligencia artificial.

## Por que el Go es tan dificil?

### Enorme espacio de busqueda

Tomando el ajedrez como ejemplo, hay aproximadamente 35 movimientos legales promedio por turno, y una partida dura aproximadamente 80 turnos. El Go tiene aproximadamente 250 movimientos legales promedio por turno, y una partida dura aproximadamente 150 turnos. Esto significa que el espacio de busqueda del Go es cientos de ordenes de magnitud mas grande que el del ajedrez.

### Posiciones dificiles de evaluar

Cada pieza de ajedrez tiene un valor claro (reina 9 puntos, torre 5 puntos, etc.), las posiciones se pueden evaluar con formulas simples. Pero en Go, el valor de una piedra depende de su relacion con las piedras circundantes, no hay metodo de evaluacion simple.

Un grupo esta vivo o muerto? Cuantos puntos vale una zona de influencia? Estas preguntas incluso para expertos humanos frecuentemente requieren calculo profundo y juicio.

### El dilema de los programas de Go tempranos

Antes de AlphaGo, los programas de Go mas fuertes solo tenian nivel de 5-6 dan amateur, muy lejos de los jugadores profesionales. Estos programas principalmente usaban el metodo de "Busqueda de Arbol Monte Carlo" (MCTS), evaluando posiciones a traves de grandes cantidades de simulaciones aleatorias.

Pero este metodo tenia limitaciones obvias: las simulaciones aleatorias no podian capturar el pensamiento estrategico del Go, los programas frecuentemente cometian errores que a los humanos les parecian muy estupidos.

## Dos eras del Go con IA

### [Era AlphaGo (2015-2017)](/docs/learn/history/ai-history/alphago-era)

Esta era comienza con AlphaGo derrotando a Fan Hui, y termina con la publicacion del paper de AlphaZero. DeepMind logro en solo dos anos el salto de derrotar a jugadores profesionales a superar los limites humanos.

Hitos clave:
- 2015.10: Derrota a Fan Hui (primera vez que se derrota a un profesional)
- 2016.03: Derrota a Lee Sedol (4:1)
- 2017.01: Master 60 victorias consecutivas en linea
- 2017.05: Derrota a Ke Jie (3:0)
- 2017.10: Publicacion de AlphaZero

### [Era KataGo (2019-presente)](/docs/learn/history/ai-history/katago-era)

Despues del retiro de AlphaGo, la comunidad de codigo abierto tomo la antorcha. IAs de codigo abierto como KataGo y Leela Zero permitieron que todos pudieran usar motores de Go de primer nivel, cambiando completamente la forma de aprender y entrenar Go.

Caracteristicas de esta era:
- Democratizacion de herramientas de IA
- Uso generalizado de IA por jugadores profesionales para entrenar
- Estilo de Go humano influenciado por IA
- Mejora general del nivel de Go

## Impacto cognitivo de la IA

### Redefinicion de "jugada correcta"

Antes de que apareciera la IA, los humanos establecieron a traves de miles de anos de acumulacion un conjunto de teoria del Go considerada "correcta". Sin embargo, muchas jugadas de la IA contradicen el conocimiento tradicional humano:

- **Invadir san-san**: El concepto tradicional consideraba invadir san-san directamente en la apertura como "vulgar", pero la IA lo hace frecuentemente
- **Aproximacion por el hombro**: Las aproximaciones por el hombro consideradas "malas jugadas" en el pasado fueron demostradas por la IA como la mejor opcion en ciertas posiciones
- **Ataque cercano**: A la IA le gusta el combate cuerpo a cuerpo, diferente del concepto tradicional humano de "el ataque comienza desde lejos"

### Limitaciones y potencial humano

La aparicion de la IA hizo que los humanos reconocieran sus limitaciones, pero tambien mostro el potencial humano.

Con la ayuda de la IA, la velocidad de crecimiento de los jovenes jugadores se acelero enormemente. Niveles que antes requerian diez anos para alcanzar, ahora pueden lograrse en tres a cinco anos. El nivel general del Go esta mejorando.

### El futuro del Go

Algunas personas se preocupan de que la IA haga que el Go pierda significado - si nunca puedes ganarle a la IA, por que seguir jugando?

Pero los hechos demuestran que esta preocupacion es innecesaria. La IA no termino el Go, sino que abrio una nueva era para el Go. Las partidas entre humanos siguen llenas de creatividad, emocion e imprevisibilidad - estas son precisamente las esencias que hacen interesante al Go.

---

A continuacion, conozcamos en detalle el desarrollo especifico de estas dos eras.

- **[Era AlphaGo](/docs/learn/history/ai-history/alphago-era)** - De derrotar profesionales a superar los limites humanos
- **[Era KataGo](/docs/learn/history/ai-history/katago-era)** - IA de codigo abierto y el nuevo ecosistema del Go

