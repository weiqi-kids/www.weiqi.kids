---
sidebar_position: 10
title: Reglas personalizadas y variantes
description: Conjuntos de reglas de Go soportados por KataGo, variantes de tablero y configuración personalizada detallada
---

# Reglas personalizadas y variantes

Este artículo presenta las diversas reglas de Go soportadas por KataGo, variantes de tamaño de tablero y cómo personalizar la configuración de reglas.

---

## Visión general de conjuntos de reglas

### Comparación de reglas principales

| Conjunto de reglas | Método de conteo | Komi | Suicidio | Recaptura de ko |
|-------------------|------------------|------|----------|-----------------|
| **Chinese** | Conteo de área | 7.5 | Prohibido | Prohibida |
| **Japanese** | Conteo de territorio | 6.5 | Prohibido | Prohibida |
| **Korean** | Conteo de territorio | 6.5 | Prohibido | Prohibida |
| **AGA** | Mixto | 7.5 | Prohibido | Prohibida |
| **New Zealand** | Conteo de área | 7 | Permitido | Prohibida |
| **Tromp-Taylor** | Conteo de área | 7.5 | Permitido | Prohibida |

### Configuración KataGo

```ini
# config.cfg
rules = chinese           # Conjunto de reglas
komi = 7.5               # Komi
boardXSize = 19          # Ancho del tablero
boardYSize = 19          # Alto del tablero
```

---

## Reglas chinas (Chinese)

### Características

```
Método de conteo: Conteo de área (piedras + territorio)
Komi: 7.5 puntos
Suicidio: Prohibido
Recaptura de ko: Prohibida (regla simplificada)
```

### Explicación del conteo de área

```
Puntuación final = Piedras propias + Puntos vacíos propios

Ejemplo:
Negro 120 piedras + Negro 65 puntos de territorio = 185 puntos
Blanco 100 piedras + Blanco 75 puntos de territorio + Komi 7.5 = 182.5 puntos
Negro gana por 2.5 puntos
```

### Configuración KataGo

```ini
rules = chinese
komi = 7.5
```

---

## Reglas japonesas (Japanese)

### Características

```
Método de conteo: Conteo de territorio (solo puntos vacíos)
Komi: 6.5 puntos
Suicidio: Prohibido
Recaptura de ko: Prohibida
Requiere marcar piedras muertas
```

### Explicación del conteo de territorio

```
Puntuación final = Puntos vacíos propios + Piedras capturadas del oponente

Ejemplo:
Negro 65 puntos de territorio + 10 capturas = 75 puntos
Blanco 75 puntos de territorio + 5 capturas + Komi 6.5 = 86.5 puntos
Blanco gana por 11.5 puntos
```

### Determinación de piedras muertas

Las reglas japonesas requieren que ambos jugadores acuerden qué piedras están muertas:

```python
def is_dead_by_japanese_rules(group, game_state):
    """Determinar piedras muertas según reglas japonesas"""
    # Necesita probar que la cadena no puede hacer dos ojos
    # Esta es la complejidad de las reglas japonesas
    pass
```

### Configuración KataGo

```ini
rules = japanese
komi = 6.5
```

---

## Reglas AGA

### Características

Las reglas de la American Go Association (AGA) combinan las ventajas de las reglas chinas y japonesas:

```
Método de conteo: Mixto (área o territorio, mismo resultado)
Komi: 7.5 puntos
Suicidio: Prohibido
Blanco necesita dar una piedra para pasar
```

### Regla de pasar

```
Negro pasa: No necesita dar piedra
Blanco pasa: Necesita entregar una piedra a Negro

Esto hace que el conteo de área y territorio den el mismo resultado
```

### Configuración KataGo

```ini
rules = aga
komi = 7.5
```

---

## Reglas Tromp-Taylor

### Características

Las reglas de Go más simples, ideales para implementación en programas:

```
Método de conteo: Conteo de área
Komi: 7.5 puntos
Suicidio: Permitido
Recaptura de ko: Super Ko (prohibir cualquier posición repetida)
No requiere determinar piedras muertas
```

### Super Ko

```python
def is_superko_violation(new_state, history):
    """Verificar si viola Super Ko"""
    for past_state in history:
        if new_state == past_state:
            return True
    return False
```

### Determinación del final del juego

```
No requiere acuerdo de piedras muertas
El juego continúa hasta:
1. Ambos jugadores pasan consecutivamente
2. Luego usar búsqueda o jugar realmente para determinar el territorio
```

### Configuración KataGo

```ini
rules = tromp-taylor
komi = 7.5
```

---

## Variantes de tamaño de tablero

### Tamaños soportados

KataGo soporta múltiples tamaños de tablero:

| Tamaño | Características | Uso recomendado |
|--------|-----------------|-----------------|
| 9×9 | ~81 puntos | Principiantes, partidas rápidas |
| 13×13 | ~169 puntos | Aprendizaje avanzado |
| 19×19 | 361 puntos | Competiciones estándar |
| Personalizado | Cualquiera | Investigación, pruebas |

### Método de configuración

```ini
# Tablero 9×9
boardXSize = 9
boardYSize = 9
komi = 5.5

# Tablero 13×13
boardXSize = 13
boardYSize = 13
komi = 6.5

# Tablero no cuadrado
boardXSize = 19
boardYSize = 9
```

### Komi recomendado

| Tamaño | Reglas chinas | Reglas japonesas |
|--------|---------------|------------------|
| 9×9 | 5.5 | 5.5 |
| 13×13 | 6.5 | 6.5 |
| 19×19 | 7.5 | 6.5 |

---

## Configuración de handicap

### Partidas con handicap

El handicap es una forma de ajustar la diferencia de fuerza:

```ini
# Handicap 2 piedras
handicap = 2

# Handicap 9 piedras
handicap = 9
```

### Posiciones de handicap

```python
HANDICAP_POSITIONS = {
    2: [(3, 15), (15, 3)],
    3: [(3, 15), (15, 3), (15, 15)],
    4: [(3, 15), (15, 3), (3, 3), (15, 15)],
    # 5-9 piedras usan puntos estrella + tengen
}
```

### Komi en partidas con handicap

```ini
# Tradicional: Sin komi o medio punto
komi = 0.5

# Moderno: Ajustar según número de piedras
# Cada piedra vale ~10-15 puntos
```

---

## Configuración de reglas en modo Analysis

### Comandos GTP

```gtp
# Establecer reglas
kata-set-rules chinese

# Establecer komi
komi 7.5

# Establecer tamaño de tablero
boardsize 19
```

### Analysis API

```json
{
  "id": "query1",
  "moves": [["B", "Q4"], ["W", "D4"]],
  "rules": "chinese",
  "komi": 7.5,
  "boardXSize": 19,
  "boardYSize": 19,
  "overrideSettings": {
    "maxVisits": 1000
  }
}
```

---

## Opciones de reglas avanzadas

### Configuración de suicidio

```ini
# Prohibir suicidio (por defecto)
allowSuicide = false

# Permitir suicidio (estilo Tromp-Taylor)
allowSuicide = true
```

### Reglas de Ko

```ini
# Simple Ko (solo prohibir recaptura inmediata)
koRule = SIMPLE

# Positional Super Ko (prohibir repetir cualquier posición, sin importar quién juega)
koRule = POSITIONAL

# Situational Super Ko (prohibir repetir posiciones del mismo jugador)
koRule = SITUATIONAL
```

### Reglas de puntuación

```ini
# Conteo de área (China, AGA)
scoringRule = AREA

# Conteo de territorio (Japón, Corea)
scoringRule = TERRITORY
```

### Reglas de impuesto

Algunas reglas tienen puntuación especial para áreas de seki:

```ini
# Sin impuesto
taxRule = NONE

# Seki sin puntos
taxRule = SEKI

# Todos los ojos sin puntos
taxRule = ALL
```

---

## Entrenamiento multi-reglas

### Ventaja de KataGo

KataGo usa un solo modelo que soporta múltiples reglas:

```python
def encode_rules(rules):
    """Codificar reglas como entrada de red neuronal"""
    features = np.zeros(RULE_FEATURE_SIZE)

    # Método de puntuación
    features[0] = 1.0 if rules.scoring == 'area' else 0.0

    # Suicidio
    features[1] = 1.0 if rules.allow_suicide else 0.0

    # Regla de Ko
    features[2:5] = encode_ko_rule(rules.ko)

    # Komi (normalizado)
    features[5] = rules.komi / 15.0

    return features
```

### Entrada consciente de reglas

```
La entrada de la red neuronal incluye:
- Estado del tablero (19×19×N)
- Vector de características de reglas (K dimensiones)

Esto permite que el mismo modelo entienda diferentes reglas
```

---

## Ejemplo de cambio de reglas

### Código Python

```python
from katago import KataGo

engine = KataGo(model_path="kata.bin.gz")

# Análisis con reglas chinas
result_cn = engine.analyze(
    moves=[("B", "Q4"), ("W", "D4")],
    rules="chinese",
    komi=7.5
)

# Análisis con reglas japonesas (misma posición)
result_jp = engine.analyze(
    moves=[("B", "Q4"), ("W", "D4")],
    rules="japanese",
    komi=6.5
)

# Comparar diferencias
print(f"Tasa de victoria de Negro (reglas chinas): {result_cn['winrate']:.1%}")
print(f"Tasa de victoria de Negro (reglas japonesas): {result_jp['winrate']:.1%}")
```

### Análisis de impacto de reglas

```python
def compare_rules_impact(position, rules_list):
    """Comparar el impacto de diferentes reglas en la evaluación de la posición"""
    results = {}

    for rules in rules_list:
        analysis = engine.analyze(
            moves=position,
            rules=rules,
            komi=get_default_komi(rules)
        )
        results[rules] = {
            'winrate': analysis['winrate'],
            'score': analysis['scoreLead'],
            'best_move': analysis['moveInfos'][0]['move']
        }

    return results
```

---

## Preguntas frecuentes

### Diferencias de reglas que causan diferentes resultados

```
La misma partida puede tener diferentes resultados según las reglas:
- Diferencias de puntuación entre conteo de área y territorio
- Manejo de áreas de seki
- Impacto de los pases (tenuki)
```

### ¿Qué reglas elegir?

| Escenario | Reglas recomendadas |
|-----------|---------------------|
| Principiantes | Chinese (intuitivo, sin disputas) |
| Competiciones online | Por defecto de la plataforma (generalmente Chinese) |
| Nihon Ki-in | Japanese |
| Implementación en programas | Tromp-Taylor (más simple) |
| Competiciones profesionales en China | Chinese |

### ¿El modelo necesita entrenarse para reglas específicas?

El modelo multi-reglas de KataGo ya es muy fuerte. Pero si solo usas un conjunto de reglas, puedes considerar:

```ini
# Entrenamiento con reglas fijas (puede mejorar ligeramente la fuerza para esas reglas)
rules = chinese
```

---

## Lectura adicional

- [Análisis del mecanismo de entrenamiento de KataGo](../training) — Implementación de entrenamiento multi-reglas
- [Integración en tu proyecto](../../hands-on/integration) — Ejemplos de uso de API
- [Evaluación y benchmarking](../evaluation) — Pruebas de fuerza con diferentes reglas
