---
sidebar_position: 2
title: Guia de inicio practico de KataGo
---

# Guia de inicio practico de KataGo

Este capitulo te guiara desde la instalacion hasta el uso practico de KataGo, cubriendo todo el conocimiento operativo practico. Ya sea que quieras integrar KataGo en tu propia aplicacion o investigar profundamente su codigo fuente, aqui es tu punto de partida.

## Por que elegir KataGo?

Entre muchas IAs de Go, KataGo es actualmente la mejor opcion, por las siguientes razones:

| Ventaja | Descripcion |
|------|------|
| **Fuerza mas alta** | Mantiene el nivel mas alto en pruebas publicas |
| **Funciones mas completas** | Prediccion de puntos, analisis de territorio, soporte multiples reglas |
| **Completamente de codigo abierto** | Licencia MIT, libre de usar y modificar |
| **Actualizaciones continuas** | Desarrollo activo y soporte de comunidad |
| **Documentacion completa** | Documentacion oficial detallada, recursos comunitarios ricos |
| **Soporte multiplataforma** | Funciona en Linux, macOS, Windows |

## Contenido de este capitulo

### [Instalacion y configuracion](./setup.md)

Construir el entorno de KataGo desde cero:

- Requisitos del sistema y sugerencias de hardware
- Pasos de instalacion para cada plataforma (macOS / Linux / Windows)
- Guia de descarga y seleccion de modelos
- Explicacion detallada del archivo de configuracion

### [Comandos comunes](./commands.md)

Dominar la forma de usar KataGo:

- Introduccion al protocolo GTP (Go Text Protocol)
- Comandos GTP comunes y ejemplos
- Metodo de uso del Analysis Engine
- Explicacion completa de la API JSON

### [Arquitectura del codigo fuente](./architecture.md)

Entender profundamente los detalles de implementacion de KataGo:

- Vista general de la estructura de directorios del proyecto
- Analisis de la arquitectura de red neuronal
- Detalles de implementacion del motor de busqueda
- Vista general del proceso de entrenamiento

## Inicio rapido

Si solo quieres probar KataGo rapidamente, esta es la forma mas simple:

### macOS (usando Homebrew)

```bash
# Instalacion
brew install katago

# Descargar modelo (elegir modelo pequeno para prueba)
curl -L -o kata-b18c384.bin.gz \
  https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz

# Ejecutar modo GTP
katago gtp -model kata-b18c384.bin.gz -config gtp_example.cfg
```

### Linux (version precompilada)

```bash
# Descargar version precompilada
wget https://github.com/lightvector/KataGo/releases/download/v1.15.3/katago-v1.15.3-opencl-linux-x64.zip

# Descomprimir
unzip katago-v1.15.3-opencl-linux-x64.zip

# Descargar modelo
wget https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz

# Ejecutar
./katago gtp -model kata-b18c384nbt-*.bin.gz -config default_gtp.cfg
```

### Verificar instalacion

Despues de iniciar exitosamente, veras el prompt GTP. Intenta ingresar los siguientes comandos:

```
name
= KataGo

version
= 1.15.3

boardsize 19
=

genmove black
= Q16
```

## Guia de escenarios de uso

Segun tu necesidad, este es el orden de lectura y enfoque sugerido:

### Escenario 1: Integracion en app de Go

Quieres usar KataGo como motor de IA en tu propia aplicacion de Go.

**Lectura enfocada**:
1. [Instalacion y configuracion](./setup.md) - Entender requisitos de despliegue
2. [Comandos comunes](./commands.md) - Especialmente la seccion de Analysis Engine

**Conocimiento clave**:
- Usar modo Analysis Engine en lugar de modo GTP
- Comunicarse con KataGo a traves de API JSON
- Ajustar parametros de busqueda segun hardware

### Escenario 2: Construir servidor de juego

Quieres configurar un servidor que permita a los usuarios jugar contra la IA.

**Lectura enfocada**:
1. [Instalacion y configuracion](./setup.md) - Seccion de configuracion de GPU
2. [Comandos comunes](./commands.md) - Seccion de protocolo GTP

**Conocimiento clave**:
- Usar modo GTP para jugar
- Estrategia de despliegue de multiples instancias
- Metodo de ajuste de fuerza

### Escenario 3: Investigacion de algoritmos de IA

Quieres investigar profundamente la implementacion de KataGo, posiblemente modificar o experimentar.

**Lectura enfocada**:
1. [Arquitectura del codigo fuente](./architecture.md) - Leer todo cuidadosamente
2. Todos los analisis de papers en el capitulo de conocimientos previos

**Conocimiento clave**:
- Estructura del codigo C++
- Detalles de arquitectura de red neuronal
- Forma de implementacion de MCTS

### Escenario 4: Entrenar tu propio modelo

Quieres entrenar desde cero o afinar modelos de KataGo.

**Lectura enfocada**:
1. [Arquitectura del codigo fuente](./architecture.md) - Seccion de proceso de entrenamiento
2. [Analisis del paper de KataGo](../background-info/katago-paper.md)

**Conocimiento clave**:
- Formato de datos de entrenamiento
- Uso de scripts de entrenamiento
- Configuracion de hiperparametros

## Sugerencias de hardware

KataGo puede ejecutarse en varios hardware, pero la diferencia de rendimiento es grande:

| Configuracion de hardware | Rendimiento esperado | Escenario adecuado |
|---------|---------|---------|
| **GPU de gama alta** (RTX 4090) | ~2000 playouts/seg | Analisis de primer nivel, busqueda rapida |
| **GPU de gama media** (RTX 3060) | ~500 playouts/seg | Analisis general, juego |
| **GPU de entrada** (GTX 1650) | ~100 playouts/seg | Uso basico |
| **Apple Silicon** (M1/M2) | ~200-400 playouts/seg | Desarrollo en macOS |
| **Solo CPU** | ~10-30 playouts/seg | Aprendizaje, pruebas |

:::tip
Incluso con hardware mas lento, KataGo puede proporcionar analisis valioso. Menos busqueda reducira la precision, pero generalmente es suficiente para ensenanza y aprendizaje.
:::

## Preguntas frecuentes

### Cual es la diferencia entre KataGo y Leela Zero?

| Aspecto | KataGo | Leela Zero |
|------|--------|------------|
| Fuerza | Mas fuerte | Mas debil |
| Funciones | Ricas (puntos, territorio) | Basicas |
| Multiples reglas | Soporta | No soporta |
| Estado de desarrollo | Activo | Modo de mantenimiento |
| Eficiencia de entrenamiento | Alta | Mas baja |

### Se necesita GPU?

No es obligatorio, pero se recomienda fuertemente:
- **Con GPU**: Puede hacer analisis rapido, obtener resultados de alta calidad
- **Sin GPU**: Puede usar backend Eigen, pero mas lento

### Diferencia en archivos de modelo?

| Tamano de modelo | Tamano de archivo | Fuerza | Velocidad |
|---------|---------|------|------|
| b10c128 | ~20 MB | Media | Mas rapida |
| b18c384 | ~140 MB | Fuerte | Rapida |
| b40c256 | ~250 MB | Muy fuerte | Media |
| b60c320 | ~500 MB | Mas fuerte | Lenta |

Generalmente se recomienda usar b18c384 o b40c256, logra un equilibrio entre fuerza y velocidad.

## Recursos relacionados

- [KataGo GitHub](https://github.com/lightvector/KataGo)
- [Sitio de entrenamiento de KataGo](https://katagotraining.org/)
- [Comunidad Discord de KataGo](https://discord.gg/bqkZAz3)
- [Lizzie](https://github.com/featurecat/lizzie) - GUI para usar con KataGo

Estas listo? Comencemos con [Instalacion y configuracion](./setup.md).

