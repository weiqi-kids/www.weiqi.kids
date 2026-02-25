---
sidebar_position: 4
title: Participar en la comunidad de codigo abierto
description: Unirse a la comunidad de codigo abierto de KataGo, contribuir poder de computo o codigo
---

# Participar en la comunidad de codigo abierto

KataGo es un proyecto de codigo abierto activo con multiples formas de contribuir.

---

## Vision general de formas de contribucion

| Forma | Dificultad | Requisitos |
|-------|------------|------------|
| **Contribuir poder de computo** | Baja | Computadora con GPU |
| **Reportar problemas** | Baja | Cuenta de GitHub |
| **Mejorar documentacion** | Media | Familiaridad con contenido tecnico |
| **Contribuir codigo** | Alta | Capacidad de desarrollo C++/Python |

---

## Contribuir poder de computo: Entrenamiento distribuido

### Introduccion a KataGo Training

KataGo Training es una red de entrenamiento distribuido global:

- Los voluntarios contribuyen poder de GPU para ejecutar auto-juego
- Los datos de auto-juego se suben al servidor central
- El servidor entrena periodicamente nuevos modelos
- Los nuevos modelos se distribuyen a los voluntarios para continuar jugando

Sitio web: https://katagotraining.org/

### Pasos para participar

#### 1. Crear cuenta

Ve a https://katagotraining.org/ para registrarte.

#### 2. Descargar KataGo

```bash
# Descargar ultima version
wget https://github.com/lightvector/KataGo/releases/download/v1.15.3/katago-v1.15.3-cuda11.1-linux-x64.zip
unzip katago-v1.15.3-cuda11.1-linux-x64.zip
```

#### 3. Configurar modo contribute

```bash
# La primera ejecucion te guiara en la configuracion
./katago contribute -username YOUR_USERNAME -password YOUR_PASSWORD
```

El sistema automaticamente:
- Descargara el modelo mas reciente
- Ejecutara auto-juego
- Subira datos de partidas

#### 4. Ejecutar en segundo plano

```bash
# Usar screen o tmux para ejecutar en segundo plano
screen -S katago
./katago contribute -username YOUR_USERNAME -password YOUR_PASSWORD
# Ctrl+A, D para salir de screen
```

### Estadisticas de contribucion

Puedes ver en https://katagotraining.org/contributions/:
- Tu ranking de contribucion
- Total de partidas contribuidas
- Modelos entrenados recientemente

---

## Reportar problemas

### Donde reportar

- **GitHub Issues**: https://github.com/lightvector/KataGo/issues
- **Discord**: https://discord.gg/bqkZAz3

### Un buen reporte de problema incluye

1. **Version de KataGo**: `katago version`
2. **Sistema operativo**: Windows/Linux/macOS
3. **Hardware**: Modelo de GPU, memoria
4. **Mensaje de error completo**: Copiar log completo
5. **Pasos para reproducir**: Como activar el problema

### Ejemplo

```markdown
## Descripcion del problema
Error de memoria insuficiente al ejecutar benchmark

## Entorno
- Version de KataGo: 1.15.3
- Sistema operativo: Ubuntu 22.04
- GPU: RTX 3060 12GB
- Modelo: kata-b40c256.bin.gz

## Mensaje de error
```
CUDA error: out of memory
```

## Pasos para reproducir
1. Ejecutar `katago benchmark -model kata-b40c256.bin.gz`
2. Esperar aproximadamente 30 segundos
3. Aparece el error
```

---

## Mejorar documentacion

### Ubicacion de documentos

- **README**: `README.md`
- **Documentacion GTP**: `docs/GTP_Extensions.md`
- **Documentacion Analysis**: `docs/Analysis_Engine.md`
- **Documentacion de entrenamiento**: `python/README.md`

### Proceso de contribucion

1. Fork del proyecto
2. Crear nueva rama
3. Modificar documentacion
4. Enviar Pull Request

```bash
git clone https://github.com/YOUR_USERNAME/KataGo.git
cd KataGo
git checkout -b improve-docs
# Editar documentacion
git add .
git commit -m "Improve documentation for Analysis Engine"
git push origin improve-docs
# Crear Pull Request en GitHub
```

---

## Contribuir codigo

### Configuracion del entorno de desarrollo

```bash
# Clonar proyecto
git clone https://github.com/lightvector/KataGo.git
cd KataGo

# Compilar (modo Debug)
cd cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=OPENCL -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# Ejecutar pruebas
./katago runtests
```

### Estilo de codigo

KataGo usa el siguiente estilo de codigo:

**C++**:
- Indentacion de 2 espacios
- Llaves en la misma linea
- Variables en camelCase
- Clases en PascalCase

```cpp
class ExampleClass {
public:
  void exampleMethod() {
    int localVariable = 0;
    if(condition) {
      doSomething();
    }
  }
};
```

**Python**:
- Seguir PEP 8
- Indentacion de 4 espacios

### Areas de contribucion

| Area | Ubicacion de archivos | Habilidades requeridas |
|------|----------------------|------------------------|
| Motor principal | `cpp/` | C++, CUDA/OpenCL |
| Programa de entrenamiento | `python/` | Python, PyTorch |
| Protocolo GTP | `cpp/command/gtp.cpp` | C++ |
| Analysis API | `cpp/command/analysis.cpp` | C++, JSON |
| Pruebas | `cpp/tests/` | C++ |

### Proceso de Pull Request

1. **Crear Issue**: Primero discutir el cambio que quieres hacer
2. **Fork & Clone**: Crear tu propia rama
3. **Desarrollar y probar**: Asegurar que todas las pruebas pasen
4. **Enviar PR**: Describir detalladamente el contenido del cambio
5. **Code Review**: Responder al feedback de los mantenedores
6. **Merge**: Los mantenedores fusionan tu codigo

### Ejemplo de PR

```markdown
## Descripcion del cambio
Agregar soporte para reglas de Nueva Zelanda

## Contenido del cambio
- Agregar regla NEW_ZEALAND en rules.cpp
- Actualizar comando GTP para soportar `kata-set-rules nz`
- Agregar pruebas unitarias

## Resultados de pruebas
- Todas las pruebas existentes pasan
- Nuevas pruebas pasan

## Issue relacionado
Fixes #123
```

---

## Recursos de la comunidad

### Enlaces oficiales

| Recurso | Enlace |
|---------|--------|
| GitHub | https://github.com/lightvector/KataGo |
| Discord | https://discord.gg/bqkZAz3 |
| Red de entrenamiento | https://katagotraining.org/ |

### Foros de discusion

- **Discord**: Discusion en tiempo real, preguntas tecnicas
- **GitHub Discussions**: Discusiones largas, propuestas de funciones
- **Reddit r/baduk**: Discusion general de IA de Go

### Proyectos relacionados

| Proyecto | Descripcion | Enlace |
|----------|-------------|--------|
| KaTrain | Herramienta de ensenanza y analisis | github.com/sanderland/katrain |
| Lizzie | Interfaz de analisis | github.com/featurecat/lizzie |
| Sabaki | Editor de registros de partidas | sabaki.yichuanshen.de |
| BadukAI | Analisis en linea | baduk.ai |

---

## Reconocimiento y recompensas

### Lista de contribuidores

Todos los contribuidores se listan en:
- Pagina de Contributors de GitHub
- Ranking de contribucion de KataGo Training

### Beneficios del aprendizaje

Beneficios de participar en proyectos de codigo abierto:
- Aprender arquitectura de sistemas de IA de nivel industrial
- Intercambiar con desarrolladores globales
- Acumular registro de contribuciones de codigo abierto
- Comprender profundamente la tecnologia de IA de Go

---

## Lectura adicional

- [Guia del codigo fuente](../source-code) — Entender la estructura del codigo
- [Analisis del mecanismo de entrenamiento de KataGo](../training) — Experimentos de entrenamiento local
- [Entender la IA de Go en un articulo](../../how-it-works/) — Principios tecnicos
