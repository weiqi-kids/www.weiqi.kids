---
sidebar_position: 2
title: Introducao Pratica ao KataGo
---

# Guia de Introducao Pratica ao KataGo

Este capitulo levara voce desde a instalacao ate o uso real do KataGo, cobrindo todo o conhecimento de operacao pratica. Seja para integrar KataGo em sua propria aplicacao ou pesquisar profundamente seu codigo-fonte, este e seu ponto de partida.

## Por que Escolher KataGo?

Entre as muitas IAs de Go, KataGo e atualmente a melhor escolha, pelas seguintes razoes:

| Vantagem | Descricao |
|------|------|
| **Forca mais forte** | Mantem consistentemente o mais alto nivel em testes publicos |
| **Funcionalidades mais completas** | Predicao de pontos, analise de territorio, suporte multi-regras |
| **Completamente codigo aberto** | Licenca MIT, livre para usar e modificar |
| **Atualizacao continua** | Desenvolvimento ativo e suporte da comunidade |
| **Documentacao completa** | Documentacao oficial detalhada, recursos da comunidade ricos |
| **Multiplataforma** | Roda em Linux, macOS, Windows |

## Conteudo deste Capitulo

### [Instalacao e Configuracao](./setup.md)

Construir ambiente KataGo do zero:

- Requisitos de sistema e sugestoes de hardware
- Passos de instalacao para cada plataforma (macOS / Linux / Windows)
- Guia de download e selecao de modelos
- Explicacao detalhada de arquivos de configuracao

### [Comandos Comuns](./commands.md)

Dominar formas de uso do KataGo:

- Introducao ao protocolo GTP (Go Text Protocol)
- Comandos GTP comuns e exemplos
- Metodo de uso do Analysis Engine
- Explicacao completa da API JSON

### [Arquitetura do Codigo-fonte](./architecture.md)

Entender profundamente detalhes de implementacao do KataGo:

- Visao geral da estrutura de diretorios do projeto
- Analise da arquitetura da rede neural
- Detalhes de implementacao do motor de busca
- Visao geral do processo de treinamento

## Inicio Rapido

Se voce so quer experimentar KataGo rapidamente, aqui esta a forma mais simples:

### macOS (usando Homebrew)

```bash
# Instalacao
brew install katago

# Download do modelo (escolha modelo menor para teste)
curl -L -o kata-b18c384.bin.gz \
  https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz

# Executar modo GTP
katago gtp -model kata-b18c384.bin.gz -config gtp_example.cfg
```

### Linux (versao pre-compilada)

```bash
# Download da versao pre-compilada
wget https://github.com/lightvector/KataGo/releases/download/v1.15.3/katago-v1.15.3-opencl-linux-x64.zip

# Descompactar
unzip katago-v1.15.3-opencl-linux-x64.zip

# Download do modelo
wget https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz

# Executar
./katago gtp -model kata-b18c384nbt-*.bin.gz -config default_gtp.cfg
```

### Verificar Instalacao

Apos iniciar com sucesso, voce vera o prompt GTP. Tente inserir os seguintes comandos:

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

## Guia por Cenario de Uso

Com base em suas necessidades, aqui esta a ordem de leitura sugerida e foco:

### Cenario 1: Integrar em App de Go

Voce quer usar KataGo como motor de IA em sua propria aplicacao de Go.

**Leitura prioritaria**:
1. [Instalacao e Configuracao](./setup.md) - Entender requisitos de implantacao
2. [Comandos Comuns](./commands.md) - Especialmente parte do Analysis Engine

**Conhecimento-chave**:
- Usar modo Analysis Engine em vez de modo GTP
- Comunicar com KataGo via API JSON
- Ajustar parametros de busca conforme hardware

### Cenario 2: Construir Servidor de Jogo

Voce quer configurar um servidor onde usuarios jogam contra IA.

**Leitura prioritaria**:
1. [Instalacao e Configuracao](./setup.md) - Parte de configuracao de GPU
2. [Comandos Comuns](./commands.md) - Parte do protocolo GTP

**Conhecimento-chave**:
- Usar modo GTP para jogos
- Estrategia de implantacao multi-instancia
- Metodo de ajuste de forca

### Cenario 3: Pesquisar Algoritmos de IA

Voce quer pesquisar profundamente a implementacao do KataGo, possivelmente modificar ou experimentar.

**Leitura prioritaria**:
1. [Arquitetura do Codigo-fonte](./architecture.md) - Leitura completa cuidadosa
2. Todos os artigos analisados no capitulo de conhecimento de fundo

**Conhecimento-chave**:
- Estrutura do codigo C++
- Detalhes da arquitetura de rede neural
- Forma de implementacao do MCTS

### Cenario 4: Treinar Seu Proprio Modelo

Voce quer treinar do zero ou fazer fine-tune de modelos KataGo.

**Leitura prioritaria**:
1. [Arquitetura do Codigo-fonte](./architecture.md) - Parte do processo de treinamento
2. [Analise do Artigo KataGo](../background-info/katago-paper.md)

**Conhecimento-chave**:
- Formato de dados de treinamento
- Uso de scripts de treinamento
- Configuracao de hiperparametros

## Sugestoes de Hardware

KataGo pode rodar em varios hardwares, mas a diferenca de desempenho e grande:

| Configuracao de Hardware | Desempenho Esperado | Cenario de Uso |
|---------|---------|---------|
| **GPU topo** (RTX 4090) | ~2000 playouts/seg | Analise top, busca rapida |
| **GPU intermediaria** (RTX 3060) | ~500 playouts/seg | Analise geral, jogo |
| **GPU basica** (GTX 1650) | ~100 playouts/seg | Uso basico |
| **Apple Silicon** (M1/M2) | ~200-400 playouts/seg | Desenvolvimento macOS |
| **Apenas CPU** | ~10-30 playouts/seg | Aprendizado, teste |

:::tip
Mesmo em hardware mais lento, KataGo pode fornecer analise valiosa. Menor volume de busca reduzira a precisao, mas para ensino e aprendizado geralmente e suficiente.
:::

## Perguntas Frequentes

### Qual a diferenca entre KataGo e Leela Zero?

| Aspecto | KataGo | Leela Zero |
|------|--------|------------|
| Forca | Mais forte | Mais fraca |
| Funcionalidades | Ricas (pontos, territorio) | Basicas |
| Multi-regras | Suporta | Nao suporta |
| Status de desenvolvimento | Ativo | Modo manutencao |
| Eficiencia de treinamento | Alta | Mais baixa |

### Precisa de GPU?

Nao e obrigatorio, mas fortemente recomendado:
- **Com GPU**: Pode fazer analise rapida, obter resultados de alta qualidade
- **Sem GPU**: Pode usar backend Eigen, mas velocidade mais lenta

### Diferenca entre arquivos de modelo?

| Tamanho do Modelo | Tamanho do Arquivo | Forca | Velocidade |
|---------|---------|------|------|
| b10c128 | ~20 MB | Media | Mais rapida |
| b18c384 | ~140 MB | Forte | Rapida |
| b40c256 | ~250 MB | Muito forte | Media |
| b60c320 | ~500 MB | Mais forte | Lenta |

Geralmente recomenda-se usar b18c384 ou b40c256, equilibrando forca e velocidade.

## Recursos Relacionados

- [GitHub KataGo](https://github.com/lightvector/KataGo)
- [Site de Treinamento KataGo](https://katagotraining.org/)
- [Comunidade Discord KataGo](https://discord.gg/bqkZAz3)
- [Lizzie](https://github.com/featurecat/lizzie) - GUI para usar com KataGo

Pronto? Vamos comecar com [Instalacao e Configuracao](./setup.md)!

