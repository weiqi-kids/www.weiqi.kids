---
sidebar_position: 4
title: Participando da Comunidade Open Source
description: Junte-se a comunidade open source do KataGo, contribua com poder computacional ou codigo
---

# Participando da Comunidade Open Source

O KataGo e um projeto open source ativo, com varias formas de participar e contribuir.

---

## Visao Geral das Formas de Contribuicao

| Forma | Dificuldade | Requisitos |
|-------|-------------|------------|
| **Contribuir poder computacional** | Baixa | Computador com GPU |
| **Reportar problemas** | Baixa | Conta no GitHub |
| **Melhorar documentacao** | Media | Familiaridade com conteudo tecnico |
| **Contribuir codigo** | Alta | Habilidades de desenvolvimento C++/Python |

---

## Contribuir Poder Computacional: Treinamento Distribuido

### Introducao ao KataGo Training

KataGo Training e uma rede de treinamento distribuido global:

- Voluntarios contribuem poder de GPU para executar auto-jogo
- Dados de auto-jogo sao enviados para servidor central
- O servidor treina novos modelos periodicamente
- Novos modelos sao distribuidos para voluntarios continuarem jogando

Site oficial: https://katagotraining.org/

### Passos para Participar

#### 1. Criar Conta

Acesse https://katagotraining.org/ e registre uma conta.

#### 2. Baixar KataGo

```bash
# Baixar versao mais recente
wget https://github.com/lightvector/KataGo/releases/download/v1.15.3/katago-v1.15.3-cuda11.1-linux-x64.zip
unzip katago-v1.15.3-cuda11.1-linux-x64.zip
```

#### 3. Configurar Modo Contribute

```bash
# Primeira execucao vai guiar voce na configuracao
./katago contribute -username SEU_USUARIO -password SUA_SENHA
```

O sistema vai automaticamente:
- Baixar o modelo mais recente
- Executar auto-jogo
- Enviar dados das partidas

#### 4. Execucao em Background

```bash
# Usar screen ou tmux para execucao em background
screen -S katago
./katago contribute -username SEU_USUARIO -password SUA_SENHA
# Ctrl+A, D para sair do screen
```

### Estatisticas de Contribuicao

Voce pode ver em https://katagotraining.org/contributions/:
- Sua posicao no ranking
- Total de partidas contribuidas
- Modelos treinados recentemente

---

## Reportar Problemas

### Onde Reportar

- **GitHub Issues**: https://github.com/lightvector/KataGo/issues
- **Discord**: https://discord.gg/bqkZAz3

### Um Bom Relatorio de Problema Contem

1. **Versao do KataGo**: `katago version`
2. **Sistema operacional**: Windows/Linux/macOS
3. **Hardware**: Modelo de GPU, memoria
4. **Mensagem de erro completa**: Copie o log completo
5. **Passos para reproduzir**: Como acionar o problema

### Exemplo

```markdown
## Descricao do Problema
Erro de memoria insuficiente ao executar benchmark

## Ambiente
- Versao KataGo: 1.15.3
- Sistema operacional: Ubuntu 22.04
- GPU: RTX 3060 12GB
- Modelo: kata-b40c256.bin.gz

## Mensagem de Erro
```
CUDA error: out of memory
```

## Passos para Reproduzir
1. Executar `katago benchmark -model kata-b40c256.bin.gz`
2. Aguardar aproximadamente 30 segundos
3. Erro aparece
```

---

## Melhorar Documentacao

### Localizacao da Documentacao

- **README**: `README.md`
- **Documentacao GTP**: `docs/GTP_Extensions.md`
- **Documentacao Analysis**: `docs/Analysis_Engine.md`
- **Documentacao de Treinamento**: `python/README.md`

### Fluxo de Contribuicao

1. Fork do projeto
2. Criar nova branch
3. Modificar documentacao
4. Submeter Pull Request

```bash
git clone https://github.com/SEU_USUARIO/KataGo.git
cd KataGo
git checkout -b improve-docs
# Editar documentacao
git add .
git commit -m "Improve documentation for Analysis Engine"
git push origin improve-docs
# Criar Pull Request no GitHub
```

---

## Contribuir Codigo

### Configurar Ambiente de Desenvolvimento

```bash
# Clonar projeto
git clone https://github.com/lightvector/KataGo.git
cd KataGo

# Compilar (modo Debug)
cd cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=OPENCL -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# Executar testes
./katago runtests
```

### Estilo de Codigo

O KataGo usa o seguinte estilo de codigo:

**C++**:
- Indentacao de 2 espacos
- Chaves na mesma linha
- Nomes de variaveis em camelCase
- Nomes de classes em PascalCase

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
- Segue PEP 8
- Indentacao de 4 espacos

### Areas de Contribuicao

| Area | Localizacao | Habilidades Necessarias |
|------|-------------|------------------------|
| Engine principal | `cpp/` | C++, CUDA/OpenCL |
| Programa de treinamento | `python/` | Python, PyTorch |
| Protocolo GTP | `cpp/command/gtp.cpp` | C++ |
| API Analysis | `cpp/command/analysis.cpp` | C++, JSON |
| Testes | `cpp/tests/` | C++ |

### Fluxo de Pull Request

1. **Criar Issue**: Primeiro discuta a mudanca que voce quer fazer
2. **Fork & Clone**: Crie sua propria branch
3. **Desenvolver e testar**: Garanta que todos os testes passem
4. **Submeter PR**: Descreva detalhadamente o conteudo da mudanca
5. **Code Review**: Responda ao feedback dos mantenedores
6. **Merge**: Mantenedores fazem merge do seu codigo

### Exemplo de PR

```markdown
## Descricao da Mudanca
Adiciona suporte para regras da Nova Zelandia

## Conteudo da Mudanca
- Adiciona regra NEW_ZEALAND em rules.cpp
- Atualiza comando GTP para suportar `kata-set-rules nz`
- Adiciona testes unitarios

## Resultados dos Testes
- Todos os testes existentes passam
- Novos testes passam

## Issue Relacionada
Fixes #123
```

---

## Recursos da Comunidade

### Links Oficiais

| Recurso | Link |
|---------|------|
| GitHub | https://github.com/lightvector/KataGo |
| Discord | https://discord.gg/bqkZAz3 |
| Rede de Treinamento | https://katagotraining.org/ |

### Foruns de Discussao

- **Discord**: Discussoes em tempo real, perguntas tecnicas
- **GitHub Discussions**: Discussoes longas, propostas de funcionalidades
- **Reddit r/baduk**: Discussoes gerais sobre IA de Go

### Projetos Relacionados

| Projeto | Descricao | Link |
|---------|-----------|------|
| KaTrain | Ferramenta de ensino e analise | github.com/sanderland/katrain |
| Lizzie | Interface de analise | github.com/featurecat/lizzie |
| Sabaki | Editor de registros de partidas | sabaki.yichuanshen.de |
| BadukAI | Analise online | baduk.ai |

---

## Reconhecimento e Recompensas

### Lista de Contribuidores

Todos os contribuidores sao listados em:
- Pagina de Contributors do GitHub
- Ranking de contribuicoes do KataGo Training

### Aprendizado Obtido

Beneficios de participar de projetos open source:
- Aprender arquitetura de sistemas de IA de nivel industrial
- Trocar experiencias com desenvolvedores de todo o mundo
- Acumular historico de contribuicoes open source
- Entender profundamente tecnologia de IA para Go

---

## Leitura Adicional

- [Guia do Codigo-fonte](../source-code) — Entender a estrutura do codigo
- [Analise do Mecanismo de Treinamento do KataGo](../training) — Experimentos de treinamento local
- [Entenda IA de Go em um Artigo](../../how-it-works/) — Principios tecnicos
