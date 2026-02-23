---
sidebar_position: 4
title: Contributing to Open Source
description: Join the KataGo open source community, contribute computing power or code
---

# Contributing to Open Source

KataGo is an active open source project with multiple ways to contribute.

---

## Contribution Methods Overview

| Method | Difficulty | Requirements |
|--------|------------|--------------|
| **Contribute computing power** | Low | Computer with GPU |
| **Report issues** | Low | GitHub account |
| **Improve documentation** | Medium | Familiar with technical content |
| **Contribute code** | High | C++/Python development skills |

---

## Contribute Computing Power: Distributed Training

### KataGo Training Introduction

KataGo Training is a global distributed training network:

- Volunteers contribute GPU power for self-play
- Self-play data uploaded to central server
- Server periodically trains new models
- New models distributed to volunteers for continued play

Website: https://katagotraining.org/

### Participation Steps

#### 1. Create Account

Go to https://katagotraining.org/ and register.

#### 2. Download KataGo

```bash
# Download latest version
wget https://github.com/lightvector/KataGo/releases/download/v1.15.3/katago-v1.15.3-cuda11.1-linux-x64.zip
unzip katago-v1.15.3-cuda11.1-linux-x64.zip
```

#### 3. Configure Contribute Mode

```bash
# First run will guide you through setup
./katago contribute -username YOUR_USERNAME -password YOUR_PASSWORD
```

The system will automatically:
- Download latest model
- Run self-play
- Upload game data

#### 4. Background Execution

```bash
# Use screen or tmux for background execution
screen -S katago
./katago contribute -username YOUR_USERNAME -password YOUR_PASSWORD
# Ctrl+A, D to detach from screen
```

### Contribution Statistics

You can view at https://katagotraining.org/contributions/:
- Your contribution ranking
- Total contributed games
- Recently trained models

---

## Report Issues

### Where to Report

- **GitHub Issues**: https://github.com/lightvector/KataGo/issues
- **Discord**: https://discord.gg/bqkZAz3

### Good Issue Reports Include

1. **KataGo version**: `katago version`
2. **Operating system**: Windows/Linux/macOS
3. **Hardware**: GPU model, memory
4. **Complete error message**: Copy full log
5. **Reproduction steps**: How to trigger the issue

### Example

```markdown
## Problem Description
Out of memory error when running benchmark

## Environment
- KataGo version: 1.15.3
- Operating system: Ubuntu 22.04
- GPU: RTX 3060 12GB
- Model: kata-b40c256.bin.gz

## Error Message
```
CUDA error: out of memory
```

## Reproduction Steps
1. Run `katago benchmark -model kata-b40c256.bin.gz`
2. Wait about 30 seconds
3. Error appears
```

---

## Improve Documentation

### Documentation Locations

- **README**: `README.md`
- **GTP docs**: `docs/GTP_Extensions.md`
- **Analysis docs**: `docs/Analysis_Engine.md`
- **Training docs**: `python/README.md`

### Contribution Process

1. Fork project
2. Create new branch
3. Edit documentation
4. Submit Pull Request

```bash
git clone https://github.com/YOUR_USERNAME/KataGo.git
cd KataGo
git checkout -b improve-docs
# Edit documentation
git add .
git commit -m "Improve documentation for Analysis Engine"
git push origin improve-docs
# Create Pull Request on GitHub
```

---

## Contribute Code

### Development Environment Setup

```bash
# Clone project
git clone https://github.com/lightvector/KataGo.git
cd KataGo

# Compile (Debug mode)
cd cpp
mkdir build && cd build
cmake .. -DUSE_BACKEND=OPENCL -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# Run tests
./katago runtests
```

### Code Style

KataGo uses the following code style:

**C++**:
- 2 space indent
- Braces on same line
- Variables use camelCase
- Classes use PascalCase

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
- Follow PEP 8
- 4 space indent

### Contribution Areas

| Area | File Location | Skills Required |
|------|--------------|-----------------|
| Core engine | `cpp/` | C++, CUDA/OpenCL |
| Training | `python/` | Python, PyTorch |
| GTP protocol | `cpp/command/gtp.cpp` | C++ |
| Analysis API | `cpp/command/analysis.cpp` | C++, JSON |
| Tests | `cpp/tests/` | C++ |

### Pull Request Process

1. **Create Issue**: Discuss your intended changes first
2. **Fork & Clone**: Create your own branch
3. **Develop & Test**: Ensure all tests pass
4. **Submit PR**: Describe changes in detail
5. **Code Review**: Respond to maintainer feedback
6. **Merge**: Maintainer merges your code

### PR Example

```markdown
## Change Description
Add support for New Zealand rules

## Changes Made
- Add NEW_ZEALAND rules in rules.cpp
- Update GTP commands to support `kata-set-rules nz`
- Add unit tests

## Test Results
- All existing tests pass
- New tests pass

## Related Issue
Fixes #123
```

---

## Community Resources

### Official Links

| Resource | Link |
|----------|------|
| GitHub | https://github.com/lightvector/KataGo |
| Discord | https://discord.gg/bqkZAz3 |
| Training Network | https://katagotraining.org/ |

### Discussion Forums

- **Discord**: Real-time discussion, technical Q&A
- **GitHub Discussions**: Long-form discussions, feature proposals
- **Reddit r/baduk**: General Go AI discussion

### Related Projects

| Project | Description | Link |
|---------|-------------|------|
| KaTrain | Teaching analysis tool | github.com/sanderland/katrain |
| Lizzie | Analysis interface | github.com/featurecat/lizzie |
| Sabaki | Game record editor | sabaki.yichuanshen.de |
| BadukAI | Online analysis | baduk.ai |

---

## Recognition & Rewards

### Contributors List

All contributors are listed on:
- GitHub Contributors page
- KataGo Training contribution leaderboard

### Learning Benefits

Benefits of participating in open source:
- Learn industrial-grade AI system architecture
- Exchange with global developers
- Build open source contribution record
- Deep understanding of Go AI technology

---

## Further Reading

- [Source Code Guide](../source-code) — Understanding code structure
- [KataGo Training Mechanism](../training) — Local training experiments
- [Understanding Go AI](../../how-it-works/) — Technical principles
