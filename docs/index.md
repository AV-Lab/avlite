# AVLite Documentation

AVLite is a lightweight, extensible autonomous vehicle software stack for rapid prototyping, research, and education.

## Core Components

| Component | Description |
|-----------|-------------|
| **c10_perception** | Interfaces and capability system for perception, localization, mapping |
| **c20_planning** | Global and local path planning |
| **c30_control** | Vehicle controllers (Stanley, PID) |
| **c40_execution** | Execution loop, simulator bridges |
| **c50_visualization** | Real-time GUI |
| **c60_common** | Utilities and settings |

## Quick Start

```bash
pip install -r requirements.txt
python -m avlite
```

## Next Steps

- [Plugin Development Guide](plugin-development.md) - Create custom components

