<p align="center">
  <img src="data/imgs/logo-black-bg.png" alt="AVLite Logo" width="200">
</p>

# AVLite - Modular Autonomous Vehicle Stack

AVLite is a lightweight, extensible autonomous vehicle software stack designed for rapid prototyping, research, and education. It provides clean abstractions for perception, planning, and control while maintaining flexibility through a plugin-based architecture.

**ROS2 & Autoware Ready**: Built-in ROS2 executor extension with native Autoware message support (Trajectory, ControlCommand, etc.).

![](docs/imgs/tk_visualizer.png)

## Architecture Overview

AVLite follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                      Visualization                          │
│              (Real-time GUI with Tkinter)                   │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │
┌─────────────────────────────────────────────────────────────┐
│                       Execution Layer                       │
│          (SyncExecuter / AsyncThreadedExecuter)             │
└─────────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Perception  │ │   Planning   │ │   Control    │ │  World       │
│              │ │              │ │              │ │  Bridge      │
│  - Detect    │ │  - Global    │ │  - Stanley   │ │              │
│  - Track     │ │  - Local     │ │  - PID       │ │  - BasicSim  │
│  - Predict   │ │  - Lattice   │ │              │ │  - CARLA     │
│              │ │              │ │              │ │  - Gazebo    │
│              │ │              │ │              │ │  - ROS2      │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

### Core Components

- **c10_perception**: Interfaces for detection, tracking, prediction, localization, mapping; HD map support
- **c20_planning**: Global planning (A*, HD map routing) and local planning (lattice-based, RRT)
- **c30_control**: Vehicle control algorithms (Stanley, PID)
- **c40_execution**: Execution orchestration with support for sync/async modes and simulator bridges
- **c50_visualization**: Real-time Tkinter-based GUI for debugging and monitoring
- **c60_common**: Utilities, settings management, and capability definitions
- **extensions**: Plugin system for custom components (includes ROS2 executor with Autoware messages)

### Key Features

**Strategy Pattern Architecture**: All major components (perception, planning, control) use the strategy pattern with automatic registration, allowing runtime selection and hot-reloading without code changes.

**Capability-Based System**: Components declare their requirements and capabilities, enabling automatic compatibility checking between perception strategies and world bridges.

**YAML-Based Configuration**: Profile-based configuration system allows quick switching between different algorithm combinations and parameters.

**Hot Reloading**: Modify code and configuration files while the system is running without restarting.

**Multiple Simulator Support**: Works with BasicSim (built-in), CARLA, Gazebo, and ROS2/Autoware through abstract world bridge interface.

**Extensible Plugin System**: Add custom perception, planning, or control algorithms as plugins without modifying core code.

## Why AVLite?

- **Lightweight**: Small codebase focused on clarity over production complexity
- **No middleware lock-in**: Works standalone; ROS2/Autoware integration is optional via built-in extension
- **Multi-simulator**: Same code runs on BasicSim, Carla, or Gazebo
- **Rapid iteration**: Hot-reload code and tune parameters without restarting
- **Minimal dependencies**: Core needs only NumPy, Matplotlib, Tkinter
- **Educational**: Numbered modules and clean abstractions for learning

## Installation

**Minimal** (core functionality):
```bash
pip install -r requirements-minimal.txt
```

**Full** (includes joystick support, dev tools, docs):
```bash
pip install -r requirements-full.txt
```

**Optional integrations** (install separately as needed):
- **CARLA**: Install from [CARLA releases](https://github.com/carla-simulator/carla/releases)
- **ROS2 + Autoware**: Install ROS2 (Humble/Iron/Jazzy) and optionally `autoware_auto_msgs` for native Autoware message support. AVLite's `executer_ros` extension provides ROS2 nodes and Autoware message converters out of the box.

Run from source:
```bash
python -m avlite
```

Install system-wide:
```bash
pip install .
```

## Quick Start

1. Launch the visualizer: `python -m avlite`
2. Select a profile from the Config tab (e.g., "default")
3. Click "Start/Stop Stack" to begin execution
4. Right-click on the plot to spawn NPC vehicles
5. Adjust parameters in real-time through the GUI

## Project Structure

AVLite uses a numbered module system for easy navigation:

```
avlite/
├── c10_perception/         # Perception components
│   ├── c11_perception_model.py
│   ├── c12_perception_strategy.py
│   ├── c13_localization_strategy.py
│   ├── c18_hdmap.py
│   └── c19_settings.py
├── c20_planning/           # Planning components
│   ├── c21_planning_model.py
│   ├── c22_global_planning_strategy.py
│   ├── c23_local_planning_strategy.py
│   ├── c24_global_planners.py
│   ├── c26_local_planners.py
│   └── c29_settings.py
├── c30_control/            # Control components
│   ├── c31_control_model.py
│   ├── c32_control_strategy.py
│   ├── c33_pid.py
│   ├── c34_stanley.py
│   └── c39_settings.py
├── c40_execution/          # Execution and simulation
│   ├── c41_execution_model.py
│   ├── c42_factory.py
│   ├── c43_sync_executer.py
│   ├── c44_async_threaded_executer.py
│   ├── c46_basic_sim.py
│   ├── c47_carla_bridge.py
│   ├── c48_gazebo_bridge.py
│   └── c49_settings.py
├── c50_visualization/      # GUI and plotting
│   ├── c51_visualizer_app.py
│   ├── c52_plot_views.py
│   └── c59_settings.py
├── c60_common/            # Utilities
│   ├── c61_setting_utils.py
│   └── c62_capabilities.py
└── extensions/            # Plugin system
    ├── test_ext/
    ├── executer_ros/
    └── multi_object_prediction/
```

The numbering scheme allows quick navigation: search for "c23" to find local planning, "c34" for Stanley controller, etc.

## Developing Custom Plugins

See the [Plugin Development Guide](docs/plugin-development.md) for detailed instructions on creating custom perception, planning, and control components.








