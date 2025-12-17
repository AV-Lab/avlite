# Architecture

## Overview

AVLite follows a layered architecture with clear separation between interfaces and implementations.

```
┌─────────────────────────────────────────────────────────────┐
│                    Visualization (c50)                      │
│                  Real-time Tkinter GUI                      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Execution Layer (c40)                    │
│         SyncExecuter / AsyncThreadedExecuter                │
│                    Factory Pattern                          │
└─────────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Perception  │ │   Planning   │ │   Control    │ │    World     │
│    (c10)     │ │    (c20)     │ │    (c30)     │ │   Bridge     │
│              │ │              │ │              │ │              │
│  Interfaces  │ │  Global +    │ │  Stanley     │ │  BasicSim    │
│  + Registry  │ │  Local       │ │  PID         │ │  Carla       │
│              │ │  Lattice     │ │              │ │  Gazebo      │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
         │              │              │              │
         └──────────────┴──────────────┴──────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                      Common (c60)                           │
│        Settings, Capabilities, Utilities                    │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│              Extensions + Community Plugins                 │
│     Perception, Planning, Control implementations           │
└─────────────────────────────────────────────────────────────┘
```

## Design Patterns

### Strategy Pattern with Auto-Registration

All major components use abstract base classes with automatic registration:

```python
class PerceptionStrategy(ABC):
    registry = {}
    
    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:
            PerceptionStrategy.registry[cls.__name__] = cls
```

When you create a subclass, it automatically registers itself and appears in the UI dropdowns. No manual registration needed.

### Capability System

Components declare what they require and provide:

```python
class MyPerception(PerceptionStrategy):
    @property
    def requirements(self) -> set[WorldCapability]:
        # What I need from the world/simulator
        return {WorldCapability.RGB_IMAGE}
    
    @property
    def capabilities(self) -> set[PerceptionCapability]:
        # What I provide
        return {PerceptionCapability.DETECTION}
```

**World Capabilities** (what simulators can provide):

- `GT_DETECTION` - Ground truth object detection
- `GT_TRACKING` - Ground truth tracking IDs
- `GT_LOCALIZATION` - Ground truth ego pose
- `RGB_IMAGE` - Camera images
- `DEPTH_IMAGE` - Depth maps
- `LIDAR` - Point cloud data

**Perception Capabilities** (what perception strategies provide):

- `DETECTION` - Object detection
- `TRACKING` - Object tracking
- `PREDICTION` - Motion prediction

### Factory Pattern

The executor factory assembles components based on configuration:

```python
executer = executor_factory(
    bridge="BasicSim",
    perception="MultiObjectPredictor",
    local_planner="GreedyLatticePlanner",
    controller="StanleyController"
)
```

It loads extensions, instantiates strategies from registries, and wires everything together.

## Core Modules

### c10_perception

Provides **interfaces** for:
- `PerceptionStrategy` - Detection, tracking, prediction
- `LocalizationStrategy` - Ego localization
- `MappingStrategy` - Environment mapping
- `HDMap` - OpenDRIVE map parsing and routing

Implementations come from extensions/plugins.

### c20_planning

- `GlobalPlannerStrategy` - Route planning (A*, HD map routing)
- `LocalPlannerStrategy` - Reactive planning
- `Lattice` - Frenet frame lattice for local planning
- `Trajectory` - Path + velocity profile

Includes built-in planners: `RaceGlobalPlanner`, `HDMapGlobalPlanner`, `GreedyLatticePlanner`.

### c30_control

- `ControlStrategy` - Vehicle control interface
- `ControlCommand` - Throttle/brake + steering output

Includes built-in controllers: `StanleyController`, `PIDController`.

### c40_execution

- `Executer` - Main execution loop (sync/async variants)
- `WorldBridge` - Simulator abstraction
- `Factory` - Component assembly

Built-in bridges: `BasicSim`, `CarlaBridge`, `GazeboBridge`.

### c50_visualization

Tkinter-based GUI with:
- Real-time plotting (XY, Frenet views)
- Component configuration
- Profile management
- Log viewer
- Extension settings

### c60_common

- Settings load/save (YAML profiles)
- Hot reloading
- Extension discovery
- Capability enums

## Data Flow

```
World Bridge
    │
    ├─► Ego State ─────────────────────────────┐
    │                                          │
    ├─► Sensor Data ──► Perception ──► Agents  │
    │                                          ▼
    │                              Local Planner
    │                                          │
    │                                          ▼
    │                              Trajectory
    │                                          │
    │                                          ▼
    │                              Controller
    │                                          │
    └─────────────── Control Command ◄─────────┘
```

1. **World Bridge** provides ego state and sensor data
2. **Perception** (optional) detects/tracks/predicts agents
3. **Local Planner** generates trajectory avoiding obstacles
4. **Controller** computes steering and throttle
5. **World Bridge** executes control command

## Extension System

```
avlite/
└── extensions/           # Built-in (core team)
    ├── multi_object_prediction/
    ├── executer_ros/
    └── test_ext/

/path/to/                 # Community plugins
└── my_plugin/
    ├── __init__.py
    ├── settings.py
    └── ...
```

Extensions are loaded at startup. Classes inheriting from base strategies auto-register.

See [Plugin Development](plugin-development.md) for creating community plugins.
