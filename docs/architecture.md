# Architecture

## Overview

AVLite follows a layered architecture with clear separation between interfaces and implementations.

```mermaid
flowchart TB
    subgraph ENTRY[" "]
        direction LR
        VIZ["Visualization\nReal-time Tkinter GUI"]
        HL["Headless Mode\nTerminal dashboard"]
        VIZ ~~~ HL
    end

    EXEC["Execution\nSync/async executer and factory"]

    subgraph COMPONENTS[" "]
        direction LR
        PERC["Perception (optional)\nLocalization · Mapping\nDetection · Tracking · Prediction"]
        PLAN["Planning\nGlobal · Local · Lattice"]
        CTRL["Control\nStanley · PID"]
        WB["World Bridge\nBasicSim · Carla · Gazebo · ROS2"]
        PERC ~~~ PLAN ~~~ CTRL ~~~ WB
    end

    COMMON["Common\nSettings · Capabilities · Trajectories · Collision checking"]

    ENTRY --> EXEC
    EXEC --> COMPONENTS
    COMPONENTS --> COMMON
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
        return {WorldCapability.CAMERA_RGB}
    
    @property
    def capabilities(self) -> set[PerceptionCapability]:
        # What I provide
        return {PerceptionCapability.DETECTION}
```

**World Capabilities** (what simulators can provide):

- `GT_DETECTION` - Ground truth object detection
- `GT_TRACKING` - Ground truth tracking IDs
- `GT_LOCALIZATION` - Ground truth ego pose
- `CAMERA_RGB` - RGB camera images
- `CAMERA_DEPTH` - Depth camera images
- `LIDAR_3D` - 3D LiDAR point cloud data
- `LIDAR_2D` - 2D LiDAR scanner data
- `RADAR` - Radar sensor data
- `WHEEL_ENCODER` - Wheel encoder for odometry
- `IMU` - Inertial measurement unit
- `GNSS` - GNSS / GPS receiver

**Perception Capabilities** (what perception strategies provide):

- `DETECTION` - Object detection
- `TRACKING` - Object tracking
- `PREDICTION` - Motion prediction

**Localization Capabilities** (what localization strategies provide):

- `LOCALIZATION_2D` - 2D pose estimation (x, y)
- `LOCALIZATION_3D` - 3D pose estimation (x, y, z)
- `LOCALIZATION_HEADING` - Heading / yaw estimation
- `LOCALIZATION_HEADING_3D` - Full 3D orientation (roll, pitch, yaw)
- `VELOCITY` - Velocity estimation

**Mapping Capabilities** (what mapping strategies provide):

- `OCCUPANCY_GRID` - Occupancy grid mapping
- `PATH_BOUNDARY` - Path boundary extraction
- `OPENDRIVE_HDMAP` - OpenDRIVE HD map integration

### Factory Pattern

The executor factory assembles components based on configuration:

```python
executer = executor_factory(
    bridge="BasicSim",
    perception_strategy_name="MultiObjectPredictor",
    localization_strategy_name="MyLocalization",
    local_planner_strategy_name="GreedyLatticePlanner",
    controller_strategy_name="StanleyController"
)
```

It loads plugins, instantiates strategies from registries, and wires everything together. Both `perception_strategy_name` and `localization_strategy_name` are optional — pass an empty string or omit them to run without that component.

Before calling `executor_factory()`, load YAML profiles with `load_stack_settings(profile, load_plugins)` in [`c43_factory.py`](../avlite/c40_execution/c43_factory.py). That loads c10–c40 settings and built-in plugin settings; it does **not** load `c50_visualization.yaml` (the GUI loads the Tk `VisualizationSettings` binder separately).

### Layer import rules

`c40_execution`, `c60_common`, and `avlite/plugins` must not import `c50_visualization`. Profile zip export that includes visualization YAML is composed in c50 via `c59_settings.get_stack_settings_classes()`, which wraps the core list from c43.

## Layers

### **Perception**

Optional monolithic or pipelined detect/track/predict strategies, plus localization and mapping interfaces. Built-in algorithms and plugin implementations register automatically and appear in UI dropdowns. Static map types (`Map`, `RaceMap`) live in c11; OpenDRIVE `HDMap` parsing is in c18. See [Plugin Development](plugin-development.md) for monolithic vs pipeline extension paths.

### **Planning**

Global route planning and reactive local planning (lattice-based). Produces trajectories for the controller.

### **Control**

Vehicle control strategies (Stanley, PID) output throttle/brake and steering.

### **Execution**

World bridge (simulator/ROS interface), executer orchestration loop, sync/async scheduling, and the factory that wires the stack from YAML configuration. Built-in bridges include BasicSim; CARLA, Gazebo, and ROS2 bridges ship as optional plugins under `related-repos/`. For a multiprocess ROS deployment with worker nodes and Autoware topics, use `c40_executer_type: ROSExecuter` with `avlite-executer-ROS2` — see [Optional Plugins](optional-plugins.md).

### **Visualization**

Tkinter GUI: real-time plots, profile/config management, schema tooltips, thread-safe log filtering (Core / Plugins / per-layer toggles), and plugin settings.

### **Common**

YAML profile load/save, hot reload, plugin discovery (`c66_plugins`), path resolution (`c67_paths`), capability enums, canonical sensor layouts (rgb, depth, lidar, imu, gnss between bridge and perception), collision checking, and settings validation.

## Data Flow

```
World Bridge → SensorFrame → Localization / Perception → PerceptionModel → Planning → Control
```

```
World Bridge
    │
    ├─► Sensor Data ──► Localization ──► Ego Pose (updated in-place)
    │                                          │
    ├─► Sensor Data ──► Perception ───► Agents │
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

1. **World Bridge** provides sensor data (IMU, LiDAR, camera, ground truth)
2. **Localization** (optional) estimates the ego pose from sensor data, updating `PerceptionModel.ego_vehicle` in-place
3. **Perception** (optional) detects/tracks/predicts surrounding agents
4. **Local Planner** generates trajectory avoiding obstacles
5. **Controller** computes steering and throttle
6. **World Bridge** executes control command

## Plugin System

```
avlite/
└── plugins/              # Built-in (core team)
    └── p50_headless_mode/

related-repos/            # Optional plugins (bridges, ROS executer, joystick)
    ├── avlite-bridge-carla/
    ├── avlite-bridge-gazebo/
    ├── avlite-bridge-ROS2/
    ├── avlite-controller-joystick/
    └── avlite-executer-ROS2/

~/.local/share/avlite/plugins/   # Community (installed)
└── my_plugin/
    ├── __init__.py
    ├── settings.py
    └── ...

~/.config/avlite/plugin_my_plugin.yaml   # Community plugin settings (user config, not in install dir)
```

Plugins are loaded at startup. Classes inheriting from base strategies auto-register.

See [Plugin Development](plugin-development.md) for creating community plugins, pNx naming, and log filtering.
