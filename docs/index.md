<p align="center">
  <img src="imgs/logo-black-bg.png" alt="AVLite Logo" width="200">
</p>

# AVLite Documentation

AVLite is a lightweight, extensible autonomous vehicle software stack for rapid prototyping, research, and education. It provides clean abstractions for perception, planning, and control while supporting multiple simulators through a unified interface.

!!! tip "ROS2 & Autoware Ready"
    AVLite includes a built-in ROS2 executor plugin (`p40_executer_ROS2`) with native Autoware message support. Publish and subscribe to `autoware_auto_msgs` types like Trajectory and ControlCommand out of the box.

**Repository**: [github.com/AV-Lab/avlite](https://github.com/AV-Lab/avlite)

## Features

- **Modular Architecture**: Swap perception, localization, planning, and control algorithms at runtime
- **Multi-Simulator Support**: Works with BasicSim (built-in), CARLA, Gazebo, and ROS2
- **ROS2 & Autoware Integration**: Built-in plugin for ROS2 with native Autoware message types
- **Optional Perception & Localization**: Both perception and localization are optional — run with ground truth or plug in your own strategies
- **Real-time Visualization**: Tkinter-based GUI for monitoring and debugging
- **Hot Reloading**: Modify code without restarting the application
- **Plugin System**: Extend functionality with community plugins
- **Profile Management**: Save and load different configurations

## Installation

### Minimal (core only)

```bash
git clone https://github.com/AV-Lab/avlite.git
cd avlite
pip install -r requirements-minimal.txt
```

### Full (includes joystick, dev tools, docs)

```bash
pip install -r requirements-full.txt
```

### Optional Integrations

- **CARLA**: Install from [CARLA releases](https://github.com/carla-simulator/carla/releases)
- **ROS2 + Autoware**: Install ROS2 (Humble/Iron/Jazzy) and optionally `autoware_auto_msgs` for native Autoware message support. The built-in `p40_executer_ROS2` plugin provides:
    - `ROSExecuter`: Synchronize AVLite with ROS2 ecosystem
    - `PlannerNode`: Publishes Autoware Trajectory messages
    - `ControllerNode`: Publishes Autoware ControlCommand messages
    - `PerceptionNode`: Publishes ego state and tracked objects
    - Message converters for seamless Autoware integration

## Quick Start

### 1. Launch the Application

```bash
python -m avlite
```

### 2. Using the GUI

1. **Config Tab**: Select a profile or create a new one
2. **Start Stack**: Click "Start/Stop Stack" to begin simulation
3. **Spawn NPCs**: Right-click on the plot to add vehicles
4. **Adjust Parameters**: Modify settings in real-time through the GUI panels

### 3. Basic Workflow

```
Load Profile → Configure Components → Start Stack → Monitor/Debug → Save Profile
```

### 4. Headless Mode (no GUI)

For deployments on a robot, server, or CI runner, run AVLite without the
Tkinter GUI using the same YAML profiles you saved from the visualizer:

```bash
# Default profile
python -m avlite headless

# Pick a saved profile
python -m avlite headless -p my_robot_profile
python -m avlite headless my_robot_profile        # positional shortcut

# Tune log noise / loop rates
python -m avlite headless -p my_robot_profile \
    --log-level WARNING \
    --control-dt 0.01 --replan-dt 0.5 --perceive
```

A live `rich` dashboard shows FPS, ego state, lap counter, and recent log
lines. Press **Ctrl+C** to stop. Requires `pip install rich`.

!!! tip "Recommended workflow"
    1. **Configure** with `python -m avlite` — pick the bridge, strategies,
       and tune parameters in the GUI.
    2. **Save** the result as a named profile from the Config tab.
    3. **Deploy** with `python -m avlite headless -p <profile>` on your
       robot or server. The same YAML profile drives both modes, so what
       you see in the visualizer is what the robot will run.

## Community Plugins

AVLite has a community plugin system that lets anyone publish perception,
planning, control, executer, or world-bridge strategies as a small Git
repository.

### Browse and install (GUI)

```bash
python -m avlite plugins
```

The browser fetches the official registry from
[avlite-community-plugins](https://github.com/AV-Lab/avlite-community-plugins),
lets you install/uninstall plugins, and (de)registers them with the
active profile. Installed plugins live under
`$XDG_DATA_HOME/avlite/plugins` (or `~/.local/share/avlite/plugins`);
override with the `AVLITE_PLUGINS_DIR` environment variable.

### Publish your plugin

1. Build a plugin following the [Plugin Development Guide](plugin-development.md).
2. Push it to a public Git repository.
3. Fork [avlite-community-plugins](https://github.com/AV-Lab/avlite-community-plugins)
   and add an entry to `plugins.yaml`:

    ```yaml
    plugins:
      - name: my_cool_planner
        repository: https://github.com/<you>/my_cool_planner
        version: latest        # or a tag/commit SHA
        description: One-line summary
        author: Your Name
    ```

4. Open a pull request. Once merged the plugin appears automatically in
   every user's `avlite plugins` browser.

## Core Components

| Component | Description |
|-----------|-------------|
| **c10_perception** | Interfaces + built-in algorithms: `FastBEVLidarDetection`, `KalmanTracker`, `LidarLocalization`; prediction and mapping interfaces |
| **c20_planning** | Global planning (`GlobalCenterlineRacePlanner`, `HDMapGlobalPlanner`) and local planning (`GreedyLatticePlanner`, lattice-based) |
| **c30_control** | Vehicle controllers (Stanley, PID) |
| **c40_execution** | Execution orchestration, `replan_global()`, simulator bridges (BasicSim with 2-D LiDAR, CARLA, Gazebo) |
| **c50_visualization** | Real-time Tkinter GUI with multiple plot views |
| **c60_common** | Settings management, `HDMap` (OpenDRIVE), capability definitions (`AnyOf`), utilities |

## Configuration

AVLite uses YAML-based configuration with **profile support** (multiple named profiles per file, e.g. `default`, `ros`, `perception`).

### Where files live

| Purpose | Location | Override env var |
|---------|----------|------------------|
| **Shipped defaults** (read-only in git) | `{repo}/configs/*.yaml` | — |
| **User profiles** (written on Save) | `~/.config/avlite/*.yaml` | `AVLITE_CONFIG_DIR` |
| **Community plugins** (installed clones) | `~/.local/share/avlite/plugins/` | `AVLITE_PLUGINS_DIR` |
| **Maps & trajectories** | Read: `~/.local/share/avlite/data/` then `{repo}/data/`; save: user dir only | `AVLITE_DATA_DIR` |
| **Log files** (when enabled) | `./logs/` (cwd at runtime) | — |

Paths stored as `data/...` in YAML are resolved against the user data directory first, then the repository `data/` folder. Saved global plans and other writes never go into the repo tree.

User and repo config files share the **same basenames** (`c10_perception.yaml`, `c40_execution.yaml`, `plugin_ros_executer.yaml`, …).

**Load order:** for each settings file, AVLite reads `~/.config/avlite/<name>.yaml` if it exists; otherwise it falls back to `{repo}/configs/<name>.yaml`.

**Save:** GUI and settings window writes go to `~/.config/avlite/` unless **Edit repository configs** is enabled (then `{repo}/configs/`).

The GUI remembers the last selected profile in `~/.config/avlite/startup_profile` and restores it on the next launch.

### Stack config files

- `c10_perception.yaml` — Perception settings
- `c20_planning.yaml` — Planning parameters
- `c30_control.yaml` — Controller tuning
- `c40_execution.yaml` — Execution and simulator settings
- `c50_visualization.yaml` — GUI preferences
- `plugin_*.yaml` — Built-in plugin settings (same names as in repo `configs/`)

### GUI: profiles and reset

- **Config tab** — profile dropdown, Save Config (visualization + execution layers).
- **Settings window** (`T`) — full stack editor, New/Delete/Rename profile, Save.
- **Copy repository configs** (settings window) — copies shipped `{repo}/configs/*.yaml` into `~/.config/avlite/` (overwrite) and reloads. Requires a git clone; plain `pip install` has no bundled configs directory.
- **Edit repository configs** (settings window, dev only) — when enabled, Save/load YAML from `{repo}/configs/` instead of home. Preference stored in `~/.config/avlite/config_target`. Hidden when bundled configs are unavailable.

### CLI validation

```bash
python -m avlite config validate
python -m avlite config validate --profile default
python -m avlite config describe --layer execution
python -m avlite config describe --layer execution --field c40_control_dt
```

Schema field descriptions appear as **tooltips** in the settings window and on main-page controls (dropdowns, Δt fields).

See [Settings naming](settings-naming.md) for key prefixes and validation details.

### Example: Switching Simulators

In the GUI Config tab, change the **Bridge** dropdown:
- `BasicSim` - Built-in 2D simulation (no external dependencies)
- `CarlaBridge` - Connect to a running CARLA simulator (`p40_bridge_carla` plugin)
- `GazeboIgnitionBridge` - Connect to Gazebo Ignition via ROS2 (`p40_bridge_gazebo` plugin)
- `ROS2WorldBridge` - Use a ROS2 topic-based world bridge (`p40_bridge_ROS2` plugin)

## Project Structure

```
avlite/
├── c10_perception/     # Perception interfaces
├── c20_planning/       # Planning algorithms
├── c30_control/        # Control strategies
├── c40_execution/      # Execution and bridges
├── c50_visualization/  # GUI components
├── c60_common/         # Shared utilities
└── plugins/            # Built-in plugins
    ├── p10_perception_MO_prediction/
    ├── p30_controller_joystick/
    ├── p40_bridge_carla/       # CARLA simulator bridge
    ├── p40_bridge_gazebo/      # Gazebo Ignition bridge
    ├── p40_bridge_ROS2/        # ROS2 world bridge
    ├── p40_executer_ROS2/      # ROS2 executor with Autoware msgs
    └── p50_headless_mode/
```

Modules use numbered prefixes (c10, c20, etc.) for easy navigation. Search for "c23" to find local planning, "c34" for Stanley controller, etc.

## Documentation

- [Architecture](architecture.md) - System design and patterns
- [Plugin Development](plugin-development.md) - Create custom components

## Support

- **Issues**: [GitHub Issues](https://github.com/AV-Lab/avlite/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AV-Lab/avlite/discussions)

## License

See the [repository](https://github.com/AV-Lab/avlite) for license information.

