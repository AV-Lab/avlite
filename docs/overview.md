# AVLite Overview

AVLite is a lightweight, extensible autonomous vehicle software stack for rapid prototyping, research, and education. It provides clean abstractions for perception, planning, and control while supporting multiple simulators through a unified interface.

!!! tip "ROS2 & Autoware Ready"
    AVLite supports ROS 2 and Autoware integration through optional world-bridge and executer plugins.

**Repository**: [github.com/AV-Lab/avlite](https://github.com/AV-Lab/avlite)

## Features

- **Modular Architecture**: Swap perception, localization, planning, and control algorithms at runtime
- **Multi-Simulator Support**: Works with BasicSim (built-in), CARLA, Gazebo, and ROS2
- **ROS2 & Autoware Integration**: Optional plugin for ROS2 with native Autoware message types
- **Optional Perception & Localization**: Both perception and localization are optional — run with ground truth or plug in your own strategies
- **Real-time Visualization**: Tkinter-based GUI for monitoring and debugging
- **Vim-style Shortcuts**: Fast, mouse-free control of the visualizer with vim motions (`j`/`k`, `g`/`G`, `Ctrl+u`/`Ctrl+d`)
- **Hot Reloading**: Modify code without restarting the application
- **Plugin System**: Extend functionality with community and member plugins
- **Multi-robot ready (extensible)**: `AgentType`, per-agent control command mapping, and `WorldBridge.control_agent` / `step()` hooks for future drones, diff-drive, and fleet sims — see [Plugin Development → Multi-robot agents and control](plugin-development.md#7-multi-robot-agents-and-control)
- **Profile Management**: Save and load different configurations

## Installation

### From PyPI (recommended)

```bash
pip install avlite
```

Requires Python 3.10+.

### From source (development)

Clone the repository to hack on AVLite or run the test suite:

```bash
git clone https://github.com/AV-Lab/avlite.git
cd avlite
pip install -e ".[dev]"
```

The editable install adds the optional development dependencies (pytest, coverage). See the [Quick Start](quick-start.md) for a guided first run.

### Optional Integrations

- **CARLA**: Install from [CARLA releases](https://github.com/carla-simulator/carla/releases), then enable the CARLA world-bridge plugin.
- **Gazebo**: Install ROS 2 and enable the Gazebo world-bridge plugin.
- **ROS2 + Autoware**: Install ROS2 (Humble/Iron/Jazzy) and optionally `autoware_auto_msgs`, then enable the ROS2 world-bridge or executer plugin (register it in the `c62_community_plugins` map in the `c69_apps` section of `configs/<profile>.yaml`, and set `c40_executer_type` to the ROS executer when using it).

## Quick Start

Launch the visualizer and drive the built-in simulator:

```bash
python -m avlite
```

In the GUI, select a profile in the **Config** tab, click **Start/Stop Stack**,
right-click the plot to spawn NPCs, and tune parameters live. For a full
guided walkthrough (install, GUI basics, and headless deployment), see the
[Quick Start guide](quick-start.md).

### Headless Mode (no GUI)

For deployments on a robot, server, or CI runner, run AVLite without the
Tkinter GUI using the same YAML profiles you saved from the visualizer:

```bash
python -m avlite headless -p my_robot_profile
```

A live `rich` dashboard shows FPS, ego state, lap counter, and recent log
lines. Press **Ctrl+C** to stop. Requires `pip install rich`. See the
[Quick Start guide](quick-start.md#4-save-a-profile-and-run-headless) for all
options.

## Community Plugins

AVLite has a community plugin system that lets anyone publish perception,
planning, control, executer, or world-bridge strategies as a small Git
repository. Community and member plugins are third-party or unverified code;
AV-Lab does not guarantee their safety. Use for research and development at
your own risk.

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

### Member plugins

The **Members** tab in `python -m avlite plugins` lists plugins from the AV-Lab
private registry. Sign in with GitHub (Device Flow) to browse and install them;
your account must have access to that registry and to each listed plugin repo.
AVLite stores the OAuth token under `~/.config/avlite/` (mode `0600`).

### Publish your plugin

See [Plugin Development — Publish to the community registry](plugin-development.md#11-publish-to-the-community-registry-pull-request) for the full checklist. Summary:

1. Build and test locally ([Plugin Development Guide](plugin-development.md)).
2. Push your plugin to a **public** Git repository.
3. Fork [avlite-community-plugins](https://github.com/AV-Lab/avlite-community-plugins)
   and add an entry to `plugins.yaml`:

    ```yaml
    plugins:
      - name: my_perception_plugin
        description: One-line summary of what the plugin does
        repository: https://github.com/your-org/your-plugin-repo
        version: latest        # or a tag/commit SHA
        author: your-org
        category:
          - PerceptionStrategy
    ```

4. Open a pull request. Once merged, the plugin appears in every user's
   `python -m avlite plugins` browser for install and register.

## Core Components

| Component | Description |
|-----------|-------------|
| **c10_perception** | Interfaces + built-in algorithms; `Map` / `RaceMap` / `HDMap` (c11), OpenDRIVE parser (c18) |
| **c20_planning** | Global planning (`GlobalCenterlineRacePlanner`, `HDMapGlobalPlanner`) and local planning (`VelocityLocalPlanner`, `GreedyLatticePlanner`, lattice-based) |
| **c30_control** | Vehicle controllers (Stanley, PID) |
| **c40_execution** | Execution orchestration, `replan_global()`, simulator bridges (BasicSim, CARLA, Gazebo) |
| **c60_apps** | App infrastructure: `c61_app_strategy`, `c62_factory`, `c63_plugins`, `c64_settings_schema`, `c65_setting_utils`, `c68_paths`, `c69_settings` |
| **p60_visualizer_tk** | Tk visualizer, settings GUI (`avlite setting`), plugin manager (`avlite plugins`) |
| **c50_common** | Algorithm utilities only (`c51`–`c55`: capabilities, sensor layouts, collision, FPS) |

## Configuration

AVLite uses YAML-based configuration with **profile support** (multiple named profiles per file, e.g. `default`, `ros`, `perception`).

### Where files live

| Purpose | Location | Override env var |
|---------|----------|------------------|
| **Shipped defaults** (read-only in git) | `{repo}/configs/*.yaml` | — |
| **User profiles** (written on Save) | `~/.config/avlite/*.yaml` | `AVLITE_CONFIG_DIR` |
| **Community plugins** (installed clones) | `~/.local/share/avlite/plugins/<name>/` — code only; registered in the `c69_apps` section (`c62_community_plugins`) of the active profile | `AVLITE_PLUGINS_DIR` |
| **Community plugin settings** | `~/.config/avlite/plugin_<name>.yaml` — user-only; no repo default | `AVLITE_CONFIG_DIR` |
| **Maps & trajectories** | Read: `~/.config/avlite/data/` then the bundled `avlite/data/` (shipped with the package); save: user dir only (GUI save dialog opens in user data dir) | `AVLITE_DATA_DIR` |
| **Log files** (when enabled) | `./logs/` (cwd at runtime) | — |

Paths stored as `data/...` in YAML are resolved against the user data directory first, then the bundled `avlite/data/` folder shipped with the package. Saved global plans and other writes never go into the repo tree. In the GUI, **Save Global Plan** (Planning panel ⬇) opens a file picker in `~/.config/avlite/data/` with a timestamped default filename.

User and repo config files share the **same basenames** (`c10_perception.yaml`, `c40_execution.yaml`, `plugin_ros_executer.yaml`, …).

**Load order:** for each settings file, AVLite reads `~/.config/avlite/<name>.yaml` if it exists; otherwise it falls back to `{repo}/configs/<name>.yaml`.

**Save:** GUI and settings window writes go to `~/.config/avlite/` unless **Edit repository configs** is enabled (then `{repo}/configs/`).

The GUI remembers the last selected profile in `~/.config/avlite/startup_profile` and restores it on the next launch.

### Config files (one per profile)

Each profile is a single `configs/<profile>.yaml` file (e.g. `configs/default.yaml`) whose top-level keys are sections:

- `c10_perception` — Perception settings
- `c20_planning` — Planning parameters
- `c30_control` — Controller tuning
- `c40_execution` — Execution and simulator settings
- `c69_apps` — App bootstrap (plugin lists, load gate, GUI profile selection)
- `plugins:` — a mapping of plugin directory name to that plugin's settings (e.g. `p60_visualizer_tk`, `p60_headless_mode`, and any community plugins)

Repo defaults ship in `{repo}/configs/`; user overrides live under `~/.config/avlite/` (or `AVLITE_CONFIG_DIR`). Reads prefer the user file and fall back to the repo copy; writes go to the user directory.

### GUI: profiles and reset

- **Config tab** — profile dropdown, Save Settings (visualization + execution layers).
- **Settings window** (`T`) — full stack editor, New/Delete/Rename profile, Save, **Export profile**, **Import profile**. The `default` profile cannot be deleted or renamed. The built-in app plugin hosting the open settings window (typically `p60_visualizer_tk`) cannot be removed from the profile plugin list while that window is running.
- **Export profile** — reads the saved profile file from disk (save first if you have unsaved widget changes) and writes a single `<profile>.yaml`. Three checkboxes control inclusion: **Stack settings** (the four core layer sections), **App settings** (the `c69_apps` section), and **Plugin settings** (the `plugins` section); all are included by default.
- **Import profile** — merges a profile `.yaml` into your config directory (profile name = file stem); confirms overwrite if it already exists.
- **Edit repository configs** (settings window, dev only) — switches read/write between `~/.config/avlite/` and `{repo}/configs/` (no file copy) and refreshes the profile dropdown from the active target. Preference stored in `~/.config/avlite/config_target`. Hidden when bundled configs are unavailable. Uncheck to return to the user config dir.

### Profile transfer

Export a profile on one machine and import it on another (e.g. robot with `AVLITE_CONFIG_DIR` or `~/.config/avlite`), then run `python -m avlite headless -p <profile>`.

```bash
python -m avlite setting-cli export-profile myprofile [-o myprofile.yaml] [--no-stack] [--no-app] [--no-plugins]
python -m avlite setting-cli import-profile myprofile.yaml [--force]
```

Known sections (core layers, `c69_apps`, and built-in plugins) are validated against their Pydantic schemas on import; invalid profiles are rejected with field-level errors (same rules as `setting-cli validate`).

### CLI validation

```bash
python -m avlite setting-cli validate
python -m avlite setting-cli validate --profile default
python -m avlite setting-cli export-profile myprofile -o myprofile.yaml
python -m avlite setting-cli import-profile myprofile.yaml --force
python -m avlite setting-cli describe --layer execution
python -m avlite setting-cli describe --layer execution --field c40_control_dt
```

Schema field descriptions appear as **tooltips** in the settings window and on main-page controls (dropdowns, Δt fields).

See [Settings naming](settings-naming.md) for key prefixes and validation details.

### Example: Switching Simulators

In the GUI Config tab, change the **Bridge** dropdown:
- `BasicSim` - Built-in 2D simulation (no external dependencies)
- `CarlaBridge` - Connect to a running CARLA simulator (optional world-bridge plugin)
- `GazeboIgnitionBridge` - Connect to Gazebo Ignition via ROS2 (optional world-bridge plugin)
- `ROS2WorldBridge` - Use a ROS2 topic-based world bridge (optional world-bridge plugin)

## Project Structure

```
avlite/
├── c10_perception/     # Perception interfaces
├── c20_planning/       # Planning algorithms
├── c30_control/        # Control strategies
├── c40_execution/      # Execution and bridges
├── c60_apps/  # App infrastructure (AppStrategy, paths)
├── c50_common/         # Shared utilities
└── plugins/            # Built-in apps and stack plugins
    ├── p60_visualizer_tk/   # visualizer + config + plugins Tk apps
    ├── p60_setting_cli/
    └── p60_headless_mode/
```

Modules use numbered prefixes (c10, c20, etc.) for easy navigation. Search for "c23" to find local planning, "c34" for Stanley controller, etc.

## Documentation

- [Quick Start](quick-start.md) - Install, launch, and run headless
- [Architecture](architecture.md) - System design and patterns
- [Algorithms](algorithms.md) - Planning algorithms and lattice parameters
- [Plugin Development](plugin-development.md) - Create custom components

## Support

- **Issues**: [GitHub Issues](https://github.com/AV-Lab/avlite/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AV-Lab/avlite/discussions)

## License

See the [repository](https://github.com/AV-Lab/avlite) for license information.
