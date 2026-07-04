<p align="center">
  <img src="imgs/logo-black-bg.png" alt="AVLite Logo" width="200">
</p>

# AVLite Documentation

AVLite is a lightweight, extensible autonomous vehicle software stack for rapid prototyping, research, and education. It provides clean abstractions for perception, planning, and control while supporting multiple simulators through a unified interface.

!!! tip "ROS2 & Autoware Ready"
    AVLite integrates optional ROS 2 plugins under `related-repos/` (e.g. `avlite-executer-ROS2`) with native Autoware message support when installed.

**Repository**: [github.com/AV-Lab/avlite](https://github.com/AV-Lab/avlite)

## Features

- **Modular Architecture**: Swap perception, localization, planning, and control algorithms at runtime
- **Multi-Simulator Support**: Works with BasicSim (built-in), CARLA, Gazebo, and ROS2
- **ROS2 & Autoware Integration**: Built-in plugin for ROS2 with native Autoware message types
- **Optional Perception & Localization**: Both perception and localization are optional — run with ground truth or plug in your own strategies
- **Real-time Visualization**: Tkinter-based GUI for monitoring and debugging
- **Hot Reloading**: Modify code without restarting the application
- **Plugin System**: Extend functionality with community and member plugins
- **Multi-robot ready (extensible)**: `AgentType`, per-agent control command mapping, and `WorldBridge.control_agent` / `step()` hooks for future drones, diff-drive, and fleet sims — see [Plugin Development → Multi-robot agents and control](plugin-development.md#7-multi-robot-agents-and-control)
- **Profile Management**: Save and load different configurations

## Installation

### Minimal (core only)

```bash
git clone https://github.com/AV-Lab/avlite.git
cd avlite
pip install -r requirements-minimal.txt
```

### Full 
It includes support for ROS 2, CARLA 5, Autwoware Messages, dev tools, docs, among others.

```bash
pip install -r requirements-full.txt
```

### Optional Integrations

- **CARLA**: Install from [CARLA releases](https://github.com/carla-simulator/carla/releases)
- **ROS2 + Autoware**: Install ROS2 (Humble/Iron/Jazzy) and optionally `autoware_auto_msgs`. Clone `related-repos/avlite-executer-ROS2`, register it in `c52_community_plugins` (`configs/c59_apps.yaml`), and set `c40_executer_type: ROSExecuter` — see [Optional Plugins](optional-plugins.md) and `related-repos/avlite-executer-ROS2/docs/ros2-executer-plugin.md` in the repository.

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
5. **Save Global Plan**: Use ⬇ in the Planning panel to export the current plan as JSON (save dialog opens in `~/.config/avlite/data/`)

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

The **Members** tab in `python -m avlite plugins` lists plugins from the
[avlite-private-plugins](https://github.com/AV-Lab/avlite-private-plugins) registry.
Sign in with GitHub (Device Flow) to browse and install them. Your account must
have access to that repository and to each listed plugin repo.

AVLite stores the OAuth token at `~/.config/avlite/github_oauth.json` (mode `0600`).
Distribution builds must set `AVLITE_GITHUB_OAUTH_CLIENT_ID` to the AV-Lab GitHub
OAuth app client id (Device Flow enabled, `repo` scope).

If you see **403 Forbidden** after sign-in but can open the registry repo in a browser,
authorize the token for AV-Lab SAML SSO (AVLite will offer the authorization link),
then click **Refresh**. If the error mentions OAuth App access restrictions, an
AV-Lab org admin must approve the OAuth app under **Organization Settings →
Third-party access**.

### Publish your plugin

See [Plugin Development — Publish to the community registry](plugin-development.md#10-publish-to-the-community-registry-pull-request) for the full checklist. Summary:

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
| **c50_apps** | App infrastructure: `c51_app_strategy`, `c52_factory`, `c53_plugins`, `c54_settings_schema`, `c55_setting_utils`, `c58_paths`, `c59_settings` |
| **p50_visualizer_tk** | Tk visualizer, settings GUI (`avlite setting`), plugin manager (`avlite plugins`) |
| **c60_common** | Algorithm utilities only (`c61`–`c65`: capabilities, sensor layouts, collision, FPS) |

## Configuration

AVLite uses YAML-based configuration with **profile support** (multiple named profiles per file, e.g. `default`, `ros`, `perception`).

### Where files live

| Purpose | Location | Override env var |
|---------|----------|------------------|
| **Shipped defaults** (read-only in git) | `{repo}/configs/*.yaml` | — |
| **User profiles** (written on Save) | `~/.config/avlite/*.yaml` | `AVLITE_CONFIG_DIR` |
| **Community plugins** (installed clones) | `~/.local/share/avlite/plugins/<name>/` — code only; registered in `c59_apps.yaml` (`c52_community_plugins`) | `AVLITE_PLUGINS_DIR` |
| **Community plugin settings** | `~/.config/avlite/plugin_<name>.yaml` — user-only; no repo default | `AVLITE_CONFIG_DIR` |
| **Maps & trajectories** | Read: `~/.config/avlite/data/` then `{repo}/data/`; save: user dir only (GUI save dialog opens in user data dir) | `AVLITE_DATA_DIR` |
| **Log files** (when enabled) | `./logs/` (cwd at runtime) | — |

Paths stored as `data/...` in YAML are resolved against the user data directory first, then the repository `data/` folder. Saved global plans and other writes never go into the repo tree. In the GUI, **Save Global Plan** (Planning panel ⬇) opens a file picker in `~/.config/avlite/data/` with a timestamped default filename.

User and repo config files share the **same basenames** (`c10_perception.yaml`, `c40_execution.yaml`, `plugin_ros_executer.yaml`, …).

**Load order:** for each settings file, AVLite reads `~/.config/avlite/<name>.yaml` if it exists; otherwise it falls back to `{repo}/configs/<name>.yaml`.

**Save:** GUI and settings window writes go to `~/.config/avlite/` unless **Edit repository configs** is enabled (then `{repo}/configs/`).

The GUI remembers the last selected profile in `~/.config/avlite/startup_profile` and restores it on the next launch.

### Stack config files

- `c10_perception.yaml` — Perception settings
- `c20_planning.yaml` — Planning parameters
- `c30_control.yaml` — Controller tuning
- `c40_execution.yaml` — Execution and simulator settings
- `c59_apps.yaml` — App bootstrap (plugin lists, load gate, GUI profile selection)
- `plugin_p50_visualizer_tk.yaml` — GUI visualization preferences
- `plugin_*.yaml` — Plugin settings: built-in plugins ship repo defaults in `configs/` with user overrides under `~/.config/avlite/`; community plugins use the same `plugin_<name>.yaml` basename but only in the user config dir (one file per registered plugin name)

### GUI: profiles and reset

- **Config tab** — profile dropdown, Save Settings (visualization + execution layers).
- **Settings window** (`T`) — full stack editor, New/Delete/Rename profile, Save, **Export profile**, **Import profile**.
- **Export profile** — reads saved YAML from disk (save first if you have unsaved widget changes); writes a zip with one file per source YAML, each containing only the selected profile key. Includes community plugin configs when referenced in `c59_apps.yaml`. GUI export includes `plugin_p50_visualizer_tk.yaml` via `settings.get_stack_settings_classes()`.
- **Import profile** — merges a profile zip into your config directory; confirms overwrite if the profile name already exists.
- **Edit repository configs** (settings window, dev only) — switches read/write between `~/.config/avlite/` and `{repo}/configs/` (no file copy) and refreshes the profile dropdown from the active target. Preference stored in `~/.config/avlite/config_target`. Hidden when bundled configs are unavailable. Uncheck to return to the user config dir.

### Profile transfer (zip)

Export a profile on one machine and import it on another (e.g. robot with `AVLITE_CONFIG_DIR` or `~/.config/avlite`), then run `python -m avlite headless -p <profile>`.

```bash
python -m avlite config-cli export-profile myprofile [-o myprofile.zip]
python -m avlite config-cli import-profile myprofile.zip [--force]
```

Headless `config-cli export-profile` exports c10–c40 settings, `c59_apps.yaml`, and plugins only (no visualization YAML). Use `avlite setting` or the visualizer settings window to export a profile that includes `plugin_p50_visualizer_tk.yaml`.

Each zip entry is validated against Pydantic schemas on export and import; invalid profiles are rejected with field-level errors (same rules as `config-cli validate`).

### CLI validation

```bash
python -m avlite config-cli validate
python -m avlite config-cli validate --profile default
python -m avlite config-cli export-profile myprofile -o myprofile.zip
python -m avlite config-cli import-profile myprofile.zip --force
python -m avlite config-cli describe --layer execution
python -m avlite config-cli describe --layer execution --field c40_control_dt
```

Schema field descriptions appear as **tooltips** in the settings window and on main-page controls (dropdowns, Δt fields).

See [Settings naming](settings-naming.md) for key prefixes and validation details.

### Example: Switching Simulators

In the GUI Config tab, change the **Bridge** dropdown:
- `BasicSim` - Built-in 2D simulation (no external dependencies)
- `CarlaBridge` - Connect to a running CARLA simulator (`avlite-bridge-carla` plugin)
- `GazeboIgnitionBridge` - Connect to Gazebo Ignition via ROS2 (`avlite-bridge-gazebo` plugin)
- `ROS2WorldBridge` - Use a ROS2 topic-based world bridge (`avlite-bridge-ROS2` plugin)

## Project Structure

```
avlite/
├── c10_perception/     # Perception interfaces
├── c20_planning/       # Planning algorithms
├── c30_control/        # Control strategies
├── c40_execution/      # Execution and bridges
├── c50_apps/  # App infrastructure (AppStrategy, paths)
├── c60_common/         # Shared utilities
└── plugins/            # Built-in apps and stack plugins
    ├── p50_visualizer_tk/   # visualizer + config + plugins Tk apps
    ├── p50_config_cli/
    └── p50_headless_mode/

related-repos/          # Optional plugins (see related-repos/README.md)
    ├── avlite-bridge-carla/
    ├── avlite-bridge-gazebo/
    ├── avlite-bridge-ROS2/
    ├── avlite-controller-joystick/
    └── avlite-executer-ROS2/
```

Modules use numbered prefixes (c10, c20, etc.) for easy navigation. Search for "c23" to find local planning, "c34" for Stanley controller, etc.

## Documentation

- [Architecture](architecture.md) - System design and patterns
- [Algorithms](algorithms.md) - Planning algorithms and lattice parameters
- [Optional Plugins](optional-plugins.md) - Related-repo bridges, ROS executer, joystick
- [Plugin Development](plugin-development.md) - Create custom components

## Support

- **Issues**: [GitHub Issues](https://github.com/AV-Lab/avlite/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AV-Lab/avlite/discussions)

## License

See the [repository](https://github.com/AV-Lab/avlite) for license information.

