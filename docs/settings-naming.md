# Core stack settings naming

Each stack layer has one settings module (`c19`, `c29`, `c39`, `c49`) with a class used as a singleton (`PerceptionSettings`, `PlanningSettings`, `ControlSettings`, `ExecutionSettings`). YAML profiles mirror attribute names exactly.

Each profile is a single `configs/<profile>.yaml` file whose top-level keys are sections: one per core layer (`c10_perception`, `c20_planning`, `c30_control`, `c40_execution`), an app-bootstrap section (`c69_apps`), and a `plugins:` mapping keyed by plugin directory name. Shipped defaults live in the repository `configs/` directory. When you save from the GUI or settings window, the profile is written to `~/.config/avlite/` using the **same filename**. On load, each profile file is read from the user directory if present, otherwise from the repo. Override the user directory with `AVLITE_CONFIG_DIR` (YAML files sit directly in that path, not in a nested `configs/` folder). Enable **Edit repository configs** in the settings window (git clone only) to switch read/write to `{repo}/configs/` instead of the user dir.

## Prefix rules

1. **Single consumer module** in the layer package → `c{NN}_{name}`  
   Example: only `c15_perception_algs.py` reads detection params → `c15_detection_z_min`.

2. **Multiple consumer modules** in the same layer package → `c{decade}_{name}`  
   Example: `c28_local_lattice_planners.py` and `c27_local_behavioral_and_velocity_planners.py` both use collision margin → `c20_collision_safety_margin`.

3. **Cross-layer orchestration** → setting lives on the **consuming** layer’s settings class, prefixed by the consumer module.  
   Example: default map path in `c62_factory.py` → `ExecutionSettings.c40_map`.  
   App bootstrap lives on `AppSettings` in c60: plugin lists and load gate use `c62_*` (consumer `c62_factory`); active profile selection uses `c60_selected_profile`.

4. **Built-in Tk plugin (`p60_visualizer_tk`)** — prefix identifies the **consumer module**, not the settings file:
   - Single consumer → `p{NN}_{name}` (e.g. `p68_log_font` → `p68_log_view.py`)
   - Multiple modules in the package → `p60_{name}` (e.g. `p60_dark_mode` → p61, p62, p64)

5. **Metadata** (`exclude`, `filepath`) is never prefixed.

6. **Redundant subsystem prefixes** are dropped when the module prefix applies: `basic_sim_lidar_range` → `c46_lidar_range`.

## Settings sections

All sections below live in the single per-profile file `configs/<profile>.yaml` (repo default) and `~/.config/avlite/<profile>.yaml` (user override on Save).

| Layer | Module | Profile section |
|-------|--------|-----------------|
| Perception | `avlite/c10_perception/c19_settings.py` | `c10_perception` |
| Planning | `avlite/c20_planning/c29_settings.py` | `c20_planning` |
| Control | `avlite/c30_control/c39_settings.py` | `c30_control` |
| Execution | `avlite/c40_execution/c49_settings.py` | `c40_execution` |
| Apps (bootstrap) | `avlite/c60_apps/c69_settings.py` (schema only; no Tk) | `c69_apps` |
| Visualization | `avlite/plugins/p60_visualizer_tk/settings.py` (`PluginSettingsSchema`) | `plugins.p60_visualizer_tk` |

Notable execution fields:

| Field | Meaning |
|-------|---------|
| `c40_execution_tasks` | List of `TaskStrategy` class names appended after each stack tick; empty disables tasks. See [Execution Tasks](execution-tasks.md). |

Every plugin (built-in or community) stores its settings under the profile's `plugins:` mapping, keyed by the plugin's directory name.

## Stack load and export

| API | Module | Includes viz section? |
|-----|--------|------------------------|
| `load_stack_settings()` | `c62_factory` | No (GUI loads `VisualizationSettings` separately) |
| `get_stack_settings_classes()` | `c62_factory` | No (c10–c40 + `AppSettings` + plugins) |

Headless `python -m avlite setting-cli describe --layer` accepts perception, planning, control, and execution (not visualization or app bootstrap).

## Plugins

Community and built-in plugins keep `PluginSettings` in `settings.py` with unprefixed snake_case parameters. AVLite sets `filepath` automatically for community plugins at registration/load time. See [Plugin Development](plugin-development.md).

| Kind | Settings module | Profile section |
|------|-----------------|-----------------|
| Community plugin | `<plugin>/settings.py` | `plugins.<name>` (user config only) |
| Built-in plugin | `avlite/plugins/<name>/settings.py` | `plugins.<name>` (repo default + user override) |

## Validation and field docs

Each settings module defines a Pydantic `*SettingsSchema` with types, defaults, and `Field(description=...)`. YAML profile sections are validated on load/save and on profile export/import.

```bash
python -m avlite setting-cli help
python -m avlite setting-cli validate              # check all profiles
python -m avlite setting-cli validate --profile default
python -m avlite setting-cli export-profile myprofile [-o myprofile.yaml] [--no-stack] [--no-app] [--no-plugins]
python -m avlite setting-cli import-profile myprofile.yaml [--force]
python -m avlite setting-cli describe --layer execution
python -m avlite setting-cli describe --layer execution --field c40_control_dt
```

Hover a field in the settings window (`T`) or on main-page controls to see its schema description (type and default in parentheses).
