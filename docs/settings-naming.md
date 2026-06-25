# Core stack settings naming

Each stack layer has one settings module (`c19`, `c29`, `c39`, `c49`) with a class used as a singleton (`PerceptionSettings`, `PlanningSettings`, `ControlSettings`, `ExecutionSettings`). YAML profiles mirror attribute names exactly.

Shipped defaults live in the repository `configs/` directory. When you save from the GUI or settings window, profiles are written to `~/.config/avlite/` using the **same filenames**. On load, each file is read from the user directory if present, otherwise from the repo. Override the user directory with `AVLITE_CONFIG_DIR` (YAML files sit directly in that path, not in a nested `configs/` folder). Enable **Edit repository configs** in the settings window (git clone only) to switch read/write to `{repo}/configs/` instead of the user dir.

## Prefix rules

1. **Single consumer module** in the layer package → `c{NN}_{name}`  
   Example: only `c15_perception_algs.py` reads detection params → `c15_detection_z_min`.

2. **Multiple consumer modules** in the same layer package → `c{decade}_{name}`  
   Example: `c26_local_lattice_planners.py` and `c27_lattice.py` both use collision margin → `c20_collision_safety_margin`.

3. **Cross-layer orchestration** → setting lives on the **consuming** layer’s settings class, prefixed by the consumer module.  
   Example: factory fallback race map in `c43_factory.py` → `ExecutionSettings.c43_race_boundary_map`.

4. **Metadata** (`exclude`, `filepath`) is never prefixed.

5. **Redundant subsystem prefixes** are dropped when the module prefix applies: `basic_sim_lidar_range` → `c46_lidar_range`.

## Settings files

| Layer | Module | Repo default | User override (on Save) |
|-------|--------|--------------|-------------------------|
| Perception | `avlite/c10_perception/c19_settings.py` | `configs/c10_perception.yaml` | `~/.config/avlite/c10_perception.yaml` |
| Planning | `avlite/c20_planning/c29_settings.py` | `configs/c20_planning.yaml` | `~/.config/avlite/c20_planning.yaml` |
| Control | `avlite/c30_control/c39_settings.py` | `configs/c30_control.yaml` | `~/.config/avlite/c30_control.yaml` |
| Execution | `avlite/c40_execution/c49_settings.py` | `configs/c40_execution.yaml` | `~/.config/avlite/c40_execution.yaml` |
| Visualization | `avlite/c50_visualization/c59_settings.py` | `configs/c50_visualization.yaml` | `~/.config/avlite/c50_visualization.yaml` |

Built-in plugins use `configs/plugin_*.yaml` in the repo and the same basename under `~/.config/avlite/` when saved.

## Plugins

Community and built-in plugins keep `PluginSettings` in `settings.py` with unprefixed snake_case parameters. See [Plugin Development](plugin-development.md).

## Validation and field docs

Each settings module defines a Pydantic `*SettingsSchema` with types, defaults, and `Field(description=...)`. YAML profiles are validated on load/save and on profile zip export/import.

```bash
python -m avlite config help
python -m avlite config validate              # check all profiles
python -m avlite config validate --profile default
python -m avlite config export-profile myprofile [-o myprofile.zip]
python -m avlite config import-profile myprofile.zip [--force]
python -m avlite config describe --layer execution
python -m avlite config describe --layer execution --field c40_control_dt
```

Hover a field in the settings window (`T`) or on main-page controls to see its schema description (type and default in parentheses).
