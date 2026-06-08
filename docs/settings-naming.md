# Core stack settings naming

Each stack layer has one settings module (`c19`, `c29`, `c39`, `c49`) with a class used as a singleton (`PerceptionSettings`, `PlanningSettings`, `ControlSettings`, `ExecutionSettings`). YAML profiles under `configs/` mirror attribute names exactly.

## Prefix rules

1. **Single consumer module** in the layer package → `c{NN}_{name}`  
   Example: only `c15_perception_algs.py` reads detection params → `c15_detection_z_min`.

2. **Multiple consumer modules** in the same layer package → `c{decade}_{name}`  
   Example: `c26_local_lattice_planners.py` and `c27_lattice.py` both use collision margin → `c20_collision_safety_margin`.

3. **Cross-layer orchestration** → setting lives on the **consuming** layer’s settings class, prefixed by the consumer module.  
   Example: factory fallback race map in `c42_factory.py` → `ExecutionSettings.c42_race_boundary_map`.

4. **Metadata** (`exclude`, `filepath`) is never prefixed.

5. **Redundant subsystem prefixes** are dropped when the module prefix applies: `basic_sim_lidar_range` → `c46_lidar_range`.

## Settings files

| Layer | Module | YAML |
|-------|--------|------|
| Perception | `avlite/c10_perception/c19_settings.py` | `configs/c10_perception.yaml` |
| Planning | `avlite/c20_planning/c29_settings.py` | `configs/c20_planning.yaml` |
| Control | `avlite/c30_control/c39_settings.py` | `configs/c30_control.yaml` |
| Execution | `avlite/c40_execution/c49_settings.py` | `configs/c40_execution.yaml` |

## Extensions

Community and built-in extensions keep `ExtensionSettings` in `settings.py` with unprefixed snake_case parameters. See [Plugin Development](plugin-development.md).

## Migrating saved profiles

After the prefix refactor, run once:

```bash
python scripts/migrate_settings_keys.py
```

Use `--dry-run` to preview changes. Custom YAML profiles outside `configs/` need the same key renames.
