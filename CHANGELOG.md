# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-07-04

### Added
- `StackCapability` — unified enum for stack-produced/consumed capabilities (`DETECTION`, `TRACKING`, `PREDICTION`, `LOCAL_PLAN`, `GLOBAL_PLAN`, `CONTROL`, `LOCALIZATION`, `MAP`, `SLAM`), replacing the per-module `PerceptionCapability` / `LocalizationCapability` / `MappingCapability` enums
- Consistent 2×2 capability contract: every strategy declares `world_requirements` (sensors) and `stack_requirements` (upstream modules) and advertises `stack_capabilities`; world bridges expose `world_capabilities` plus an optional `stack_capabilities` for supplying ground truth
- Runtime enforcement of `stack_requirements` in the executer: modules whose upstream dependencies are unmet are warned at assembly and their steps are gated
- Ego actuation now requires an available pose source: when `LOCALIZATION` is unavailable (no localization strategy and no ground-truth localization from the world) both sync and async executers halt ego control so the vehicle does not move (`Executer._can_actuate()`)
- Bridge settings: scrollable, interactive checklist of the bridge's providable world/ground-truth capabilities that controls which data is fed to the stack, backed by `ExecutionSettings.c41_provided` and `is_capability_provided()` / `provided_capability_names()` helpers
- `scripts/migrate_configs.py` — one-time migration from the old per-layer/per-plugin YAML files to the new per-profile `configs/<profile>.yaml` layout, applying field renames (`c52_*` → `c62_*`, `c50_selected_profile` → `c60_selected_profile`, `p5x_*` → `p6x_*`)
- `c65_setting_utils.section_key()` / `profile_file_path()` and file-level `delete_profile()` / `rename_profile()` for the single-file-per-profile model
- `setting-cli export-profile --no-app` / `--no-plugins` flags and Export dialog checkboxes (include app settings / include plugin settings)
- `AppStrategy` registry in `c50_apps/c51_app_strategy.py` — pluggable CLI/GUI entry points (`setting`, `config-cli`, `plugins`, `headless`, default visualizer)
- Standalone settings GUI: `python -m avlite setting` (no visualizer panels)
- `p50_config_cli` built-in plugin — terminal profile validate/describe/import/export (`config-cli` subcommand)
- Community plugin import infrastructure: `sync_community_plugins()`, `CommunityPluginFinder` meta-path hook, and dashed-name → Python import mapping (`plugin_import_name`)
- `PluginPaths.repo_root()` and repo-relative community plugin path resolve/shorten
- `Executer.apply_global_plan()` — shared helper to push a global plan to the local planner and controller
- `sync_perception_pipeline_from_c19()` — keep visualization perception settings aligned with the active c19 profile after stack load
- `VisualizerApp.on_community_plugins_changed()` — reload stack and refresh UI after community plugin install/uninstall
- Clean visualizer shutdown (`WM_DELETE_WINDOW` stops execution before destroy); stop execution on profile switch
- `load_boundary_segments()` and `boundary_segments_from_global_plan()` helpers in `c46_basic_sim.py`
- `test/c60_apps/` — factory smoke, plugin settings, schema validation, log routing
- `test/c50_common/test_c50_import_boundary.py` — stack core (`c10`–`c40`, `c50_common`) must not import disallowed `c60_apps` or `p60_*` apps
- README, architecture, and plugin-development docs for optional community plugins (`avlite-bridge-*`, `avlite-executer-ROS2`, `avlite-controller-joystick`)

### Changed
- **Breaking:** Capability API renamed for symmetry — strategy `requirements` → `world_requirements`, strategy `capabilities` → `stack_capabilities`; `WorldBridge.capabilities` → `world_capabilities` (update any custom strategies/bridges)
- Ground truth is now supplied through the world bridge's `stack_capabilities` (e.g. `BasicSim` provides `DETECTION`/`TRACKING`/`LOCALIZATION`) instead of dedicated `GT_*` world capabilities, and is filtered by the Bridge checklist
- **Breaking:** Renamed terminal CLI `config-cli` → `setting-cli` and built-in plugin `p60_config_cli` → `p60_setting_cli` (pairs with GUI `avlite setting`)
- **Breaking:** Profile export supports optional stack layer sections via `include_stack` / `--no-stack` and a **Stack settings** checkbox in the Export dialog (alongside app and plugin toggles)
- **Breaking:** Renamed `c50_apps` → `c60_apps` (inner modules `c5x` → `c6x`) and `c60_common` → `c50_common` (inner modules `c6x` → `c5x`); built-in plugins `p50_*` → `p60_*` (`p60_visualizer_tk`, `p60_config_cli`, `p60_headless_mode`) with inner modules `p5x` → `p6x`
- **Breaking:** App bootstrap fields renamed `c52_load_plugins` / `c52_default_plugins` / `c52_community_plugins` → `c62_*` and `c50_selected_profile` → `c60_selected_profile`; visualizer fields `p5x_*` → `p6x_*`
- **Breaking:** Config layout is now **one file per profile** — `configs/<profile>.yaml` with sections `c10_perception`, `c20_planning`, `c30_control`, `c40_execution`, `c69_apps`, and `plugins:` (per-plugin settings keyed by directory name); replaces the per-layer (`c10_perception.yaml`, …) and per-plugin (`plugin_*.yaml`, `c59_apps.yaml`) files
- **Breaking:** Profile export/import is a single `.yaml` (was a zip); `c65_setting_utils.export_profile()` / `import_profile()` gate stack layers, `c69_apps`, and `plugins` sections behind include flags
- **Breaking:** `p50_visualizer_tk` modules renumbered to `p51`–`p59` (apps at `p51_visualizer_app`, `p52_setting_app`, `p53_plugins_app`); settings GUI CLI renamed from `avlite config` to `avlite setting`; visualization YAML keys updated (`p56_*` plot, `p57_*` stack panels, `p58_*` log)
- **Breaking:** Merged Tk plugins into one package: `p50_visualizer_tk` hosts the visualizer, settings GUI (`avlite setting`), and plugin manager (`avlite plugins`); removed `p50_config_tk` and `p50_plugins_app_tk`
- **Breaking:** App bootstrap settings moved to `c59_settings.py` / `configs/c59_apps.yaml` (`c50_load_plugins`, `c50_default_plugins`, `c50_community_plugins`, profile selection)
- **Breaking:** Plugin lists removed from `c40_execution.yaml` (`c40_default_plugins`, `c40_community_plugins` → `c50_*` on `AppSettings`)
- **Breaking:** Visualization YAML is `plugin_p50_visualizer_tk.yaml` only (deleted `plugin_p50_config_tk.yaml`, `plugin_p50_plugins_app_tk.yaml`; no legacy `c50_apps.yaml` fallback)
- **Breaking:** Visualization settings fields use consumer prefixes (`p50_*`, `p56_*`, `p57_*`, `p58_*`, …) in `p50_visualizer_tk/settings.py`; `AppSettingsUI` Tk binder moved out of `c59_settings.py`
- c50_apps private helpers grouped into module-local classes (public API unchanged)
- p50 plugin modules use top-level imports for clearer inter-module dependencies (optional deps use try/except sentinels at module scope)
- **Breaking:** p50 app entry classes renamed from `*AppStrategy` to `*App` (`VisualizationApp` for the default visualizer CLI entry; base class remains `AppStrategy`)
- **Breaking:** Expand `c50_apps` to full app infrastructure: `c51_app_strategy`, `c52_factory`, `c53_plugins`, `c54_settings_schema`, `c55_setting_utils`, `c58_paths` (includes `DataPaths`); `c60_common` slimmed to `c61`–`c65` only
- **Breaking:** Renamed `c50_app_strategy` → `c51_app_strategy`; moved `c43_factory` → `c52_factory`, `c66_plugins` → `c53_plugins`, `c68_settings_schema` → `c54_settings_schema`, `c69_setting_utils` → `c55_setting_utils`
- **Breaking:** Stack core may import `c51_app_strategy`, `c54_settings_schema`, and `c58_paths` only (import-boundary allowlist)
- **Breaking:** Slim `c50_apps` to non-Tk app infrastructure (prior release); shared Tk in `p50_config_tk`
- **Breaking:** Tk apps moved from `c50_apps` to built-in plugins: `p50_visualizer_tk`, `p50_config_tk`, `p50_plugins_app_tk` (prior release)
- **Breaking:** `import_app_plugins()` moved from `c66_plugins` to `c50_app_strategy`
- **Breaking:** Terminal config commands moved from `avlite config validate|describe|…` to `avlite config-cli …`; `avlite config` opens the settings GUI
- **Breaking:** Optional integrations moved out of `avlite/plugins/` into separate community plugins with new names:
  - `p40_bridge_carla` → `avlite-bridge-carla`
  - `p40_bridge_gazebo` → `avlite-bridge-gazebo`
  - `p40_bridge_ROS2` → `avlite-bridge-ROS2`
  - `p40_executer_ROS2` → `avlite-executer-ROS2`
  - `p30_controller_joystick` → `avlite-controller-joystick`
- Execution profiles register optional plugins via `c40_community_plugins`; only `p50_headless_mode` remains built-in
- Global plan GUI flow uses `apply_global_plan()` instead of manual local-planner/controller calls
- Plugin log routing recognizes `avlite-*` community plugin names (controller → control layer, bridge/executer → execution layer)
- `requirements-full.txt` no longer includes `pygame` (install from the joystick plugin's requirements when needed)
- Stanley controller clips steering error before exponential slowdown to avoid numeric overflow
- ROS execution profile defaults updated (planner, perception pipeline, boundary file, sim timing)
- Consolidated modules: deleted `c17_map.py`, `c50_stack_loader.py`, `_visualization_ui.py`, `c66_hdmap.py`
- Stack orchestration moved into `c43_factory` (`load_stack_settings`, core `get_stack_settings_classes()`); GUI profile export uses `c59_settings.get_stack_settings_classes()` for visualization YAML
- Renamed `c60_plugins.py` → `c53_plugins.py`
- `c67_paths` slimmed to `ConfigPaths`, `PluginPaths`, and `DataPaths`; `c54_plugins` refactored into internal Git/plugin operation classes
- `Map` / `RaceMap` live in `c11_perception_model.py`; OpenDRIVE `HDMap` parser in `c18_hdmap_parser.py`
- `VisualizationSettings` Tk binder merged back into `c59_settings.py` (schema + runtime class)

### Removed
- **Breaking:** `PerceptionCapability`, `LocalizationCapability`, and `MappingCapability` enums (folded into `StackCapability`)
- **Breaking:** `GT_DETECTION` / `GT_TRACKING` / `GT_LOCALIZATION` members of `WorldCapability` (ground truth now flows through world `stack_capabilities`)
- **Breaking:** `ExecutionSettings.c41_provide_ground_truth` / `c41_provide_rgb` / `c41_provide_depth` / `c41_provide_lidar` booleans and their fixed Bridge checkboxes (replaced by the `c41_provided` capability checklist)
- Built-in plugins: `p30_controller_joystick`, `p40_bridge_carla`, `p40_bridge_gazebo`, `p40_bridge_ROS2`, `p40_executer_ROS2` (now optional community plugins)
- Bundled `configs/plugin_p30_controller_joystick.yaml` and `configs/plugin_p40_*.yaml`
- `async` profile from `c40_execution.yaml` and `c50_apps.yaml`
- Per-profile embedded settings from `c10_perception.yaml`, `c20_planning.yaml`, and `c30_control.yaml`
- `c17_mapping_algs.py` placeholder (mapping interface remains in `c14_mapping_strategy.py`)

### Fixed
- Missing config file on load treated as defaults (debug log) instead of error in `load_setting`

## [0.3.2] - 2026-06-27

### Added
- GUI startup profile: last selected profile saved in `~/.config/avlite/startup_profile` and restored on launch
- Settings window **Export profile** / **Import profile**: zip of per-file profile slices (including community plugin YAMLs when referenced); validated against Pydantic schemas on export and import
- `avlite config export-profile` and `avlite config import-profile` CLI subcommands
- **Edit repository configs** (settings window, git clone only): optional dev mode to switch read/write between user dir and `{repo}/configs/`; preference in `~/.config/avlite/config_target`
- `c53_plugins.py` (plugin discovery, loading, log routing) and `c58_paths.py` (config/XDG paths)
- Thread-safe log filter snapshots in the visualizer (Core / Plugins / per-layer toggles)
- Community plugin import skips `.venv`, `site-packages`, and similar vendor directories
- Tabbed **Community** / **Members** plugin browser (`python -m avlite plugins`)
- GitHub Device Flow sign-in for AV-Lab private registry (`AV-Lab/avlite-private-plugins`)
- OAuth token persistence at `~/.config/avlite/github_oauth.json`; `AVLITE_GITHUB_OAUTH_CLIENT_ID` override
- SAML SSO authorize-link handling on 403; Copy button for device-flow user code
- Safety disclaimers on both plugin tabs
- Tests in `test/c60_apps/test_c63_plugins.py`
- UI logo resolved from repo `data/imgs/` independent of CWD

### Changed
- User data directory for maps and trajectories is now `~/.config/avlite/data/` (override with `AVLITE_DATA_DIR`)
- **Save Global Plan** (Planning panel ⬇) opens a native save dialog in `~/.config/avlite/data/` instead of a typed path prompt
- **Breaking:** Renamed `avlite/extensions/` → `avlite/plugins/` with `pNx` package naming (e.g. `p40_executer_ROS2`, `p40_bridge_carla`)
- **Breaking:** Renamed `configs/ext_*.yaml` → `configs/plugin_*.yaml`; `ExtensionSettings` → `PluginSettings`; `load_extensions` → `load_plugins`; `c40_default_extensions` → `c40_default_plugins`
- Renamed `c60_common` modules: `c61_capabilities`, `c62_sensor_data`, `c66_hdmap`, `c68_settings_schema`, `c69_setting_utils` (update imports if you extend AVLite)
- **Use repository configs** superseded by **Edit repository configs** (path switch instead of deleting local YAML)
- Authenticated git clone/update for private plugin repos (non-interactive git, Basic auth header, timeouts)
- Shared `apply_ttk_theme()` helper for consistent dark/light styling across windows

### Removed
- **Copy repository configs** button and `copy_repo_configs_to_user()` (use **Edit repository configs** to work on repo YAML, or import a profile zip to populate the user dir)
- Bundled `p10_perception_MO_prediction` from `avlite/plugins/`; default configs updated

### Fixed
- Settings window (`T`) no longer crashes when opening the repository-config controls after a partial code reload
- Logo and About dialog load assets via repo/package path instead of process CWD
- Dark theme consistent when launched from any directory; standalone plugins window uses equilux
- Theme re-applied after profile load when `dark_mode` changes in YAML
- `ControlComand` typo alias for `ControlCommand` (backward compatibility)

## [0.3.1] - 2026-06-21

### Added
- User config directory (`~/.config/avlite/`, override with `AVLITE_DATA_DIR`): flat YAML layout mirroring repo `configs/` basenames; load prefers user copy, falls back to repo; Save writes user dir only
- User data directory for maps and trajectories (`~/.local/share/avlite/data/`, override with `AVLITE_DATA_DIR`): read checks user then repo; saves (e.g. global plans) go to user dir only
- **Use repository configs** button in the settings window to discard local YAML overrides and reload shipped defaults
- `avlite config help` subcommand; bare `avlite config` prints help instead of erroring
- Schema tooltips on main-page controls (strategy dropdowns, timing fields) in addition to the settings window
- `SensorFrame` and canonical sensor layouts in `c62_sensor_data.py` between world bridges and perception/localization
- Config and data path resolution tests (`test_c61_config_paths.py`)

### Changed
- Perception/localization sub-strategy dropdowns use **None** as the default label (replacing legacy “Ground Truth” / “Default Perception Model” sentinels)
- Community plugin install paths stored relative to the plugins directory when registered from the plugin browser
- Tooltip text shows field description first, then type and default in parentheses

### Fixed
- `avlite config --help` and missing subcommand no longer emit a spurious parse error line
- HD map, default trajectory, and global-plan save paths resolved consistently via `get_absolute_path()` (including when cwd is not the repo root)

## [0.3.0] - 2026-06-21

### Added
- Pydantic-backed settings schemas for stack layers and extensions with field descriptions
- `avlite config validate` and `avlite config describe` CLI subcommands
- Hover tooltips in the settings window (`c56`) showing field descriptions from schemas

### Changed
- Config load/save validates types and reports field-level errors instead of silently assigning bad values

### Fixed
- Declare `scipy` and `pydantic` as core dependencies in package metadata and requirements files

## [0.2.0] - 2026-06-04

### Added
- `KalmanTracker` (`c15_perception_algs`): constant-velocity Kalman filter multi-object tracker with greedy nearest-neighbour data association; persistent `agent_id` across frames and velocity estimation
- `FastBEVLidarDetection` (`c15_perception_algs`): BEV LiDAR segmentation by consecutive-gap splitting + rotating-calipers minimum bounding rectangle; supports 2D scans `(N,2)` and 3D point clouds `(N,3+)` with z-band filtering; exposes `detection_clusters` diagnostic field on `PerceptionModel`
- `LidarLocalization` (`c16_localization_algs`): ICP scan-to-map ego localization; seeds reference map from first scan, then estimates translation + rotation via iterative closest-point alignment
- `c17_mapping_algs.py`: placeholder for future online-mapping algorithms
- `GlobalCenterlineRacePlanner` (`c25_global_race_planners`): race-line planner from a JSON left/right boundary file; computes centre-line path with curvature-adapted target velocities (`v = min(v_max, sqrt(a_lat/κ))`)
- `HDMap` moved from `c10_perception/c18_hdmap` to `c60_common/c67_hdmap` and re-imported by all dependent modules
- `HDMapGlobalPlanner` split out into its own module `c24_global_hdmap_planners`
- `AnyOf` capability requirement class and `satisfies_requirements()` helper in `c61_capabilities`; allows a strategy to declare that any one of several world capabilities suffices (e.g. `AnyOf(LIDAR_2D, LIDAR_3D)`)
- `Executer.replan_global()`: recompute the global plan at runtime from the current ego pose and push it to the local planner and controller
- `BasicSim.get_lidar_data()`: 2-D LiDAR simulation via ray-segment intersection against agent bounding boxes and road boundaries; configurable range, beam count, and FOV
- `BasicSim.reset()`: clears simulated NPC agents and their controllers
- `LIDAR_2D` capability declared by `BasicSim`
- `rename_setting_profile()` utility in `c69_setting_utils`
- `race_boundary_map` setting in `PlanningSettings`; BasicSim LiDAR settings in `ExecutionSettings`
- `data/race_boundary_yas_marina.json`: Yas Marina race boundary data file

### Changed
- `set_global_plan()` in `LocalPlanningStrategy` (and `LatticePlanningStrategy`) now accepts an optional `ego_xy` parameter to initialise the Frenet location from the actual ego position instead of the plan's start point
- Emergency-stop detection in `GreedyLatticePlanner` changed from all-zeros velocity check to a trailing-velocity threshold (`velocity[-1] < 0.5` and `mean < 3.0`), reducing false positives at low speed
- Velocity clamping on insufficient stopping distance replaced by smooth linear ramp (current → obstacle speed) for more natural deceleration
- Emergency-stop velocity profile now ramps from current ego speed to zero instead of instantaneous zero-fill
- Ground-truth perception step copies agent lists into the executer's `pm` instead of aliasing the world's model; prevents perception resets from clearing simulator-spawned NPC agents
- Perception step in `SyncExecuter` moved before the planning step so the planner always operates on the current frame's obstacles
- `Executer.reset()` now calls `world.reset()` to also clear simulated NPC state
- `min_ramp_start_velocity` raised from 0.5 → 3.0 m/s to avoid deadlock when the ego is behind the plan start
- Default global planner changed from `RaceGlobalPlanner` to `GlobalCenterlineRacePlanner` throughout
- Factory creates a separate `world_pm` `PerceptionModel` for the world bridge, decoupling simulator state from the executer's perception model
- Velocity discrepancy between local and global reference plans now logged at INFO level in `GreedyLatticePlanner`

## [0.1.1] - 2026-05-06

### Added
- `FpsTracker` class (`c60_common/c65_fps_tracker.py`) for instantaneous FPS tracking with wall-clock or sim-time domain
- Shared step methods `_localization_step()`, `_perception_step()`, `_replan_step()`, `_control_step()` extracted to `Executer` base class, eliminating duplication between sync and async executers
- `FpsTracker` instances for all four execution loops (perception, planning, control, localization) in `Executer`; reset on `reset()`
- `perception_dt` setting in `VisualizationSettings` and execution control panel in the GUI
- Color-coded FPS display in headless dashboard (green ≥ 90 % of target, yellow < 90 %, red = 0) with actual/target rate shown
- `AsyncThreadedExecuter` planner thread now runs the perception step at `perception_dt` rate with a startup guard
- `ext_ROS2_worldbridge.yaml` and `ext_carla.yaml` config files
- `avlite.__version__` exposed from package metadata via `importlib.metadata`
- Replan stability logic in `LocalPlannerStrategy`: `should_switch_plan`, `_replan_wait_time`, `_urgent_collision_threshold`
- S-coordinate lap detection in `LocalPlannerStrategy.step()` to handle lateral displacement edge cases

### Changed
- Migrated packaging from ROS2/ament `setup.py` / `package.xml` to standard `pyproject.toml`
- Removed ROS2-specific `data_files` and `ament_index` artifacts from build config
- CLI `--control-dt` and `--replan-dt` now default to `None` and fall back to the active profile's settings instead of hard-coded values
- `control_dt`, `replan_dt`, and `perception_dt` GUI changes are persisted to the ROS2 extension YAML via `_sync_exec_dt()` so they take effect on the next ROS2 launch
- Config YAMLs updated across all modules to reflect new settings structure

### Removed
- `package.xml` (replaced by `pyproject.toml`)

## [0.1.0] - Initial release

### Added
- Modular AV stack: perception, planning, control, execution, visualization layers
- Lattice-based local planner with global trajectory fallback
- Frenet-coordinate tracking (`traversed_s`, `traversed_d`)
- ROS2 / Gazebo / Carla extension hooks
