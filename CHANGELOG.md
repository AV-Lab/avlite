# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GUI startup profile: last selected profile saved in `~/.config/avlite/startup_profile` and restored on launch
- Settings window **Export profile** / **Import profile**: zip of per-file profile slices (including community plugin YAMLs when referenced); validated against Pydantic schemas on export and import
- `avlite config export-profile` and `avlite config import-profile` CLI subcommands
- **Edit repository configs** (settings window, git clone only): optional dev mode to switch read/write between user dir and `{repo}/configs/`; preference in `~/.config/avlite/config_target`
- `c60_plugins.py` (plugin discovery, loading, log routing) and `c67_paths.py` (config/XDG paths)
- Thread-safe log filter snapshots in the visualizer (Core / Plugins / per-layer toggles)
- Community plugin import skips `.venv`, `site-packages`, and similar vendor directories

### Changed
- User data directory for maps and trajectories is now `~/.config/avlite/data/` (override with `AVLITE_DATA_DIR`)
- **Save Global Plan** (Planning panel ⬇) opens a native save dialog in `~/.config/avlite/data/` instead of a typed path prompt
- **Breaking:** Renamed `avlite/extensions/` → `avlite/plugins/` with `pNx` package naming (e.g. `p40_executer_ROS2`, `p40_bridge_carla`)
- **Breaking:** Renamed `configs/ext_*.yaml` → `configs/plugin_*.yaml`; `ExtensionSettings` → `PluginSettings`; `load_extensions` → `load_plugins`; `c40_default_extensions` → `c40_default_plugins`
- Renamed `c60_common` modules: `c61_capabilities`, `c62_sensor_data`, `c66_hdmap`, `c68_settings_schema`, `c69_setting_utils` (update imports if you extend AVLite)
- **Use repository configs** superseded by **Edit repository configs** (path switch instead of deleting local YAML)

### Removed
- **Copy repository configs** button and `copy_repo_configs_to_user()` (use **Edit repository configs** to work on repo YAML, or import a profile zip to populate the user dir)

### Fixed
- Settings window (`T`) no longer crashes when opening the repository-config controls after a partial code reload

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
