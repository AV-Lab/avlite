# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
