# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Common: `CameraParams` (intrinsic `K`, per-frame `world_to_camera`, width, height) and the optional `SensorFrame.camera_params` field — camera geometry a fusion strategy needs to project world-frame `lidar` into the image. Extrinsic targets the OpenCV optical frame (x right, y down, z forward). Additive: `WorldBridge.get_camera_params()` defaults to `None`, so no existing bridge changes; bridges declaring `CAMERA_RGB` / `CAMERA_DEPTH` should override it
- Execution: `c40_start_pose` (`[x, y, theta]` or null) — profile-defined ego start; factory falls back to the global-plan start point when null
- Perception: `State.set_start()` — capture current pose as the snapshot restored by `reset()` (via `get_copy` / `copy_from`)
- Visualizer: **Save Start** on the Execution state row — writes live ego pose into `c40_start_pose` and the active profile YAML
- Plugin registry: optional `display_name` — human-readable plugin title in the Plugins browser and the docs store; falls back to `name`, which stays the install-folder / import identifier
- Plugin registry: optional `site_url` — **Open Website** button plus a Website row in the plugin details window, and a **Site** button on the docs plugin cards

### Changed
- Perception: `State` / `AgentState` reset snapshot is a polymorphic copy of all fields (drops per-field `__init_*` / `AgentState.reset` override)
- Execution: `BasicSim.reset()` restores ego and NPC start poses (and NPC controllers) instead of clearing agents
- Execution: drop duplicate `world.reset()` call in `ExecutionStrategy.reset()`
- Docs / README: Tk visualizer demo uses a looping video (`docs/imgs/tk_visualizer.mp4`) instead of a static screenshot; landing shot fills the content column
- Docs: call out pause / step / interactive debug early (landing value strip, Overview features, Quick Start)
- Docs: Community Plugins cards are no longer whole-card GitHub links — explicit **Site** / **Repo** buttons sit above the GitHub stats footer, dependency notes clamp to two lines (full text on hover), and the links carry per-plugin aria-labels
- Docs: plugin registry field tables list every field with a required column (README, Overview, Plugin Development)

### Fixed
- Execution: unpaced (free-run) Stop/Start no longer integrates the pause as wall-clock Δt — ego and NPCs jumped by up to 1 s of motion on resume (Sync and Async)
- Common: `TrajectoryTracker` initializes `path_s` from cumulative arc-length instead of re-projecting the reference through KD-tree Frenet conversion — closed tracks with `first==last` (e.g. bundled Yas Marina race line) no longer get non-monotonic `path_s` with `path_s[-1] == 0`
- Common: Frenet XY→SD picks the better adjacent segment around the nearest waypoint (and SD→XY brackets by arc-length) — on-path points after corners no longer pick up a huge false CTE from the previous segment
- Common / Planning: lattice sampling, replan end-of-track gates, and race lap detection use `TrajectoryTracker.track_end_s` (`path_s[-1]`) instead of the stale `path_s[-2]` workaround — avoids `IndexError` on 1-point paths and restores the final closed-track segment after the cumulative `path_s` fix
- Control: Pure Pursuit clamps lookahead with `max(path_s)` rather than `path_s[-1]`, so a corrupted end sample cannot pin every lookahead to the start/finish
- Common: `TrajectoryTracker` / `slice_trajectory_horizon` tolerate a 1-point path (final waypoint) — Frenet conversion no longer indexes `next_wp=1` and crashes `VelocityLocalPlanner.replan` at path end
- Visualizer: Control **Step** / Steer / Accel apply plant control and sync stack PM via `apply_world_control` (same dual-write as teleport after the world/stack ego split)
- Visualizer: **Save Start** snapshots velocity 0 so Reset matches a cold profile start (live speed is preserved while driving)
- Execution: perception, planning, and control share one `SensorFrame` per tick instead of each fetching its own — the stack no longer assumes the world holds still between stages, so bridges whose sensors evolve independently (CARLA async mode) stay coherent. `_localization_step` / `_perception_step` / `_replan_step` / `_control_step` now take the snapshot as an argument; each executer loop resolves its pacing gates first and fetches at most once (skipping the fetch entirely when no stage is due)
- Visualizer: Control **Align** teleports plant ego and syncs stack PM (stack-only writes were undone by GT localization after the world/stack ego split)
- Common: `TrajectoryTracker.update_waypoint_by_wp` / `update_to_next_waypoint` clamp `next_wp` at the path end — `%` precedence previously left `next_wp == len(path)` and crashed plot/step at the final waypoint
- Common: `create_quintic_trajectory_sd` b-vector matches the constraint matrix (end 1st / start 2nd derivatives were swapped)

## [0.5.3] - 2026-07-24

### Added
- Execution: per-module pacing flags `c40_pace_perception` / `c40_pace_replan` / `c40_pace_control` / `c40_pace_sim` (default on); when off, that module runs best-effort (sim uses clamped wall-clock Δt)
- Perception: `State.copy_from` for in-place dataclass field sync (preserves object identity)
- Factory: distinct world ego vs stack `pm.ego_vehicle` so estimated localization can own stack pose without mutating the plant
- Visualizer: Settings toolbar plot toggles (Global / Local Frenet / Local Global); local panel shown iff Frenet or Local Global is on
- Visualizer: Δt pacing checkboxes left of each period entry; entry disabled when pacing is off
- Visualizer: right-click spawn places the agent on press (live preview with orientation arrow while dragging)
- Visualizer: `teleport_ego` syncs plant → stack PM so the drawn ego moves immediately

### Changed
- Sync / async executers honor pace flags in `step` / worker loops (control holds last command between recomputes when pace_control is off)
- Headless / ROS step kwargs pass through pace settings
- Visualizer: Control stack first row pack no longer expands (top-aligned with Perception / Planning)

### Fixed
- Interactive teleport left stack PM ego stale until the next exec tick (world/stack ego split)

### Added
- Execution: optional `stack_event` on `PerceptionModel` and `ControlCommandBase`; executer harvests after localize / perceive / control (same clear-once pattern as plans)
- Docs: Execution Tasks page (philosophy-first TaskRunner / stack extension); Raising events (stamp vs `task_runner.notify` vs lifecycle)
- Docs: flexible stack composition (any module optional; end-to-end sensors→plan / sensors→control via capabilities)
- Docs: Community Plugins page — live registry card grid with GitHub stats, search/filter/sort

### Changed
- Docs: MkDocs landing chrome (dark black header, no Home/Docs tabs; left nav peer pages with active highlight); light-mode landing header uses brand primary

## [0.5.0] - 2026-07-23

### Added
- Planning: `c28_preferred_extra_clearance` — gate hard-preferring centerline (`d≈0`) lattice edges on corridor-to-obstacle clearance beyond the collision hard floor
- Collision: `check_collision` returns `min_clearance` (approx. line–obstacle distance minus ego half-width + safety margin); lattice edges store it for cost
- Execution: `TaskStrategy` / `TaskRunner` (`c43`) with `EVERY_CYCLE` / `INTERVAL` / `ON_EVENT`; lifecycle + domain `notify`; built-ins `GoalArrivalMonitor`, `StopExecAtGoalTask`, `TelemetryTask` (`c47`); setting `c40_execution_tasks`
- Planning: optional `stack_event` on `LocalPlan` / `GlobalPlan`; executer harvests after replan
- Visualizer: Execution **Tasks** chip row (registry `+` picker, per-chip ⓘ / ×, wraps on resize)
- Visualizer: local-plan right-click spawn — hold and drag to set agent heading (blue orientation arrow); spawn on release

### Fixed
- Collision: ego trajectory corridor extends by half vehicle length before buffering so front/rear body is covered (flat-cap tube)
- Lattice: when an agent blocks ahead, level-0 / partial-replan / ShortestPath share lateral preference via `_candidates_for_selection` (avoids early cut-back into side traffic)
- Lattice: `_edge_cost` uses real `min_clearance` (was a dead default); d≈0 hard-prefer only when clearance ≥ `c28_preferred_extra_clearance`
- Velocity: tight-gap speed-match commits a max-decel step at `current_wp` so async replan actually brakes (was re-commanding current speed at the stop budget)
- Velocity: matched-speed follow below the cruise-gap threshold uses the gap-aware kinematic profile instead of only painting lead speed

### Changed
- Execution: `BasicSim.spawn_agent` keeps the caller-provided `theta` (no longer overwrites with route heading under NPC control)
- Execution: remove `ExecutionStrategy.run()` and `_stop_event`; cooperative cancel is `stop()` / public `stopped` flag. Headless owns the paced step loop and Ctrl+C `threading.Event`; `AsyncThreadedExecuter` drops `__kill_flag` and checks `stopped`
- Execution: public `task_runner` (was `_task_runner`); remove `ExecutionStrategy.notify` — callers use `executer.task_runner.notify(event)`; `TaskRunner` binds its owning executer
- Execution: world-capability sensor filtering lives in `WorldBridge.get_sensor_frame()` (removed `ExecutionStrategy._fetch_sensor_frame`); `ExecutionStrategy` public API grouped above private helpers
- Visualizer: contract popup soft-requirement label `may ·` → `optional ·` (API still `MayUse`)
- Capabilities: shared `CapabilityGroup` base for `AnyOf`/`MayUse` with reload-safe `.matches()`; structured `combine_stack_requirements(..., soft=)` (keeps each `AnyOf`, merges `MayUse`, strips AND-covered caps) replaces `required_stack_capabilities` / `used_stack_capabilities` / `flatten_stack_requirements`; removed `is_any_of` / `is_may_use` / `is_requirement_wrapper`

## [0.4.5] - 2026-07-11

### Added
- Visualizer: Help → **Update…** checks PyPI and can `pip install --upgrade avlite`; quiet startup toast when a newer version is available and exec is idle (`c66_app_update.AppUpdater`)
- Capabilities: `MAP_HD` and `MAP_RACE_TRACK` replace generic `MAP`; `MapReader` advertises the matching cap from the loaded map type; HD/race global planners require the typed cap
- Capabilities: `MayUse(...)` soft requirement — never blocks stack assembly; modules may use listed caps when present (e.g. local planners + DETECTION/PREDICTION)
- Capabilities: `used_stack_capabilities` — flattens hard + soft (`MayUse`) deps for visualizer “consumed” coloring (assembly still uses hard-only `required_stack_capabilities`)
- Visualizer: stack Combobox **ⓘ** / right-click contract popup shows world requirements, stack requirements (incl. world GT), and provided capabilities; labeled `all ·` / `any ·` / `may ·` rows; Escape and Close always work
- Visualizer: Bridge Setting Combobox **ⓘ** / right-click shows world capabilities, stack requirements, and stack capabilities for the selected `WorldBridge`
- Control: `PurePursuitController` (`c35_pure_pursuit`) — geometric path Pure Pursuit with speed-adaptive lookahead and velocity PID (`c35_*` settings)
- Control: `FollowTheGapController` (`c35_pure_pursuit`) — LiDAR Follow-the-Gap (widest forward free gap → Pure Pursuit steer; optional cruise speed when no plan); documented with Pure Pursuit in `docs/algorithms.md`
- Plugins browser: **Show Installed only** / **Show Active only** filter checkboxes with status tooltips
- Settings window: plugin double-click dialog adds **Open on GitHub**
- Planning: `GlobalRacePlanner` — raceline optimizer for race-boundary maps: blended minimum-curvature / shortest-path bounded least-squares optimization with a lateral + longitudinal acceleration-limited velocity profile (`c25_*` settings, defaults sized for a Dallara Super Formula platform); closed-track aware, with per-stage info logging. Documented in `docs/algorithms.md`
- Visualizer global race plot: hover over the colored raceline to read the target speed at the nearest waypoint in m/s and km/h
- `c50_common`: `c53_stack_datatypes` — `StackCapability` → payload type registry and `AgentType` → control command map (`control_type_for_agent`); `WORLD_CAPABILITY_SENSOR_FIELDS` on world sensor datatypes

### Fixed
- Visualizer: contract popup no longer crashes on `MayUse` after stack reload, and `PerceptionPipeline` ⓘ shows stage requirements/capabilities after selecting tracker/predictor (reload-safe name matching)
- Perception: empty detect/track stages no longer hard-require `DETECTION`/`TRACKING` on `PerceptionPipeline` (requirements come only from active children; world GT injection stays a bridge concern)
- Execution: `BasicSim` advertises `TRACKING` ground truth so default `PerceptionPipeline` (empty detect/track) no longer skips every perception step when those caps are still required by active stages
- Perception: `PerceptionPipeline` drops redundant `MayUse` stack deps already covered by hard child requirements (e.g. tracker `DETECTION` + predictor soft deps)
- Visualizer: `PREDICTION` contract coloring — local planners soft-use prediction (not tracking); green includes `MayUse`; orange ignores parent `PerceptionPipeline` for stages and respects Bridge Setting checkboxes for world GT
- Visualizer: controller `CONTROL` capability colored consumed (green) when the world bridge declares it in `stack_requirements`
- Visualizer: Perception map row starts at column 0; Default Map sits beside mapping without stretching the frame
- Perception: empty detect/track stages no longer advertise `DETECTION`/`TRACKING` on `PerceptionPipeline` (still required from world GT)
- Visualizer: `GlobalRacePlanner` now maps to the race plot view (was unset, falling back with an error)
- Visualizer: close the global matplotlib figure before plot recreation on stack reload or planner switch (fixes figure leak warning)

### Changed
- Control: `FollowTheGapController` — safety bubble (`c35_bubble_radius`), prefer interior gaps over ±90° edges, and path-biased gap pick when a trajectory is available (`c35_min_gap_width`)
- Executer: unmet hard **module** `stack_requirements` raise `ValueError` at stack build (world unmet deps and duplicate providers still warn)
- **Breaking:** Key strategy methods gain optional `(perception_model, sensors)` in addition to existing args (executer / pipeline / UI supply both). Control keeps `control(ego, plan=None, control_dt=None, …)`; detect keeps optional `rgb_img`/`depth_img`/`lidar_data` alongside `sensors`. LiDAR for FTG via `sensors.lidar`. Docs: `architecture.md`, `plugin-development.md`, `algorithms.md`
- Bridge Setting: split `c41_provided` into `c41_world_capabilities` (sensors) and `c41_world_stack_capabilities` (bridge GT); helpers moved to `c41_world_bridge` (`is_world_capability_enabled` / `is_world_stack_capability_enabled`); settings stay data-only
- WorldBridge: optional `stack_requirements` (default empty); `BasicSim` requires `CONTROL`; executer validates world deps with other modules
- Map interface: unified `c40_map` setting (replaces `c40_hd_map` / `c43_race_boundary_map`); factory loads once via `Map.open` into planners, `MapReader`, and WorldBridge
- Mapping: `MapReader` (`MappingStrategy`) is the optional stack MAP provider; BasicSim stays standalone (`DETECTION` + `TRACKING` + `LOCALIZATION` only) and may hold `map` for LiDAR geometry without advertising MAP
- Executer: optional `mapping=` module; removed special-case MAP injection from `world.map`
- Visualizer: mapping Combobox next to localization; Default Map picker always binds to `c40_map`
- Global planners take `Map` / `RaceMap` / `HDMap` objects (not filepaths)
- Removed `c46_lidar_boundary_file`; sim geometry comes from the injected map
- Executer `_validate_stack` warns on unmet requirements and duplicate capability providers
- Visualizer: Default Map and Default Global Plan pickers include an empty option
- Capabilities: leaf strategies declare contracts as public `frozenset` class attributes (ABC abstract `@property` still enforced); pipelines keep dynamic instance `@property`; visualizer contract popup reads class attrs without stub construction (`_peek_strategy` removed)
- Planning: `GlobalPlannerStrategy` world/stack requirements and capabilities are abstract; `HDMapGlobalPlanner`, `GlobalCenterlineRacePlanner`, and `GlobalRacePlanner` declare them explicitly
- Execution: `WorldBridge.stack_capabilities` is abstract (with `world_capabilities`); concretes such as `BasicSim` must declare both
- Planning: velocity and lattice local planners soft-depend on `MayUse(DETECTION, PREDICTION)` instead of `DETECTION`/`TRACKING`
- Perception: `PerceptionPipeline.stack_capabilities` follows active stages only (no `PREDICTION` unless a predictor is set; empty detect/track require world GT via `stack_requirements` and do not re-advertise those caps)
- Perception: detect/track/predict, localization, and mapping capability contracts are abstract; concretes declare them explicitly
- Visualizer contract popup: orange for redundantly provided stack capabilities (also offered by another top-level module or checked world GT)
- Planning: local planner / sub-stage capability contracts are abstract; concretes declare `world_requirements`, `stack_requirements`, and `stack_capabilities` explicitly
- Visualizer: empty selection allowed for perception, localization, global/local planner, and controller (module omitted from the stack)
- Visualizer: moved `spawn_agent`, `replan_global`, and `apply_global_plan` from `ExecutionStrategy` onto `VisualizerApp` (interactive helpers; executer stays tick/lifecycle only)
- Plugins Members tab: sign-in status and button moved above the warning
- Settings window: plugin double-click shows **Settings file** and **Source file location** instead of package name
- Control: agent-type → command mapping (`control_type_for_agent`, registries) moved from `c38_control_mapping.py` into `c31_control_model.py`; `c38` removed
- `c50_common` renumber: `c52_world_sensor_datatypes`, `c53_stack_datatypes`, `c54_trajectory_tracker`, `c55_collision_checking`, `c56_fps_tracker`
- Strategy ABCs: capability attrs are soft class-level defaults (algorithm methods stay abstract) so older plugins load without declaring `stack_*`
- `control_type_for_agent` / `DEFAULT_CONTROL_TYPE_BY_AGENT` live in `c53_stack_datatypes` (not `c31`)

## [0.4.4] - 2026-07-06

### Added
- Visualizer settings: **Show prediction**, **Occupancy flow**, and **Mapping** row in the settings window (`p67_show_prediction` gates all prediction overlays)
- Bridge Setting: side-by-side **world capabilities** / **stack capabilities** columns with scrollable checklists, capability tooltips, and theme-aware canvas backgrounds
- Settings editor: per-section **Reset to stack defaults** / **Reset to plugin defaults** buttons (schema-validated source defaults)
- Planning: `c27_follow_gap_gain` — proportional speed reduction when the ego is inside the follow-gap standoff
- Planning: `c20_beside_agent_sweep_time` — shorter forward sweep for beside/just-behind agents in lattice collision precompute
- Planning: `c20_beside_agent_rear_window` — bounds the beside-sweep to agents within N metres behind the ego (overtake cut-back protection); agents further back stay static boxes
- Planning: `ShortestPathLatticePlanner` — lattice planner that commits to the globally optimal chain (dynamic programming over the lattice DAG: longest reachable depth first, then minimum summed edge cost) instead of greedy per-level selection
- Docs: vim-style visualizer shortcut guide in overview and quick-start

### Changed
- Perception panel: removed **Show** / **Extras** checkboxes and the Extras row; **loc:** localization dropdown is inline on the main panel
- Bridge capability panels use content-sized height (capped at 96px) instead of a fixed tall canvas
- Frenet view: ego is positioned ~25% from the left (more s-range ahead) instead of centered
- Global map full-view zoom: 5% border margin so the track no longer touches plot edges
- Control gauges: label/gauge rows aligned via grid; value boxes tint by deviation magnitude
- `_sync_exec_dt` moved onto `VisualizationSettings` as an instance method
- Settings window: **Save** / **Close** moved to a shared footer (visualizer and standalone modes)
- Stack/profile reset applies schema-validated source defaults per layer instead of reloading widgets only
- Collision checking returns the **nearest** blocker (smallest path index), not the first agent in list order — cut-ins re-target onto the closest lead
- Collision precompute forward-sweeps agents **only when a predictor supplies a trajectory** — the constant-velocity fallback was removed, so with prediction disabled every agent stays a static box (no fabricated forward projection)
- Velocity planner: when inside the follow gap, brake proportionally to re-open standoff before resuming cruise speed
- Planning settings: clearer `c20_collision_safety_margin` / `c20_obstacle_inflation_margin` / `c20_min_velocity_threshold` descriptions; schema defaults `c20_boundary_margin` 0.25 m and `c20_collision_safety_margin` 0.5 m
- Bundled profiles: dark mode defaults restored; planning/control tuning for **local planning** and **global planning** profiles
- Docs: algorithms collision-margin table aligned with ego-side vs agent-side clearance semantics; visualizer screenshot updated
- Visualizer local plot: blitting for stable-view frames (limits/size unchanged) and view-window decimation of reference trajectory, boundaries, and track-boundary segments so follow-mode full redraws are cheaper; static Frenet (ax2) geometry is cached per plan instead of recomputed every frame — reduces main-thread plot cost that was capping planner FPS with the local plot enabled

### Fixed
- Far-behind agents painting front lattice edges red: with prediction disabled no forward sweep occurs, and with prediction on the beside-sweep is scoped to the rear window so a trailing vehicle no longer flags edges ahead of the ego
- Lattice overtake sideswipe: beside/just-behind moving agents are swept forward (via predictor) so the ego stays clear of a just-passed agent before cutting back to the reference line
- Startup layout jump in Bridge Setting (empty-then-shrink canvas height)
- Plot views rendering with wrong aspect ratio before first layout pass (`_canvas_ready` guard + `after_idle` replot)
- Light mode: world/stack capability canvases now refresh background when toggling dark/light theme
- Premature `update_idletasks()` before plot views were gridded at startup
- Standalone settings window (`python -m avlite setting`) closes cleanly instead of leaving a hidden host window

## [0.4.3] - 2026-07-05

### Fixed
- Bundled profile configs missing after `pip install` — ship `avlite/configs/` as package data so default profiles resolve without a git checkout

### Changed
- Bundled profiles moved from repo-root `configs/` to `avlite/configs/`; `ConfigPaths.bundled_dir()` resolves to the packaged directory

## [0.4.2] - 2026-07-05

### Fixed
- README logo and screenshot now use absolute raw GitHub URLs so they render on the PyPI project page (relative paths 404 on PyPI); also corrects the logo path after data moved to `avlite/data/`

## [0.4.1] - 2026-07-05

### Added
- Curated top-level public API in `avlite/__init__.py` — strategy base classes, data models, capabilities, sensor datatypes, `TrajectoryTracker`, factory helpers (`executor_factory`, `load_stack_settings`), and settings singletons re-exported for plugin authors (`from avlite import ControlStrategy, EgoState, …`)
- Lazy loading (PEP 562 `__getattr__`) so `import avlite` stays lightweight; `TYPE_CHECKING` imports for IDE and static-analysis support
- Bundled sample maps and plans under `avlite/data/` (included in distributions via `pyproject.toml` package-data)
- `docs/quick-start.md` — guided install, visualizer walkthrough, and headless deployment
- MkDocs Material theme upgrades: tabs, instant navigation, search suggest/highlight, footer, and copyright line

### Changed
- **Breaking:** Default bundled data moved from repo-root `data/` to `avlite/data/`; `DataPaths.resolve()` falls back to the packaged `avlite/data` directory after the user data dir and legacy `~/.config/avlite/data/`
- README and docs index simplified: installation via `requirements.txt` / `pip install -e ".[dev]"`; removed `related-repos/` references; optional plugins described generically
- Plugin development and architecture docs updated for the bundled-data layout and optional-plugin messaging
- MkDocs nav restructured around the Quick Start guide; updated visualizer screenshot

### Fixed
- Log view: cancel scheduled poll timers and detach logging handlers on visualizer shutdown (`LogView.shutdown()`)
- Visualizer window close: call `LogView.shutdown()` and `quit()` before destroy
- Executer loop: cancel pending `after()` callback when stopping execution (`ExecView.stop_exec()`)

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
- `LocalPlanningPipeline` — staged local planning (behavioral → path → velocity), mirroring `PerceptionPipeline`; threads a mutable `LocalPlan` through stages
- Stage interfaces and registries: `LocalBehavioralPlanningStrategy`, `LocalPathPlanningStrategy`, `LocalVelocityPlanningStrategy`
- `LocalBehavior` enum and `LocalPlan.behavior` field
- Pipeline stage planners: `CruiseBehavioralPlanner`, `ReferencePathPlanner`; dual-role `VelocityLocalPlanner` and `GreedyLatticePlanner` (standalone `LocalPlanningStrategy` and pipeline stage)
- Planning settings: `c23_behavioral_strategy`, `c23_path_strategy`, `c23_velocity_strategy`
- Planning panel: pipeline sub-strategy dropdowns when `LocalPlanningPipeline` is selected; in-place pipeline refresh without full stack reload
- `test/c20_planning/test_c23_local_planning_pipeline.py`

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
- **Breaking:** Local planning modules reorganized by pipeline stage:
  - `c26_local_planners.py` → `c27_local_behavioral_and_velocity_planners.py`
  - `c27_local_lattice_planners.py` + `c28_lattice.py` → `c28_local_lattice_planners.py`
  - New `c26_local_path_planners.py` (`ReferencePathPlanner`)
  - `c23_local_planning_strategy.py` trimmed to base + ABCs + pipeline only
- **Breaking:** Planning settings renamed to match consumer file numbers in `c29_settings.py` and all `configs/*.yaml` profiles:
  - Velocity `c26_*` → `c27_*` (e.g. `c27_max_deceleration`, `c27_planning_horizon_points`)
  - Lattice `c27_*` → `c28_*` (e.g. `c28_planning_horizon`, `c28_maneuver_distance`)
  - Shared `c27_min_ramp_start_velocity` → `c20_min_ramp_start_velocity`
- Local planners decoupled from control layer: removed `controller` constructor parameter; velocity profiling uses `c27_max_deceleration` instead of control settings
- Velocity/lattice planners: inlined thin single-use helpers (structure only; no behavior change)
- Docs updated: `docs/algorithms.md`, `docs/settings-naming.md`, `README.md` module map
- For extenders/custom YAML: update imports to `c27_local_behavioral_and_velocity_planners`, `c28_local_lattice_planners`, `c26_local_path_planners`; rename `c20_planning` keys per the table above

### Removed
- **Breaking:** `PerceptionCapability`, `LocalizationCapability`, and `MappingCapability` enums (folded into `StackCapability`)
- **Breaking:** `GT_DETECTION` / `GT_TRACKING` / `GT_LOCALIZATION` members of `WorldCapability` (ground truth now flows through world `stack_capabilities`)
- **Breaking:** `ExecutionSettings.c41_provide_ground_truth` / `c41_provide_rgb` / `c41_provide_depth` / `c41_provide_lidar` booleans and their fixed Bridge checkboxes (replaced by the `c41_provided` capability checklist)
- Built-in plugins: `p30_controller_joystick`, `p40_bridge_carla`, `p40_bridge_gazebo`, `p40_bridge_ROS2`, `p40_executer_ROS2` (now optional community plugins)
- Bundled `configs/plugin_p30_controller_joystick.yaml` and `configs/plugin_p40_*.yaml`
- `async` profile from `c40_execution.yaml` and `c50_apps.yaml`
- Per-profile embedded settings from `c10_perception.yaml`, `c20_planning.yaml`, and `c30_control.yaml`
- `c17_mapping_algs.py` placeholder (mapping interface remains in `c14_mapping_strategy.py`)
- **Breaking:** Deleted modules `c26_local_planners.py`, `c27_local_lattice_planners.py`, `c28_lattice.py` (no import shims)
- **Breaking:** Removed planning settings `c26_stopping_decel_factor`, `c26_fallback_deceleration` (superseded by `c27_max_deceleration`)

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
