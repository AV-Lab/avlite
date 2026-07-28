# Plugin Development

AVLite supports three types of plugins:
- **Built-in plugins** (`avlite/plugins/`): Maintained by the core team
- **Community plugins**: Public registry via pull request to [avlite-community-plugins](https://github.com/AV-Lab/avlite-community-plugins)
- **Member plugins**: AV-Lab private registry ([avlite-private-plugins](https://github.com/AV-Lab/avlite-private-plugins)); browse/install via the **Members** tab after GitHub sign-in

This guide covers creating community plugins. Classes inheriting from base strategies automatically register and appear in the UI.

## What you can extend

| Layer | Base classes | UI / config |
|-------|--------------|-------------|
| **Perception** | `PerceptionStrategy` **or** `DetectionStrategy` / `TrackingStrategy` / `PredictionStrategy` via `PerceptionPipeline` | Main **Perception** dropdown; pipeline sub-dropdowns (Detect / Track / Predict) appear only when `PerceptionPipeline` is selected |
| **Localization** | `LocalizationStrategy` | Separate **Localization** dropdown (optional; independent of perception) |
| **Mapping** | `MappingStrategy` | Extendable via plugin; registry-based like other strategies |
| **Planning** | `GlobalPlannerStrategy`, `LocalPlanningStrategy` | Global / local planner dropdowns |
| **Control** | `ControlStrategy` | Controller dropdown |
| **Execution** | `WorldBridge`, `ExecutionStrategy`, `TaskStrategy` | Bridge / executer dropdowns; `c40_execution_tasks` list |

### Perception: monolithic vs pipeline

Perception is the most flexible layer — you can replace the whole stack in one class, or plug in individual stages.

```mermaid
flowchart TB
  subgraph mono ["Monolithic PerceptionStrategy"]
    M1["MyPerception.perceive()"]
    M1 --> M2["detect + track + predict in one class"]
  end
  subgraph pipe ["PerceptionPipeline composes sub-strategies"]
    P1["DetectionStrategy.detect()"]
    P2["TrackingStrategy.track()"]
    P3["PredictionStrategy.predict()"]
    P1 --> P2 --> P3
  end
  Exec["Execution selects perception by class name"]
  Exec --> mono
  Exec --> pipe
```

**Monolithic (`PerceptionStrategy`):**

- Implement all stages in one `perceive()` method.
- Select your class name in the main **Perception** dropdown, or set `c40_perception` under the `c40_execution:` section of `configs/<profile>.yaml`.
- Example built-in: `MultiObjectPredictor` (`p10_perception_MO_prediction`).

**Pipelined (`PerceptionPipeline` + sub-strategies):**

- Set perception to **`PerceptionPipeline`** in the main dropdown.
- The GUI shows **Detect**, **Track**, and **Predict** sub-dropdowns (only visible in pipeline mode).
- Configure sub-strategies under the `c10_perception:` section of `configs/<profile>.yaml`:

```yaml
c10_perception:
  c12_detection_strategy: MyDetector      # empty → ground truth from bridge
  c12_tracking_strategy: MyTracker        # empty → ground truth from bridge
  c12_prediction_strategy: MyPredictor    # empty → prediction stage skipped
```

- Each sub-strategy has its **own registry**; plugin classes auto-register like monolithic strategies.
- Empty name: detection/tracking use **ground truth** from the world bridge when available; prediction is skipped if unset.
- Non-empty unknown name: factory **raises on reload** (shown in the visualizer as "Reload failed").
- Mix core and plugin sub-strategies freely (e.g. core `FastBEVLidarDetection` + plugin `MyPredictor`).

**Localization** is separate from `PerceptionPipeline`: optional `LocalizationStrategy` in its own dropdown; updates `PerceptionModel.ego_vehicle` in-place.

### Planning, control, and world bridge

- **Planning** — subclass `GlobalPlannerStrategy` (`plan()`) or `LocalPlanningStrategy` (`replan()`); selected via global/local planner dropdowns; configured under `c40_execution:` in `configs/<profile>.yaml` (`c40_global_planner`, `c40_local_planner`).
- **Control** — subclass `ControlStrategy` (`control()`); selected via controller dropdown (`c40_controller`).
- **World bridge** — subclass `WorldBridge`; implement sensor getters and `control_ego_state()`; selected via **Bridge** dropdown (`c40_bridge`). See built-in `p40_bridge_*` plugins for reference.

### Execution tasks (TaskStrategy)

Execution tasks are an **orthogonal extension** of the running stack — mission, supervision, instrumentation — not a second planner. Philosophy and future-facing use: [Execution Tasks](execution-tasks.md).

Subclass `TaskStrategy` (`c43_task_strategy.py`) and implement `execute(executer, event=None)`. Declare `schedule` (`EVERY_CYCLE` / `INTERVAL` / `ON_EVENT`), and for events `listen_events`. List class names in `c40_execution_tasks` or pick them in the visualizer Execution **Tasks** chip row (`+` registry picker; not free text). Monitors may `executer.task_runner.notify(event)`; responses listen with `ON_EVENT`. Stack modules raise events by stamping `stack_event` on `PerceptionModel` / `LocalPlan` / `GlobalPlan` / control commands (no `task_runner` on the strategy API) — see [Execution Tasks → Raising events](execution-tasks.md#raising-events).

`placement` may be `INLINE` / `THREAD` / `PROCESS`, but the base executer runs **INLINE only** (non-INLINE logs a warning and falls back).

```python
from avlite import TaskStrategy, TaskSchedule, StackEvent

class MyOnPlanFailure(TaskStrategy):
    schedule = TaskSchedule.ON_EVENT
    listen_events = frozenset({StackEvent.LOCAL_PLAN_FAILED})

    def execute(self, executer, event=None) -> None:
        ...  # supervisor / instrumentation hook
```

```yaml
c40_execution:
  c40_execution_tasks: [MyOnPlanFailure]
```

## Community Plugin Structure

Create your plugin anywhere on your system:

```
/path/to/my_plugin/
├── __init__.py      # Export classes
├── settings.py      # Optional: PluginSettings if you have tunable params
├── my_strategy.py   # Your implementation
├── README.md        # Optional: shown in the Plugins browser
└── requirements.txt # Optional: extra pip dependencies
```

Tunable parameters are saved outside the plugin tree in `~/.config/avlite/plugin_<plugin_name>.yaml` (see section 1). Do not ship a `config/` folder or plugin-local YAML profiles in your repository.

Do not commit a `.venv` inside your plugin directory — AVLite scans all `.py` files under the plugin path and skips common vendor folders (`.venv`, `site-packages`, etc.), but keeping the venv outside the plugin tree is cleaner.

## 1. Settings File (Optional)

If your plugin has tunable parameters, add `settings.py` with a `PluginSettings` class. AVLite creates settings widgets automatically and saves profiles to `~/.config/avlite/plugin_<plugin_name>.yaml` (or under `AVLITE_CONFIG_DIR` if set). The filename uses the registry key / `c62_community_plugins` entry name — you do **not** need `exclude`, `filepath`, or a `config/` folder in your plugin package; AVLite derives the settings path when the plugin is registered and loaded.

```python
# settings.py
class PluginSettings:
    # Your parameters (appear in UI automatically)
    my_param: float = 1.0
```

Optionally add a `PluginSettingsSchema` (Pydantic) with `Field(description=...)` for tooltips in the settings window, same as built-in plugins.

If `~/.config/avlite/plugin_<plugin_name>.yaml` does not exist yet, AVLite may still **read** a legacy file at `<install>/config/<plugin_name>.yaml` from older setups; new saves always go to the user config directory.

Built-in plugins under `avlite/plugins/` are different: they set `filepath = "configs/plugin_*.yaml"` explicitly so shipped defaults live in the repository `configs/` directory and fall back from the user config dir on load.

## 2. Example: Custom Perception (Monolithic)

```python
from avlite.c10_perception.c12_perception_strategy import PerceptionStrategy
from avlite.c50_common.c51_capabilities import WorldCapability, StackCapability
from .settings import PluginSettings

class MyPerception(PerceptionStrategy):
    def __init__(self, perception_model, setting=None):
        super().__init__(perception_model, setting)

    world_requirements = frozenset({WorldCapability.CAMERA_RGB, WorldCapability.LIDAR_3D})
    stack_requirements = frozenset()
    stack_capabilities = frozenset({
        StackCapability.DETECTION, StackCapability.TRACKING, StackCapability.PREDICTION,
    })
    
    def perceive(self, perception_model=None, sensors=None):
        # Fuse camera and LiDAR to detect, track, and predict agents.
        # Read sensors.rgb / sensors.lidar as needed; update perception_model.
        if perception_model is not None:
            self.perception_model = perception_model
        return self.perception_model
```

Leaf strategies declare contracts as public ``frozenset`` class attributes (satisfies
the ABC abstract ``@property`` without constructing an instance). Pipelines keep
instance ``@property`` aggregators. ``@property`` overrides remain supported.
In the visualizer, **ⓘ** / right-click shows `all ·` / `any ·` / `optional ·`
requirement rows and colors provided caps green (consumed, including soft `MayUse`),
orange (redundant with another top-level module or checked world GT), or gray (unused).

All key methods (`perceive`, `detect`, `track`, `predict`, `localize`, `replan`,
`plan`, `control`) take the same optional pair ``perception_model`` + ``sensors``,
supplied by the executer, pipeline, or UI. See [Architecture → Capability System](architecture.md#capability-system).

## 3. Example: Detection, Tracking, or Prediction Sub-Strategy

Use `DetectionStrategy`, `TrackingStrategy`, or `PredictionStrategy` when you only need
to implement one stage of the pipeline. These plug into `PerceptionPipeline` and are
selected by name when **Perception** is set to `PerceptionPipeline`.

```python
from avlite.c10_perception.c12_perception_strategy import DetectionStrategy
from avlite.c50_common.c51_capabilities import WorldCapability, StackCapability
from avlite.c10_perception.c11_perception_model import PerceptionModel

class MyDetector(DetectionStrategy):
    world_requirements = frozenset({WorldCapability.CAMERA_RGB})
    stack_requirements = frozenset()
    stack_capabilities = frozenset({StackCapability.DETECTION})

    def detect(
        self,
        perception_model=None,
        sensors=None,
        rgb_img=None,
        depth_img=None,
        lidar_data=None,
    ) -> PerceptionModel:
        # Prefer sensors.rgb / sensors.lidar; unpacked args remain for convenience
        return perception_model
```

```python
from avlite.c10_perception.c12_perception_strategy import TrackingStrategy
from avlite.c50_common.c51_capabilities import WorldCapability, StackCapability
from avlite.c10_perception.c11_perception_model import PerceptionModel

class MyTracker(TrackingStrategy):
    world_requirements = frozenset()
    stack_requirements = frozenset({StackCapability.DETECTION})
    stack_capabilities = frozenset({StackCapability.TRACKING})

    def track(self, perception_model=None, sensors=None) -> PerceptionModel:
        # Your tracking logic here
        return perception_model
```

```python
from avlite.c10_perception.c12_perception_strategy import PredictionStrategy
from avlite.c50_common.c51_capabilities import WorldCapability, StackCapability
from avlite.c10_perception.c11_perception_model import PerceptionModel

class MyPredictor(PredictionStrategy):
    world_requirements = frozenset()
    stack_requirements = frozenset({StackCapability.DETECTION, StackCapability.TRACKING})
    stack_capabilities = frozenset({StackCapability.PREDICTION})

    def predict(self, perception_model=None, sensors=None) -> PerceptionModel | None:
        # Your prediction logic here
        return perception_model
```

Configure pipeline sub-strategies under the `c10_perception:` section of `configs/<profile>.yaml`:

```yaml
c10_perception:
  c12_detection_strategy: MyDetector
  c12_tracking_strategy: MyTracker
  c12_prediction_strategy: MyPredictor
```

## 4. Example: Custom Localization

Localization strategies estimate the ego vehicle’s pose and update
`self.perception_model.ego_vehicle` **in-place** (no return value).

```python
from avlite.c10_perception.c13_localization_strategy import LocalizationStrategy
from avlite.c50_common.c51_capabilities import WorldCapability, StackCapability

class MyLocalization(LocalizationStrategy):
    def __init__(self, perception_model, setting=None):
        super().__init__(perception_model, setting)

    world_requirements = frozenset({WorldCapability.LIDAR_3D})
    stack_requirements = frozenset()
    stack_capabilities = frozenset({StackCapability.LOCALIZATION})
    
    def localize(self, perception_model=None, sensors=None) -> None:
        # Estimate the ego pose from sensors and update in-place
        if perception_model is not None:
            self.perception_model = perception_model
        if sensors is not None and sensors.lidar is not None:
            # ... your scan-matching / localization logic ...
            self.perception_model.ego_vehicle.x = estimated_x
            self.perception_model.ego_vehicle.y = estimated_y
            self.perception_model.ego_vehicle.theta = estimated_theta
    
    def reset(self):
        pass
```

## 5. Example: Custom Planner

Local planning strategies (and behavioral / path / velocity sub-stages) must declare
`world_requirements`, `stack_requirements`, and `stack_capabilities` explicitly.
Use `MayUse(...)` for soft deps the planner utilizes when present (e.g. detections
and prediction trajectories for collision sweeps) but does not require for assembly.

In the visualizer, click **ⓘ** (or right-click the Combobox) to inspect a strategy’s
contract against the live stack and world bridge.

```python
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
from avlite.c50_common.c51_capabilities import MayUse, StackCapability, WorldCapability

class MyLocalPlanner(LocalPlanningStrategy):
    world_requirements = frozenset()
    stack_requirements = frozenset({
        StackCapability.GLOBAL_PLAN,
        StackCapability.LOCALIZATION,
        MayUse(StackCapability.DETECTION, StackCapability.PREDICTION),
    })
    stack_capabilities = frozenset({StackCapability.LOCAL_PLAN})

    def replan(self, perception_model=None, sensors=None):
        # Your local planning logic; use self.pm / perception_model for agents
        pass
```

Hard OR requirements use `AnyOf` (e.g. LiDAR 2D or 3D). Soft optional deps use `MayUse`.

## 6. Example: Custom Controller

```python
from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
from avlite.c20_planning.c21_planning_model import GlobalPlan, LocalPlan
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c30_control.c31_control_model import ControlCommand, ControlCommandBase
from avlite.c50_common.c52_world_sensor_datatypes import SensorFrame

class MyController(ControlStrategy):
    def control(
        self,
        ego: EgoState,
        plan: GlobalPlan | LocalPlan | None = None,
        control_dt=None,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
    ) -> ControlCommandBase:
        if plan is not None:
            self.set_plan(plan)
        return ControlCommand(steer=0.0, acceleration=1.0)

    def reset(self):
        pass
```

Use `set_trajectory_tracker(tj)` when you already have a built `TrajectoryTracker`; use `set_plan(plan)` or the `plan` argument on `control()` when you hold a `GlobalPlan` or `LocalPlan`. The executer also passes optional `perception_model` and `sensors`.

## 7. Multi-robot agents and control

AVLite separates **what an agent is** (platform metadata) from **how it is actuated** (control command payload). Phase 1 adds the structure; the executer still drives ego via `control_ego_state` only. Multi-agent actuation and physics sub-stepping are reserved for later.

### Agent identity

| ID | Role |
|----|------|
| `EGO_AGENT_ID = 0` | Ego vehicle (`EgoState` in `perception_model.ego_vehicle`) |
| `1, 2, 3, …` | NPCs in `perception_model.agent_vehicles` (assigned by `add_agent_vehicle`) |

```python
from avlite.c10_perception.c11_perception_model import AgentState, AgentType, EGO_AGENT_ID

npc = AgentState(x=10.0, y=0.0, agent_type=AgentType.DIFF_DRIVE)
agent_id = perception_model.add_agent_vehicle(npc)  # returns 1, 2, ...
```

IDs are stable integers — not `Enum.auto()` values and not random — so bridges, logs, and tracking stay debuggable.

### Two layers (do not conflate)

```mermaid
flowchart LR
  AgentType["AgentType on AgentState\nplatform metadata"]
  CmdClass["ControlCommandBase subclass\nactuation payload"]
  Map["control_type_for_agent()\nc53_stack_datatypes.py"]
  Bridge["WorldBridge.control_type(agent)"]
  AgentType --> Map --> CmdClass
  Bridge --> Map
```

- **`AgentType`** — platform category on [`AgentState`](../avlite/c10_perception/c11_perception_model.py) (Ackermann car, diff-drive, aerial, pedestrian, …).
- **Control command classes** — actuation payload dataclasses in [`c31_control_model.py`](../avlite/c30_control/c31_control_model.py); default `AgentType` → command mapping in [`c53_stack_datatypes.py`](../avlite/c50_common/c53_stack_datatypes.py).

### Default control mapping

| `AgentType` | Command class | Fields |
|-------------|---------------|--------|
| `ACKERMANN` | `AckermannControlCommand` | `steer`, `acceleration` |
| `DIFF_DRIVE`, `CYCLIST` | `DiffDriveControlCommand` | `linear`, `angular` |
| `AERIAL`, `SURFACE_VESSEL`, `UNDERWATER`, `PEDESTRIAN`, `DYNAMIC_OBJECT` | `BodyVelocityControlCommand` | `vx`, `vy`, `vz`, `yaw_rate` |

Set `agent_type` when spawning non-car NPCs. Do not infer platform type from `agent_id` — use `agent.agent_type`.

### Backward compatibility

- `ControlCommand` and `ControlComand` remain aliases for **`AckermannControlCommand`** only.
- Existing car stack code needs no changes; polymorphic APIs use **`ControlCommandBase`**.
- Use `isinstance(cmd, ControlCommandBase)` for any command type; `isinstance(cmd, ControlCommand)` matches Ackermann only.

### WorldBridge API (phase 1 vs future)

| Method / field | Phase 1 today | Future |
|--------|---------------|--------|
| `control_ego_state(cmd)` | Required; all bridges implement this | Unchanged |
| `map` | Optional `Map \| None` for **simulation** (e.g. LiDAR geometry); does not advertise stack `MAP_HD` / `MAP_RACE_TRACK` — use `MapReader` / mapping module for that | Unchanged |
| `control_type(agent)` | Default: `control_type_for_agent(agent)` | Override only for bridge-specific exceptions |
| `control_agent(id, cmd)` | Default: ego delegates to `control_ego_state`; NPC raises `NotImplementedError` | Override + declare `WorldCapability.AGENT_CONTROL` |
| `teleport_agent(agent_state)` | Default: ego delegates to `teleport_ego` using pose (`x`, `y`, `theta`) from `agent_state`; NPC raises `NotImplementedError`. Identity is `agent_state.agent_id`; velocity/size/type are not applied | Override for sim teleport of any agent |
| `get_*(agent_id=EGO_AGENT_ID)` | Default: ego returns data or `None`; NPC raises `NotImplementedError` | Per-agent sensors in Carla / ROS bridges |
| `get_sensor_frame(agent_id=...)` | Ego: calls legacy `get_*()` with no kwargs (BasicSim-compatible) | Non-ego: passes `agent_id` to each getter |
| `step(dt)` | Default no-op; executer does not call it yet | Physics tick with held command; executer sub-stepping |

`control_type(agent)` lives on **`WorldBridge` only** — not on `ControlStrategy`. The bridge knows what actuation format the sim or robot accepts; the controller expresses what it computes via the return type of `control()`.

**Multi-agent sensors:** override getters with an `agent_id` parameter when your bridge serves more than ego. Ego-only bridges (e.g. BasicSim) need no update — `get_sensor_frame()` uses the legacy no-kwargs call path for ego.

### State model — today vs future

**Today:** `AgentState` uses pose (`x`, `y`, `z`, `theta`) plus scalar **`velocity`** (speed along heading). This matches the car-centric stack (planning, collision checking, BasicSim, visualization).

**Aerial / holonomic agents:** `BodyVelocityControlCommand` is defined, but base `AgentState` does not yet carry `vx`, `vy`, or `vz`. For 2D / bird's-eye use cases, heading plus scalar speed is an acceptable lite projection. Full drone or multirotor simulation needs richer state later.

**Future pattern:** subclass when kinematics diverge — for example `DroneAgentState(AgentState)` with body or world velocity fields and custom `predict()` / integration. Same idea as `EgoState(AgentState)`. Lists typed as `list[AgentState]` accept subclasses. Keep **`agent_type`** for dispatch; use the **subclass** for extra fields and integration logic.

### Plugin author checklist

- Set `agent_type` at spawn for non-car NPCs.
- Return the command type your controller produces; built-in controllers still return Ackermann today.
- Bridge: implement only what you need now (`control_ego_state`); opt into `control_agent` and `AGENT_CONTROL` when the sim supports NPC actuation.
- Do not branch on `agent_id` heuristics for platform type — use `agent.agent_type`.
- Converters (Ackermann → diff-drive, etc.) are **not in core yet**; keep them in your plugin until a shared module (e.g. `c38_control_converters.py`) lands.

### Prediction models on `PerceptionModel`

Forecast payloads live on a single typed object: **`perception_model.prediction`**. Per-agent types store data in **`dict[int, …]` keyed by `agent_id`** (not list index). The lump-sum occupancy type is **`AggregatedOccupancyFlow`** (one grid sequence for the whole scene).

```python
from avlite.c10_perception.c11_perception_model import PerceptionModel, SingleTrajectory

pm.prediction = SingleTrajectory(
    predict_delta_t=0.1,
    trajectories={agent.agent_id: path_xy for agent in pm.agent_vehicles},
)
path = pm.prediction.trajectories.get(agent.agent_id)  # [n_steps, 2] world x,y [m]
```

**Timesteps — do not mix tracking and prediction:**

| YAML key | Stage | Used by |
|----------|-------|---------|
| `c11_predict_delta_t` | Forecast | Default `predict_delta_t` on prediction objects; collision step indexing |
| `c15_tracking_dt` | Tracking | Kalman filter only (`KalmanTracker._dt`) |

## 8. Example: Custom World Bridge

```python
from avlite.c10_perception.c11_perception_model import EGO_AGENT_ID
from avlite.c40_execution.c41_world_bridge import WorldBridge
from avlite.c30_control.c31_control_model import ControlCommandBase
from avlite.c50_common.c51_capabilities import StackCapability, WorldCapability

class MyBridge(WorldBridge):
    world_capabilities = frozenset({WorldCapability.LIDAR_2D})
    # Ground truth provided by the world (satisfies downstream stack_requirements)
    stack_capabilities = frozenset({StackCapability.LOCALIZATION})
    # Optional: what the bridge needs from the stack (BasicSim uses CONTROL)
    stack_requirements = frozenset({StackCapability.CONTROL})

    def control_ego_state(self, cmd: ControlCommandBase, dt=0.01):
        # Send control to your simulator or robot
        pass

    # Optional (future): multi-agent fleets — declare WorldCapability.AGENT_CONTROL
    def control_agent(self, agent_id: int, cmd: ControlCommandBase, dt=0.01):
        if agent_id == EGO_AGENT_ID:
            return self.control_ego_state(cmd, dt=dt)
        # Resolve agent, optionally convert cmd, actuate in sim
        ...
```

`control_type(agent)` defaults to `control_type_for_agent(agent)` from [`c53_stack_datatypes.py`](../avlite/c50_common/c53_stack_datatypes.py). Use `world.step(dt)` to advance the sim without a new command from the stack (default no-op until a bridge overrides it and the executer wires sub-stepping).

## 9. Export Classes

```python
# __init__.py
from .my_strategy import MyPerception, MyLocalization, MyController
from .settings import PluginSettings

__all__ = ["MyPerception", "MyLocalization", "MyController", "PluginSettings"]
```

When you rename a module file, update the import path in `__init__.py` to match (e.g. `from .p31_joystick_controller import JoystickController`).

## 10. Register Your Community Plugin

**Via GUI** (recommended):
1. Open AVLite
2. Go to Config tab
3. Add entry under community plugins: `my_plugin` → install path (or use **Install** then **Register** from `python -m avlite plugins`)
4. Save profile

Double-click a plugin in the list to view its **Settings file** and **Source file location** separately. **Reset to Installed** repopulates the list from plugin directories under the user install dir (`~/.local/share/avlite/plugins/`).

**Via settings file** (the `c69_apps` section of `configs/<profile>.yaml`, or your saved copy under `~/.config/avlite/`):

```yaml
c69_apps:
  c62_community_plugins:
    my_plugin: my_plugin                  # installed under ~/.local/share/avlite/plugins/
    dev_plugin: ~/src/my_plugin           # local dev checkout outside the plugins dir
```

The map value is the **install path**, not the settings YAML path. When a plugin lives under `~/.local/share/avlite/plugins/` (override install root with `AVLITE_PLUGINS_DIR`), AVLite stores the name sentinel (`my_plugin: my_plugin`). Paths outside that directory are stored as `~/...` or an absolute path. Plugin settings always live in `~/.config/avlite/plugin_<name>.yaml`, independent of the install location.

Your classes will now appear in the UI dropdowns.

## Member plugins (AV-Lab)

The **Members** tab in `python -m avlite plugins` lists plugins from the AV-Lab
private registry. Sign in with GitHub (Device Flow) to browse and install them;
your GitHub account must have access to the registry and to each listed plugin
repository. Install and register work the same as community plugins once you are
signed in. See [Member plugins](overview.md#member-plugins) in the main docs.

## 11. Publish to the community registry (pull request)

To list your plugin in every user's **Community** tab (`python -m avlite plugins`), add it to the official public registry via pull request.

Registry repository: [github.com/AV-Lab/avlite-community-plugins](https://github.com/AV-Lab/avlite-community-plugins)

### Before you open a PR

1. **Test locally** — register the plugin on a profile (section 10) and confirm your strategies appear in the GUI dropdowns and the stack runs.
2. **Public Git repository** — the registry clones your repo; private repos will not install for other users.
3. **Plugin layout** — at minimum:
   ```
   my_cool_planner/
   ├── __init__.py       # exports strategy classes (required for discovery)
   ├── my_planner.py     # your implementation
   └── README.md         # shown in the Plugins browser (recommended)
   ```
4. **Optional** — `settings.py` with `PluginSettings` if you have tunable parameters; `requirements.txt` if you depend on extra pip packages (users install these into their AVLite environment).
5. **Do not commit** a `.venv` inside the plugin repo.

### Registry entry

Fork [avlite-community-plugins](https://github.com/AV-Lab/avlite-community-plugins), add one item under `plugins:` in `plugins.yaml`, and open a pull request:

```yaml
plugins:
  - name: my_perception_plugin
    display_name: My Perception Plugin   # optional
    description: One-line summary of what the plugin does
    repository: https://github.com/your-org/your-plugin-repo
    version: latest              # or a git tag / commit SHA
    author: your-org
    category:
      - PerceptionStrategy
    min_avlite_version: "0.4.5"  # optional
    dependency_notes: ""         # optional
    site_url: ""                 # optional
```

| Field | Required | Notes |
|-------|:--------:|-------|
| `name` | yes | Unique registry id; also the install folder name under `~/.local/share/avlite/plugins/`, the `avlite.plugins.<name>` import path (dashes become underscores), and the `plugin_<name>.yaml` settings basename. Use lowercase with underscores, no spaces, and don't change it once published. |
| `display_name` | no | Human-readable name shown in the Plugins browser and the online plugin store, e.g. `My Perception Plugin`. Omit it to display `name` instead. |
| `description` | yes | Short text in the plugin list. |
| `repository` | yes | HTTPS Git URL (GitHub is supported for README preview in the browser). |
| `version` | yes | `latest` clones the default branch; pin a tag or SHA for reproducible installs. |
| `author` | yes | Display name, handle, or organization. |
| `category` | yes | List of strategy types this plugin provides (see table below). Shown in the Plugins browser **Category** column. |
| `min_avlite_version` | no | Minimum AVLite version (semver, e.g. `0.4.5`). Installs are blocked below it. Omit or leave empty if unknown. |
| `dependency_notes` | no | Extra setup beyond `requirements.txt` (system packages, ROS, simulators). Shown after install. Use `""` when pip-only. |
| `site_url` | no | Project website or documentation page. Adds a **Site** link in the plugin store and an **Open Website** button in the Plugins browser. Use `""` when the repository is the only home. |

**Category values** (use the names from [avlite-community-plugins](https://github.com/AV-Lab/avlite-community-plugins)):

| Category | Use when your plugin implements… |
|----------|----------------------------------|
| `PerceptionStrategy` | Sensing, detection, tracking, segmentation, fusion (includes monolithic perception and pipeline sub-strategies such as `DetectionStrategy`) |
| `LocalizationStrategy` | Pose estimation, SLAM-based localization |
| `MappingStrategy` | Map building, SLAM mapping, environment representation |
| `PlanningStrategy` | Global/local planners, behavior planning, decision-making |
| `ControlStrategy` | Vehicle controllers, actuation |
| `ExecutionStrategy` | Runtime execution, scheduling, orchestration |
| `TaskStrategy` | Stack-extension tasks after the tick (cycle / interval / event) |
| `WorldBridge` | Bridges to simulators, middleware, or external world interfaces |

A plugin can list **multiple** categories if it exports more than one strategy type, e.g. `[PerceptionStrategy, LocalizationStrategy]`.

Keep entries sorted alphabetically by `name` if the registry already follows that convention.

### Pull request checklist

- [ ] Plugin works when registered manually (section 10)
- [ ] Repository is public and cloneable
- [ ] `__init__.py` exports all strategy classes users should select
- [ ] README explains what the plugin provides and any extra setup
- [ ] Registry `name` matches how you refer to the plugin in docs
- [ ] Registry `category` matches the base class(es) you export
- [ ] No secrets, large binaries, committed virtualenv, or `config/` folder with plugin-local YAML profiles in the plugin repo

In the PR description, briefly state what layer(s) the plugin extends (perception, planning, control, bridge, etc.) and link to an example profile or usage steps if helpful.

### After merge

Once the PR is merged to `main`, AVLite fetches the updated registry automatically the next time a user opens **Plugins** (`python -m avlite plugins`). They can **Install**, then **Register** to add the plugin to their active profile (`c62_community_plugins` in the `c69_apps` section of `configs/<profile>.yaml`).

You do not need a new AVLite release for registry-only changes.

### Updating your listing

- **New plugin version** — push to your repo; users click **Update** in the Plugins browser (or reinstall). Bump `version` in `plugins.yaml` if you want to pin a new tag/SHA for fresh installs.
- **Change metadata** — open another PR on avlite-community-plugins to edit `display_name`, `description`, `author`, `category`, `version`, or `site_url`. Avoid changing `name`: it is the install folder and settings-file identifier, so renaming it orphans existing installs.

## 12. Built-in plugin naming (`pNx`)

Built-in plugins under `avlite/plugins/` use a **directory name** and optional **module file names** with a `pNx` prefix:

- **Directory:** `p{layer}{variant}_{description}` — e.g. `p30_controller_joystick`, `p40_executer_ROS2`
- **Module files:** use the same convention when the file belongs to a specific layer — e.g. `p31_joystick_controller.py`, `p42_perception_node.py`

The **first digit after `p`** maps to the log-panel layer toggle:

| Digit | Layer |
|-------|-------|
| 1 | Perception |
| 2 | Planning |
| 3 | Control |
| 4 | Execution |
| 5 | Visualization |
| 6 | Common |

The plugin **directory name** and **module file name** can differ. For example, package `p30_controller_joystick` may contain module `p31_joystick_controller.py`.

Logger names follow Python's `__name__`, e.g. `avlite.plugins.p30_controller_joystick.p31_joystick_controller`. Log routing uses the **first module segment** under the package (`p31_joystick_controller`) before falling back to the directory name.

## 13. Log panel filtering

The visualizer log toolbar provides:

- **Core** — master toggle for all core stack logs (`avlite.c10_*` … `avlite.c50_*`). Does not change the per-layer checkbox states.
- **Plugins** — master toggle for all `avlite.plugins.*` logs.
- **Per-layer checkboxes** (Perception, Planning, Control, Execution, Visualization, Common) — filter core logs and plugin logs routed to that layer.

Plugin logs are routed to a layer toggle as follows:

1. Take the **first module segment** under the plugin package (e.g. `p31_joystick_controller` from `avlite.plugins.p30_controller_joystick.p31_joystick_controller`).
2. If it matches `pNx`, use that digit for the layer.
3. Otherwise fall back to the **plugin directory name** (e.g. `p40_bridge_carla` for `carla_bridge` module).
4. If still no `pNx` match (typical community plugins), the log is shown whenever **Plugins** is on.

| Logger | Module segment | Layer source |
|--------|----------------|--------------|
| `...p30_controller_joystick.p31_joystick_controller` | `p31_joystick_controller` | module → Control |
| `...p40_bridge_carla.carla_bridge` | `carla_bridge` | package fallback → Execution |
| `...p40_executer_ROS2.p42_perception_node` | `p42_perception_node` | module → Execution |
| `...sample_avlite_plugin.test_plugin` | `test_plugin` | no pNx → Plugins master only |

Filtering reads a thread-safe snapshot updated on the main thread only (safe when worker threads emit logs during execution).

## Base Classes Reference

| Base Class | Purpose | Key Method |
|------------|---------|------------|
| `PerceptionStrategy` | Monolithic detection/tracking/prediction | `perceive()` |
| `DetectionStrategy` | Detection sub-strategy (used by `PerceptionPipeline`) | `detect()` |
| `TrackingStrategy` | Tracking sub-strategy (used by `PerceptionPipeline`) | `track()` |
| `PredictionStrategy` | Prediction sub-strategy (used by `PerceptionPipeline`) | `predict()` |
| `LocalizationStrategy` | Localization | `localize()` |
| `MappingStrategy` | Mapping | TBD |
| `LocalPlanningStrategy` | Local planning | `replan()` |
| `GlobalPlannerStrategy` | Global planning | `plan()` |
| `ControlStrategy` | Vehicle control | `control()` |
| `TaskStrategy` | Stack-extension execution tasks | `execute(executer, event=None)` |
| `WorldBridge` | Simulator integration | `control_ego_state()`, `control_type(agent)`, `control_agent()`, `teleport_agent()`, `get_*(agent_id=...)`, `step()` |

## Apps (`AppStrategy`)

CLI and GUI entry points register via :class:`~avlite.c60_apps.c61_app_strategy.AppStrategy` (same auto-register pattern as perception/planning strategies). Subclass as ``class MyToolApp(AppStrategy)``, set ``cli_name`` and ``help``, implement ``run()``, and optionally ``configure_parser()`` for flags or nested subcommands. Built-in p60 entry classes use the ``*App`` suffix (e.g. ``SettingCliApp``, ``VisualizationApp``); the base framework class remains ``AppStrategy``. Importing the module registers the app.

| App | Plugin / module | Command |
|-----|-----------------|---------|
| Visualizer (default) | `p60_visualizer_tk` (`p61_visualizer_app`) | `python -m avlite` |
| Settings GUI | `p60_visualizer_tk` (`p62_setting_app`) | `python -m avlite setting` |
| Plugin manager | `p60_visualizer_tk` (`p63_plugins_app`) | `python -m avlite plugins` |
| Headless runner | `p60_headless_mode` | `python -m avlite headless` |
| Setting CLI | `p60_setting_cli` | `python -m avlite setting-cli` |

Built-in ``p60_*`` plugin packages are imported at startup via ``bootstrap_apps()`` in ``c61_app_strategy``. The merged ``p60_visualizer_tk`` package hosts shared Tk code (``p64_setting_views``, ``p65_ui_lib``, ``settings.py``) and three standalone ``AppStrategy`` entry modules. Stack core (``c10``–``c40``, ``c50_common``) may import ``c61_app_strategy``, ``c64_settings_schema``, ``c68_paths``, and ``c69_settings`` only; it must not import ``p60_*`` Tk apps.

## See Also

Built-in plugins in `avlite/plugins/` (maintained by core team):
- `p60_visualizer_tk` — Tk visualizer GUI, settings window (`avlite setting`), and plugin browser (`avlite plugins`)
- `p60_headless_mode` — Headless terminal dashboard runner
- `p60_setting_cli` — Terminal profile validate/describe/import/export

Additional world bridges (CARLA, Gazebo, ROS2), executers, and controllers are available as optional plugins, installed via `c62_community_plugins` in the `c69_apps` section of `configs/<profile>.yaml`.

Settings for built-in plugins: `configs/plugin_*.yaml` in the repo (same basename under `~/.config/avlite/` when saved). Community plugin settings use the same `plugin_<name>.yaml` basename but live only under `~/.config/avlite/` (no repo default).
