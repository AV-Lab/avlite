# Plugin Development

AVLite supports two types of plugins:
- **Built-in plugins** (`avlite/plugins/`): Maintained by the core team
- **Community plugins**: External directories you create and register

This guide covers creating community plugins. Classes inheriting from base strategies automatically register and appear in the UI.

## Community Plugin Structure

Create your plugin anywhere on your system:

```
/path/to/my_plugin/
├── __init__.py      # Export classes
├── settings.py      # PluginSettings class
└── my_strategy.py   # Your implementation
```

Do not commit a `.venv` inside your plugin directory — AVLite scans all `.py` files under the plugin path and skips common vendor folders (`.venv`, `site-packages`, etc.), but keeping the venv outside the plugin tree is cleaner.

## 1. Settings File (Required)

```python
# settings.py
class PluginSettings:
    exclude = ["exclude", "filepath"]
    filepath: str = "configs/plugin_my_plugin.yaml"
    
    # Your parameters (appear in UI automatically)
    my_param: float = 1.0
```

## 2. Example: Custom Perception

```python
from avlite.c10_perception.c12_perception_strategy import PerceptionStrategy
from avlite.c60_common.c61_capabilities import WorldCapability, PerceptionCapability
from .settings import PluginSettings

class MyPerception(PerceptionStrategy):
    def __init__(self, perception_model, setting=None):
        super().__init__(perception_model, setting)
    
    @property
    def requirements(self) -> set[WorldCapability]:
        return {WorldCapability.CAMERA_RGB, WorldCapability.LIDAR_3D}
    
    @property
    def capabilities(self) -> set[PerceptionCapability]:
        return {PerceptionCapability.DETECTION, PerceptionCapability.TRACKING,
                PerceptionCapability.PREDICTION}
    
    def perceive(self, rgb_img=None, depth_img=None, lidar_data=None,
                 perception_model=None):
        # Fuse camera and LiDAR to detect, track, and predict agents
        # Update self.perception_model.agents in-place, then return it
        return self.perception_model
```

## 3. Example: Custom Detection, Tracking, or Prediction Sub-Strategy

Use `DetectionStrategy`, `TrackingStrategy`, or `PredictionStrategy` when you only need
to implement one stage of the pipeline. These plug into `PerceptionPipeline` and are
selected by name in the `PerceptionSettings`.

```python
from avlite.c10_perception.c12_perception_strategy import DetectionStrategy
from avlite.c60_common.c61_capabilities import WorldCapability
from avlite.c10_perception.c11_perception_model import PerceptionModel

class MyDetector(DetectionStrategy):
    @property
    def requirements(self) -> set[WorldCapability]:
        return {WorldCapability.CAMERA_RGB}

    def detect(self, perception_model: PerceptionModel,
               rgb_img=None, depth_img=None, lidar_data=None) -> PerceptionModel:
        # Your detection logic here
        return perception_model
```

```python
from avlite.c10_perception.c12_perception_strategy import TrackingStrategy
from avlite.c60_common.c61_capabilities import WorldCapability
from avlite.c10_perception.c11_perception_model import PerceptionModel

class MyTracker(TrackingStrategy):
    @property
    def requirements(self) -> set[WorldCapability]:
        return set()

    def track(self, perception_model: PerceptionModel) -> PerceptionModel:
        # Your tracking logic here
        return perception_model
```

```python
from avlite.c10_perception.c12_perception_strategy import PredictionStrategy
from avlite.c60_common.c61_capabilities import WorldCapability
from avlite.c10_perception.c11_perception_model import PerceptionModel

class MyPredictor(PredictionStrategy):
    @property
    def requirements(self) -> set[WorldCapability]:
        return set()

    def predict(self, perception_model: PerceptionModel) -> PerceptionModel | None:
        # Your prediction logic here
        return perception_model
```

To use these sub-strategies with `PerceptionPipeline`, set the appropriate fields in
`configs/c10_perception.yaml`:

```yaml
detection_strategy: MyDetector
tracking_strategy: MyTracker
prediction_strategy: MyPredictor
```

## 4. Example: Custom Localization

Localization strategies estimate the ego vehicle’s pose and update
`self.perception_model.ego_vehicle` **in-place** (no return value).

```python
from avlite.c10_perception.c13_localization_strategy import LocalizationStrategy
from avlite.c60_common.c61_capabilities import WorldCapability, LocalizationCapability

class MyLocalization(LocalizationStrategy):
    def __init__(self, perception_model, setting=None):
        super().__init__(perception_model, setting)
    
    @property
    def requirements(self) -> set[WorldCapability]:
        return {WorldCapability.LIDAR_3D}
    
    @property
    def capabilities(self) -> set[LocalizationCapability]:
        return {LocalizationCapability.LOCALIZATION_2D, LocalizationCapability.LOCALIZATION_HEADING}
    
    def localize(self, imu=None, lidar=None, rgb_img=None) -> None:
        # Estimate the ego pose from sensor data and update in-place
        if lidar is not None:
            # ... your scan-matching / localization logic ...
            self.perception_model.ego_vehicle.x = estimated_x
            self.perception_model.ego_vehicle.y = estimated_y
            self.perception_model.ego_vehicle.theta = estimated_theta
    
    def reset(self):
        pass
```

## 5. Example: Custom Controller

```python
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c30_control.c31_control_model import ControlComand

class MyController(ControlStrategy):
    def control(self, ego, tj=None, control_dt=None) -> ControlComand:
        # Your logic here
        return ControlComand(throttle=1.0, steer=0.0)
    
    def reset(self):
        pass
```

## 6. Export Classes

```python
# __init__.py
from .my_strategy import MyPerception, MyLocalization, MyController
from .settings import PluginSettings

__all__ = ["MyPerception", "MyLocalization", "MyController", "PluginSettings"]
```

When you rename a module file, update the import path in `__init__.py` to match (e.g. `from .p31_joystick_controller import JoystickController`).

## 7. Register Your Community Plugin

**Via GUI** (recommended):
1. Open AVLite
2. Go to Config tab
3. Add entry under community plugins: `my_plugin` -> `/path/to/my_plugin`
4. Save profile

**Via settings file** (`configs/c40_execution.yaml` or your saved copy under `~/.config/avlite/`):

```yaml
c40_community_plugins:
  my_plugin: /path/to/my_plugin
```

When a plugin is installed through `python -m avlite plugins`, its path is stored under `~/.local/share/avlite/plugins/` (override with `AVLITE_PLUGINS_DIR`).

Your classes will now appear in the UI dropdowns.

## 8. Built-in plugin naming (`pNx`)

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

## 9. Log panel filtering

The visualizer log toolbar (`c55_log_view`) provides:

- **Core** — master toggle for all core stack logs (`avlite.c10_*` … `avlite.c60_*`). Does not change the per-layer checkbox states.
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
| `WorldBridge` | Simulator integration | `control_ego_state()` |

## See Also

Built-in plugins in `avlite/plugins/` (maintained by core team):
- `p40_bridge_carla` — CARLA simulator world bridge
- `p40_bridge_gazebo` — Gazebo Ignition world bridge
- `p40_bridge_ROS2` — ROS2 world bridge
- `p40_executer_ROS2` — ROS2 executor with Autoware message support
- `p10_perception_MO_prediction` — Multi-object prediction perception
- `p30_controller_joystick` — Xbox-style joystick controller
- `p50_headless_mode` — Headless runner and config CLI

Settings for built-in plugins: `configs/plugin_*.yaml` in the repo (same basename under `~/.config/avlite/` when saved).
