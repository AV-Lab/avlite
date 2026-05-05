# Plugin Development

AVLite supports two types of extensions:
- **Built-in extensions** (`avlite/extensions/`): Maintained by the core team
- **Community plugins**: External directories you create and register

This guide covers creating community plugins. Classes inheriting from base strategies automatically register and appear in the UI.

## Community Plugin Structure

Create your plugin anywhere on your system:

```
/path/to/my_plugin/
├── __init__.py      # Export classes
├── settings.py      # ExtensionSettings class
└── my_strategy.py   # Your implementation
```

## 1. Settings File (Required)

```python
# settings.py
class ExtensionSettings:
    exclude = ["exclude", "filepath"]
    filepath: str = "configs/my_plugin.yaml"
    
    # Your parameters (appear in UI automatically)
    my_param: float = 1.0
```

## 2. Example: Custom Perception

```python
from avlite.c10_perception.c12_perception_strategy import PerceptionStrategy
from avlite.c60_common.c62_capabilities import WorldCapability, PerceptionCapability
from .settings import ExtensionSettings

class MyPerception(PerceptionStrategy):
    def __init__(self, perception_model, setting=None):
        super().__init__(perception_model, setting)
    
    @property
    def requirements(self) -> set[WorldCapability]:
        return {WorldCapability.GT_DETECTION}
    
    @property
    def capabilities(self) -> set[PerceptionCapability]:
        return {PerceptionCapability.DETECTION}
    
    def perceive(self, rgb_img=None, depth_img=None, lidar_data=None, 
                 perception_model=None):
        # Your logic here
        return self.perception_model
```

## 3. Example: Custom Detection, Tracking, or Prediction Sub-Strategy

Use `DetectionStrategy`, `TrackingStrategy`, or `PredictionStrategy` when you only need
to implement one stage of the pipeline. These plug into `PerceptionPipeline` and are
selected by name in the `PerceptionSettings`.

```python
from avlite.c10_perception.c12_perception_strategy import DetectionStrategy
from avlite.c60_common.c62_capabilities import WorldCapability
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
from avlite.c60_common.c62_capabilities import WorldCapability
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
from avlite.c60_common.c62_capabilities import WorldCapability
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
from avlite.c60_common.c62_capabilities import WorldCapability, LocalizationCapability

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
from .settings import ExtensionSettings

__all__ = ["MyPerception", "MyLocalization", "MyController", "ExtensionSettings"]
```

## 7. Register Your Community Plugin

**Via GUI** (recommended):
1. Open AVLite
2. Go to Config tab
3. Add entry to Community Extensions: `my_plugin` -> `/path/to/my_plugin`
4. Save profile

**Via settings file** (`configs/c40_execution.yaml`):
```yaml
community_plugins:
  my_plugin: /path/to/my_plugin
```

Your classes will now appear in the UI dropdowns.

## Base Classes Reference

| Base Class | Purpose | Key Method |
|------------|---------|------------|
| `PerceptionStrategy` | Monolithic detection/tracking/prediction | `perceive()` |
| `DetectionStrategy` | Detection sub-strategy (used by `PerceptionPipeline`) | `detect()` |
| `TrackingStrategy` | Tracking sub-strategy (used by `PerceptionPipeline`) | `track()` |
| `PredictionStrategy` | Prediction sub-strategy (used by `PerceptionPipeline`) | `predict()` |
| `LocalizationStrategy` | Localization | `localize()` |
| `MappingStrategy` | Mapping | TBD |
| `LocalPlannerStrategy` | Local planning | `replan()` |
| `GlobalPlannerStrategy` | Global planning | `plan()` |
| `ControlStrategy` | Vehicle control | `control()` |
| `WorldBridge` | Simulator integration | `control_ego_state()` |

## See Also

Built-in extensions in `avlite/extensions/` (maintained by core team):
- `bridge_gazebo` - Gazebo world bridge
- `multi_object_prediction` - Perception with prediction
- `executer_ros` - ROS2 integration
