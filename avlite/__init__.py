"""AVLite - Modular Autonomous Vehicle Stack.

The most important classes are re-exported here so that users (and plugin
authors in particular) can import them directly from the top-level package::

    from avlite import ControlStrategy, EgoState, GlobalPlan, WorldCapability

instead of reaching into the ``c10_*``/``c20_*``/... subpackages. Re-exports are
resolved lazily (PEP 562 module ``__getattr__``) so ``import avlite`` stays cheap
and only pulls in a module (and its heavy dependencies) when a name is accessed.
"""

import importlib
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING


def _package_version() -> str:
    try:
        return version("avlite")
    except PackageNotFoundError:
        pass
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if pyproject.is_file():
        match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"), re.M)
        if match:
            return match.group(1)
    return "unknown"


__version__ = _package_version()


# ---------------------------------------------------------------------------
# Curated public API — maps each exported name to its source module.
# Grouped by purpose; loaded lazily via __getattr__ below.
# ---------------------------------------------------------------------------

_LAZY: dict[str, str] = {
    # -- Strategy base classes (primary plugin-development surface) ----------
    # Perception
    "PerceptionStrategy": "avlite.c10_perception.c12_perception_strategy",
    "DetectionStrategy": "avlite.c10_perception.c12_perception_strategy",
    "TrackingStrategy": "avlite.c10_perception.c12_perception_strategy",
    "PredictionStrategy": "avlite.c10_perception.c12_perception_strategy",
    "PerceptionPipeline": "avlite.c10_perception.c12_perception_strategy",
    # Localization
    "LocalizationStrategy": "avlite.c10_perception.c13_localization_strategy",
    # Mapping
    "MappingStrategy": "avlite.c10_perception.c14_mapping_strategy",
    # Global planning
    "GlobalPlannerStrategy": "avlite.c20_planning.c22_global_planning_strategy",
    # Local planning
    "LocalPlanningStrategy": "avlite.c20_planning.c23_local_planning_strategy",
    "LocalBehavioralPlanningStrategy": "avlite.c20_planning.c23_local_planning_strategy",
    "LocalPathPlanningStrategy": "avlite.c20_planning.c23_local_planning_strategy",
    "LocalVelocityPlanningStrategy": "avlite.c20_planning.c23_local_planning_strategy",
    "LocalPlanningPipeline": "avlite.c20_planning.c23_local_planning_strategy",
    # Control
    "ControlStrategy": "avlite.c30_control.c32_control_strategy",
    # Execution
    "ExecutionStrategy": "avlite.c40_execution.c42_execution_strategy",
    "WorldBridge": "avlite.c40_execution.c41_world_bridge",
    # Apps
    "AppStrategy": "avlite.c60_apps.c61_app_strategy",
    # -- Perception / world data models -------------------------------------
    "PerceptionModel": "avlite.c10_perception.c11_perception_model",
    "State": "avlite.c10_perception.c11_perception_model",
    "AgentState": "avlite.c10_perception.c11_perception_model",
    "EgoState": "avlite.c10_perception.c11_perception_model",
    "AgentType": "avlite.c10_perception.c11_perception_model",
    "EGO_AGENT_ID": "avlite.c10_perception.c11_perception_model",
    "Map": "avlite.c10_perception.c11_perception_model",
    "HDMap": "avlite.c10_perception.c11_perception_model",
    "RaceMap": "avlite.c10_perception.c11_perception_model",
    # -- Planning data models -----------------------------------------------
    "GlobalPlan": "avlite.c20_planning.c21_planning_model",
    "LocalPlan": "avlite.c20_planning.c21_planning_model",
    "LocalBehavior": "avlite.c20_planning.c21_planning_model",
    # -- Control commands ---------------------------------------------------
    "ControlCommandBase": "avlite.c30_control.c31_control_model",
    "ControlCommand": "avlite.c30_control.c31_control_model",
    "AckermannControlCommand": "avlite.c30_control.c31_control_model",
    "DiffDriveControlCommand": "avlite.c30_control.c31_control_model",
    "BodyVelocityControlCommand": "avlite.c30_control.c31_control_model",
    # -- Capabilities -------------------------------------------------------
    "WorldCapability": "avlite.c50_common.c51_capabilities",
    "StackCapability": "avlite.c50_common.c51_capabilities",
    "AnyOf": "avlite.c50_common.c51_capabilities",
    "satisfies_requirements": "avlite.c50_common.c51_capabilities",
    # -- Sensor datatypes ---------------------------------------------------
    "SensorFrame": "avlite.c50_common.c52_sensor_datatypes",
    "ImuReading": "avlite.c50_common.c52_sensor_datatypes",
    "GnssReading": "avlite.c50_common.c52_sensor_datatypes",
    "WheelOdometry": "avlite.c50_common.c52_sensor_datatypes",
    "RgbImage": "avlite.c50_common.c52_sensor_datatypes",
    "DepthImage": "avlite.c50_common.c52_sensor_datatypes",
    "LidarCloud": "avlite.c50_common.c52_sensor_datatypes",
    # -- Trajectory ---------------------------------------------------------
    "TrajectoryTracker": "avlite.c50_common.c53_trajectory_tracker",
    # -- Runtime helpers ----------------------------------------------------
    "executor_factory": "avlite.c60_apps.c62_factory",
    "load_stack_settings": "avlite.c60_apps.c62_factory",
    # -- Settings singletons ------------------------------------------------
    "PerceptionSettings": "avlite.c10_perception.c19_settings",
    "PlanningSettings": "avlite.c20_planning.c29_settings",
    "ControlSettings": "avlite.c30_control.c39_settings",
    "ExecutionSettings": "avlite.c40_execution.c49_settings",
    "AppSettings": "avlite.c60_apps.c69_settings",
}


def __getattr__(name: str):
    """Lazily import a curated public name on first access (PEP 562)."""
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # cache so subsequent accesses skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted([*_LAZY, "__version__"])


__all__ = [*sorted(_LAZY), "__version__"]


if TYPE_CHECKING:  # static-analysis / IDE resolution only; no runtime cost
    from avlite.c10_perception.c11_perception_model import (
        EGO_AGENT_ID,
        AgentState,
        AgentType,
        EgoState,
        HDMap,
        Map,
        PerceptionModel,
        RaceMap,
        State,
    )
    from avlite.c10_perception.c12_perception_strategy import (
        DetectionStrategy,
        PerceptionPipeline,
        PerceptionStrategy,
        PredictionStrategy,
        TrackingStrategy,
    )
    from avlite.c10_perception.c13_localization_strategy import LocalizationStrategy
    from avlite.c10_perception.c14_mapping_strategy import MappingStrategy
    from avlite.c10_perception.c19_settings import PerceptionSettings
    from avlite.c20_planning.c21_planning_model import (
        GlobalPlan,
        LocalBehavior,
        LocalPlan,
    )
    from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
    from avlite.c20_planning.c23_local_planning_strategy import (
        LocalBehavioralPlanningStrategy,
        LocalPathPlanningStrategy,
        LocalPlanningPipeline,
        LocalPlanningStrategy,
        LocalVelocityPlanningStrategy,
    )
    from avlite.c20_planning.c29_settings import PlanningSettings
    from avlite.c30_control.c31_control_model import (
        AckermannControlCommand,
        BodyVelocityControlCommand,
        ControlCommand,
        ControlCommandBase,
        DiffDriveControlCommand,
    )
    from avlite.c30_control.c32_control_strategy import ControlStrategy
    from avlite.c30_control.c39_settings import ControlSettings
    from avlite.c40_execution.c41_world_bridge import WorldBridge
    from avlite.c40_execution.c42_execution_strategy import ExecutionStrategy
    from avlite.c40_execution.c49_settings import ExecutionSettings
    from avlite.c50_common.c51_capabilities import (
        AnyOf,
        StackCapability,
        WorldCapability,
        satisfies_requirements,
    )
    from avlite.c50_common.c52_sensor_datatypes import (
        DepthImage,
        GnssReading,
        ImuReading,
        LidarCloud,
        RgbImage,
        SensorFrame,
        WheelOdometry,
    )
    from avlite.c50_common.c53_trajectory_tracker import TrajectoryTracker
    from avlite.c60_apps.c61_app_strategy import AppStrategy
    from avlite.c60_apps.c62_factory import executor_factory, load_stack_settings
    from avlite.c60_apps.c69_settings import AppSettings
