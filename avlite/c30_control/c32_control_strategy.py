from __future__ import annotations
import logging
from abc import ABC, abstractmethod

from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
from avlite.c50_common.c54_trajectory_tracker import TrajectoryTracker
from avlite.c20_planning.c21_planning_model import GlobalPlan, LocalPlan
from avlite.c30_control.c31_control_model import ControlCommand, ControlCommandBase
from avlite.c30_control.c39_settings import ControlSettings
from avlite.c50_common.c51_capabilities import AnyOf, StackCapability
from avlite.c50_common.c52_world_sensor_datatypes import SensorFrame

log = logging.getLogger(__name__)


class ControlStrategy(ABC):
    """Abstract base for vehicle controllers.

    The tick entrypoint :meth:`control` takes ``ego`` / optional ``plan`` /
    ``control_dt``, plus optional ``perception_model`` and ``sensors`` supplied
    by the executer for stack/world context.
    """
    registry = {}

    world_requirements = frozenset()
    stack_requirements = frozenset({
        AnyOf(StackCapability.GLOBAL_PLAN, StackCapability.LOCAL_PLAN),
        StackCapability.LOCALIZATION,
    })
    stack_capabilities = frozenset({StackCapability.CONTROL})

    def __init__(self, tj: TrajectoryTracker | None = None):

        self.tj: TrajectoryTracker | None = tj
        self.cmd: ControlCommand = ControlCommand()
        self.cte_steer: float = 0
        self.cte_velocity: float = 0

        # Kinematic constraints — owned by the control layer
        self.ego_distance_front_axle: float = ControlSettings.c32_ego_distance_front_axle
        self.ego_max_velocity: float = ControlSettings.c32_ego_max_velocity
        self.ego_max_acceleration: float = ControlSettings.c32_ego_max_acceleration
        self.ego_min_acceleration: float = ControlSettings.c32_ego_min_acceleration
        self.ego_max_steering: float = ControlSettings.c32_ego_max_steering
        self.ego_min_steering: float = ControlSettings.c32_ego_min_steering

    def set_trajectory_tracker(self, tj: TrajectoryTracker | None = None) -> None:
        """Set the active reference path from a built TrajectoryTracker.

        Use this when the caller already holds a tracker (e.g. factory init or
        apply_global_plan). For planning-layer objects, prefer set_plan().
        """
        log.debug("Controller trajectory tracker updated")
        self.tj = tj

    def set_plan(self, plan: GlobalPlan | LocalPlan) -> None:
        """Set the active reference path from a GlobalPlan or LocalPlan.

        Extracts the TrajectoryTracker via plan.as_trajectory(). Use this when
        the caller holds a plan object rather than a raw tracker.
        """
        self.tj = plan.as_trajectory()

    @abstractmethod
    def control(
        self,
        ego: EgoState,
        plan: GlobalPlan | LocalPlan | None = None,
        control_dt: float | None = None,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
    ) -> ControlCommandBase:
        """Run one control step.

        Args:
            ego: Ego pose/velocity for this tick.
            plan: Optional global/local plan; when set, updates ``self.tj``.
            control_dt: Control timestep [s].
            perception_model: Optional stack world-state snapshot.
            sensors: Optional world sensor snapshot (e.g. ``sensors.lidar``).

        Returns:
            Actuation command for the world bridge.
        """
        pass


    @abstractmethod
    def reset(self):
        pass
    

    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:  
            ControlStrategy.registry[cls.__name__] = cls
