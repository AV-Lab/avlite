from __future__ import annotations
import logging
from typing import Optional

from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c60_common.c63_trajectory_tracker import TrajectoryTracker
from avlite.c20_planning.c21_planning_model import LocalPlan
from avlite.c30_control.c31_control_model import ControlCommand, ControlCommandBase
from avlite.c30_control.c39_settings import ControlSettings
from abc import ABC, abstractmethod
import logging

log = logging.getLogger(__name__)


class ControlStrategy(ABC):
    registry = {}

    def __init__(self, tj: Optional[TrajectoryTracker] = None):
        self.tj: Optional[TrajectoryTracker] = tj
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


    def set_trajectory(self, tj: TrajectoryTracker):
        log.debug("Controller Trajectory updated")
        self.tj = tj

    def set_plan(self, plan: LocalPlan):
        """Set the active trajectory from a LocalPlan."""
        self.tj = plan.as_trajectory() if plan is not None else None

    @abstractmethod
    def control(self, ego: EgoState, plan: Optional[LocalPlan]=None, control_dt:float=None) -> ControlCommandBase:
        pass


    @abstractmethod
    def reset(self):
        pass
    

    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:  
            ControlStrategy.registry[cls.__name__] = cls
    
