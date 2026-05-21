from __future__ import annotations
import logging
from typing import Optional

from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c60_common.c63_trajectory_tracker import TrajectoryTracker
from avlite.c30_control.c31_control_model import ControlComand
from avlite.c30_control.c39_settings import ControlSettings
from abc import ABC, abstractmethod
import logging

log = logging.getLogger(__name__)

class ControlStrategy(ABC):
    registry = {}

    def __init__(self, tj: Optional[TrajectoryTracker] = None):
        self.tj: Optional[TrajectoryTracker] = tj
        self.cmd: ControlComand = ControlComand()
        self.cte_steer: float = 0
        self.cte_velocity: float = 0

        # Kinematic constraints — owned by the control layer
        self.ego_distance_front_axle: float = ControlSettings.ego_distance_front_axle
        self.ego_max_velocity: float = ControlSettings.ego_max_velocity
        self.ego_max_acceleration: float = ControlSettings.ego_max_acceleration
        self.ego_min_acceleration: float = ControlSettings.ego_min_acceleration
        self.ego_max_steering: float = ControlSettings.ego_max_steering
        self.ego_min_steering: float = ControlSettings.ego_min_steering


    def set_trajectory(self, tj: TrajectoryTracker):
        log.debug("Controller Trajectory updated")
        self.tj = tj

    @abstractmethod
    def control(self, ego: EgoState, tj: Optional[TrajectoryTracker]=None, control_dt:float=None) -> ControlComand:
        pass


    @abstractmethod
    def reset(self):
        pass
    

    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:  
            ControlStrategy.registry[cls.__name__] = cls
    



    # # methods used for multiprocessing
    # def get_copy(self):
    #     return copy.deepcopy(self)
    # def update_serializable_trajectory(self, path: list[tuple[float, float]], velocity_list: list[float]):
    #     self.tj = Trajectory(path, velocity_list)
    #     log.info("Controller Trajectory updated")
    # def get_control_dt(self)->float:
    #     return self.__control_dt
    # def set_control_dt(self, dt:float):
    #     self.__control_dt = dt
    # def get_cte_steer(self)->float:
    #     return self.cte_steer
    # def get_cte_velocity(self)->float:
    #     return self.cte_velocity
    # def get_cmd(self)->ControlComand:
    #     return self.cmd
