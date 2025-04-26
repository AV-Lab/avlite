from __future__ import annotations
import logging
from typing import Optional

from c10_perception.c11_perception_model import EgoState
from c20_planning.c28_trajectory import Trajectory
from c30_control.c31_control_model import ControlComand
from abc import ABC, abstractmethod
import logging
import copy 

log = logging.getLogger(__name__)

class ControlStrategy(ABC):
    tj: Optional[Trajectory]
    cmd: ControlComand
    cte_steer: float
    cte_velocity: float

    def __init__(self, tj: Trajectory = None):
        self.tj = tj
        self.cmd = ControlComand()
        self.cte_steer = 0
        self.cte_velocity = 0
        self.__control_dt:float=0

    @abstractmethod
    def control(self, ego: EgoState, tj: Trajectory=None) -> ControlComand:
        pass
    

    def update_trajectory(self, tj: Trajectory):
        log.debug("Controller Trajectory updated")
        self.tj = tj



    @abstractmethod
    def reset(self):
        pass
    
    # methods used for multiprocessing
    def get_copy(self):
        return copy.deepcopy(self)
    def update_serializable_trajectory(self, path: list[tuple[float, float]], velocity_list: list[float]):
        self.tj = Trajectory(path, velocity_list)
        log.info("Controller Trajectory updated")
    def get_control_dt(self)->float:
        return self.__control_dt
    def set_control_dt(self, dt:float):
        self.__control_dt = dt
    def get_cte_steer(self)->float:
        return self.cte_steer
    def get_cte_velocity(self)->float:
        return self.cte_velocity
    def get_cmd(self)->ControlComand:
        return self.cmd
