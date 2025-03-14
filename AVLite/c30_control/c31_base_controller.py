from __future__ import annotations
import logging
from typing import Optional

from c10_perceive.c12_state import EgoState
from c20_plan.c26_trajectory import Trajectory
from abc import ABC, abstractmethod
import logging
import copy 

log = logging.getLogger(__name__)


class ControlComand:
    steer: float
    acceleration: float

    def __init__(self, steer:float=0, acc:float=0):
        self.steer = steer
        self.acceleration = acc

    def __str__(self):
        return f"Steer: {self.steer:+.2f}, Acc: {self.acceleration:+.2f}"

    def __repr__(self):
        return self.__str__()


class BaseController(ABC):
    tj: Optional[Trajectory]
    cmd: ControlComand
    cte_steer: float
    cte_velocity: float
    __control_dt:float=0

    def __init__(self, tj: Trajectory = None):
        self.tj = tj
        self.cmd = ControlComand()
        self.cte_steer = 0
        self.cte_velocity = 0

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
