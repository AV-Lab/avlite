from __future__ import annotations
import logging
from typing import Optional

from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c20_planning.c28_trajectory import Trajectory
from avlite.c30_control.c31_control_model import ControlComand
from abc import ABC, abstractmethod
import logging

log = logging.getLogger(__name__)

class ControlStrategy(ABC):
    registry = {}

    def __init__(self, tj: Optional[Trajectory] = None):
        self.tj: Optional[Trajectory] = tj
        self.cmd: ControlComand = ControlComand()
        self.cte_steer: float = 0
        self.cte_velocity: float = 0


    def set_trajectory(self, tj: Trajectory):
        log.debug("Controller Trajectory updated")
        self.tj = tj

    @abstractmethod
    def control(self, ego: EgoState, tj: Optional[Trajectory]=None, control_dt:float=None) -> ControlComand:
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
