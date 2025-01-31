from __future__ import annotations
import logging
from typing import Optional

from c10_perceive.c12_state import EgoState
from c20_plan.c24_trajectory import Trajectory
from abc import ABC, abstractmethod
import logging

log = logging.getLogger(__name__)


class ControlComand:
    steer: float
    acceleration: float

    def __init__(self, steer=0, acc=0):
        self.steer = steer
        self.acceleration = acc

    def __str__(self):
        return f"Steer: {self.steer:+.2f}, Acc: {self.acceleration:+.2f}"

    def __repr__(self):
        return self.__str__()


class BaseController(ABC):
    tj: Optional[Trajectory]

    def __init__(self, tj: Trajectory = None):
        self.tj = tj

    @abstractmethod
    def control(self, ego: EgoState, tj: Trajectory=None) -> ControlComand:
        pass
    

    def update_trajectory(self, tj: Trajectory):
        self.tj = tj

    @abstractmethod
    def reset(self):
        pass
