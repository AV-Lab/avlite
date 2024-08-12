import logging
from race_plan_control.plan.trajectory import Trajectory
from abc import ABC, abstractmethod
import logging

log = logging.getLogger(__name__)


class Controller(ABC):
    def __init__(self):
        self.last_steer = None
        self.last_acc = None

    @abstractmethod
    def control(self, cte: float, tj: Trajectory = None):
        pass

    @abstractmethod
    def reset():
        pass

    # TODO: future work
    class ControlComand:
        def __init__(self, steer=0, acc=0):
            self.steer = steer
            self.acc = acc

        def __str__(self):
            return f"Steer: {self.steer:+.2f}, Acc: {self.acc:+.2f}"

        def __repr__(self):
            return self.__str__()
