from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum


class PlannerType(Enum):
    RACE_PLANNER = "Race Planner"
    HD_MAP_PLANNER = "HD Map Plannr"


@dataclass
class GlobalPlan:
    start: tuple[float, float] = (0.0, 0.0)
    goal: tuple[float, float] = (0.0, 0.0)
    path: list[tuple[float, float]] = field(default_factory=list)
    velocity: list[float] = field(default_factory=list)
    left_boundary_d: list[float] = field(default_factory=list)
    right_boundary_d: list[float] = field(default_factory=list)
    left_boundary_path: list[tuple[float, float]] = field(default_factory=list)
    right_boundary_path: list[tuple[float, float]] = field(default_factory=list)

class BaseGlobalPlanner(ABC):
    registry = {}

    def __init__(self):
        self.global_plan: GlobalPlan = GlobalPlan()

    @abstractmethod
    def plan(self, start: tuple[float, float], goal: tuple[float, float]) -> None:
        pass

    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:  # only register non-abstract subclasses
            BaseGlobalPlanner.registry[cls.__name__] = cls



