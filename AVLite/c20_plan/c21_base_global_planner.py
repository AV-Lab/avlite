from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum

class PlannerType(Enum):
    RACE_PLANNER = "Race Planner"
    HD_MAP_PLANNER = "HD Map Planner"

@dataclass
class GlobalPlan:
    start: tuple[float, float] = (0.0, 0.0)
    goal: tuple[float, float] = (0.0, 0.0)
    path: list[tuple[float, float]] = field(default_factory=list)
    velocity_profile: list[float] = field(default_factory=list)
    left_boundary_d: list[float] = field(default_factory=list)
    right_boundary_d: list[float] = field(default_factory=list)

class BaseGlobalPlanner(ABC):
    global_plan: GlobalPlan

    def __init__(self):
        pass

    @abstractmethod
    def plan(self, start: tuple[float, float], goal: tuple[float, float]) -> None:
        pass
