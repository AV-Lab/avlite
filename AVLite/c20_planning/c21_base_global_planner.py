from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import networkx as nx




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
    start_point: tuple[float, float] = (0.0, 0.0)
    goal_point: tuple[float, float] = (0.0, 0.0)



class BaseGlobalPlanner(ABC):
    registry = {}

    def __init__(self):
        self.global_plan: GlobalPlan = GlobalPlan()
        self.graph = nx.DiGraph()

    @abstractmethod
    def plan(self) -> None:
        """Plan a path from start to goal."""
        pass

    
    def set_start_goal(self, start_point: tuple[float, float], goal_point: tuple[float, float]) -> None:
        """Set start and goal points for the planner."""
        self.global_plan.start_point = start_point
        self.global_plan.goal_point = goal_point
        
    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:  # only register non-abstract subclasses
            BaseGlobalPlanner.registry[cls.__name__] = cls



