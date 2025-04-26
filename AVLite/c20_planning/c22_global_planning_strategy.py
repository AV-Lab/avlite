from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import networkx as nx

from c20_planning.c21_planning_model import GlobalPlan


class GlobalPlannerStrategy(ABC):
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
        self.global_plan.start = start_point
        self.global_plan.goal = goal_point
        
    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:  # only register non-abstract subclasses
            GlobalPlannerStrategy.registry[cls.__name__] = cls



