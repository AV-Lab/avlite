import logging
from abc import ABC, abstractmethod
import networkx as nx

from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c50_common.c51_capabilities import StackCapability, WorldCapability

log = logging.getLogger(__name__)

class GlobalPlannerStrategy(ABC):
    registry = {}

    def __init__(self):
        self.global_plan: GlobalPlan = GlobalPlan()
        self.start_point = None
        self.goal_point = None

    @property
    def world_requirements(self) -> set[WorldCapability]:
        """World (sensor) capabilities this planner requires (default: none)."""
        return set()

    @property
    def stack_requirements(self) -> set[StackCapability]:
        """Upstream stack capabilities a global planner depends on."""
        return {StackCapability.MAP, StackCapability.LOCALIZATION}

    @property
    def stack_capabilities(self) -> set[StackCapability]:
        return {StackCapability.GLOBAL_PLAN}

    @abstractmethod
    def plan(self) -> GlobalPlan:
        """Plan a path from start to goal."""
        pass

    
    def set_start_goal(self, start_point: tuple[float, float], goal_point: tuple[float, float]) -> None:
        """Set start and goal points for the planner."""
        self.global_plan.start_point = start_point
        self.global_plan.goal_point = goal_point
        self.start_point = start_point
        self.goal_point = goal_point
        
    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:  
            GlobalPlannerStrategy.registry[cls.__name__] = cls



