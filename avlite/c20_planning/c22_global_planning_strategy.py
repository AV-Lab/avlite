from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import ClassVar

from avlite.c10_perception.c11_perception_model import Map, PerceptionModel
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c50_common.c51_capabilities import StackCapability, StackRequirement, WorldRequirement
from avlite.c50_common.c52_world_sensor_datatypes import SensorFrame

log = logging.getLogger(__name__)


class GlobalPlannerStrategy(ABC):
    """Abstract base for global route planners.

    Entrypoint :meth:`plan` takes optional ``perception_model`` and ``sensors``,
    supplied by the UI/executer. Start and goal are set beforehand via
    :meth:`set_start_goal` (not part of the snapshot pair).
    """
    registry = {}

    world_requirements: ClassVar[frozenset[WorldRequirement]] = frozenset()
    stack_requirements: ClassVar[frozenset[StackRequirement]] = frozenset({StackCapability.LOCALIZATION})
    stack_capabilities: ClassVar[frozenset[StackCapability]] = frozenset({StackCapability.GLOBAL_PLAN})

    def __init__(self, map: Map | None = None):
        self.map = map
        self.global_plan: GlobalPlan = GlobalPlan()
        self.start_point = None
        self.goal_point = None

    @abstractmethod
    def plan(
        self,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
    ) -> GlobalPlan:
        """Plan a path from start to goal.

        Args:
            perception_model: Stack world-state snapshot (e.g. live ego).
                Built-ins may ignore it and use :attr:`start_point` /
                :attr:`goal_point` from :meth:`set_start_goal`.
            sensors: World sensor snapshot for this call (``None`` if unused).

        Returns:
            The computed :class:`GlobalPlan` (also stored on ``self.global_plan``).
        """
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
