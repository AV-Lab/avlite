"""Path local-planning stage.

Holds the geometric path planners. Currently a single minimal
:class:`ReferencePathPlanner` that follows the global reference; the lattice
path planners live in :mod:`c28_local_lattice_planners`.
"""

from __future__ import annotations

import logging

from avlite.c10_perception.c12_perception_strategy import PerceptionModel
from avlite.c20_planning.c21_planning_model import GlobalPlan, LocalPlan
from avlite.c20_planning.c23_local_planning_strategy import (
    LocalPathPlanningStrategy,
    LocalPlanningStrategy,
)
from avlite.c20_planning.c29_settings import PlanningSettings, PlanningSettingsSchema
from avlite.c50_common.c51_capabilities import StackCapability

log = logging.getLogger(__name__)


class ReferencePathPlanner(LocalPlanningStrategy, LocalPathPlanningStrategy):
    """Minimal path stage: use the global reference trajectory as the local path.

    Dual-role: usable standalone as a :class:`LocalPlanningStrategy` (falls back
    to the base behaviour of following the global trajectory), or as the path
    stage of :class:`LocalPlanningPipeline` via :meth:`plan_path`.
    """

    def __init__(
        self,
        global_plan: GlobalPlan,
        env: PerceptionModel,
        setting: PlanningSettingsSchema = PlanningSettings,
    ):
        super().__init__(global_plan=global_plan, pm=env, setting=setting)

    world_requirements = frozenset()
    stack_requirements = frozenset({StackCapability.GLOBAL_PLAN, StackCapability.LOCALIZATION})
    stack_capabilities = frozenset({StackCapability.LOCAL_PLAN})

    def replan(
        self,
        perception_model=None,
        sensors=None,
    ):
        # The path is simply the global reference; there is nothing to search.
        if perception_model is not None:
            self.pm = perception_model
        pass

    def plan_path(self, plan: LocalPlan) -> LocalPlan:
        plan.path = list(self.global_trajectory.path)
        plan.velocity = list(self.global_trajectory.velocity)
        # Leave trajectory unset so a downstream velocity stage builds (and may
        # mutate) a fresh tracker rather than the shared global one.
        plan.trajectory = None
        return plan
