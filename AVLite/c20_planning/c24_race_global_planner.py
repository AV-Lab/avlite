import json
import logging
from c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy, GlobalPlan

log = logging.getLogger(__name__)


class RaceGlobalPlanner(GlobalPlannerStrategy):
    def __init__(self):
        super().__init__()


    def plan(self, start: tuple[float, float], goal: tuple[float, float]) -> None:
        pass


