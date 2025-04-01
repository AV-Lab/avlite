from c20_plan.c21_base_global_planner import BaseGlobalPlanner, GlobalPlan


class RaceGlobalPlanner(BaseGlobalPlanner):
    def __init__(self):
        super().__init__()

    def plan(self, start: tuple[float, float], goal: tuple[float, float]) -> None:
        pass
