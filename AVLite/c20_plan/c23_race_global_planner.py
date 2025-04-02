from c20_plan.c21_base_global_planner import BaseGlobalPlanner, GlobalPlan
import json

import logging
log = logging.getLogger(__name__)

class RaceGlobalPlanner(BaseGlobalPlanner):
    def __init__(self):
        super().__init__()

    def plan(self, start: tuple[float, float], goal: tuple[float, float]) -> None:
        pass

    def load_race_line(self, path_to_track:str):
        with open(path_to_track, "r") as f:
            track_data = json.load(f)
            self.global_plan.path = [point[:2] for point in track_data["ReferenceLine"]]  # ignoring z
            self.global_plan.velocity = track_data["ReferenceSpeed"]
            self.global_plan.left_boundary_d = track_data["LeftBound"]
            self.global_plan.right_boundary_d = track_data["RightBound"]
        logging.info(f"Track data loaded from {path_to_track}")

    def save_race_line(self):
        pass
