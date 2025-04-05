from dataclasses import dataclass


@dataclass
class PlanningSettings:
    exclude = []
    filepath: str="configs/c29_planning.yaml"

