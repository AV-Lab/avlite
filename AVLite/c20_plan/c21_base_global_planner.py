class BaseGlobalPlanner:
    start:tuple[float,float]
    goal:tuple[float,float]
    path: list[tuple[float,float]]
    velocity: list[float]
    left_boundary_d: list[float]
    right_boundary_d: list[float]

    def __init__(self):
        pass

    def plan(self, start:tuple[float,float], goal:tuple[float,float], velocity: list[float]) -> list[tuple[float,float]]:
        pass
