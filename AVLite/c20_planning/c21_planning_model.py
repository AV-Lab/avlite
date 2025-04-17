from dataclasses import dataclass, field
from c20_planning.c28_trajectory import Trajectory
from c20_planning.c27_lattice import Edge

@dataclass
class GlobalPlan:
    start: tuple[float, float] = (0.0, 0.0)
    goal: tuple[float, float] = (0.0, 0.0)
    trajectory: Trajectory = Trajectory()
    left_boundary_d: list[float] = field(default_factory=list)
    right_boundary_d: list[float] = field(default_factory=list)
    left_boundary_path: list[tuple[float, float]] = field(default_factory=list)
    right_boundary_path: list[tuple[float, float]] = field(default_factory=list)
    start_point: tuple[float, float] = (0.0, 0.0)
    goal_point: tuple[float, float] = (0.0, 0.0)


#TODO still
@dataclass 
class LocalPlan:
    horizon: int = 1
    trajectory: Trajectory = Trajectory()

    


