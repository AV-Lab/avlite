from typing import Optional
from dataclasses import dataclass, field
from c20_planning.c28_trajectory import Trajectory
from c20_planning.c27_lattice import Edge

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from c20_planning.c25_hdmap_global_planner import HDMap

@dataclass
class GlobalPlan:
    start_point: tuple[float, float] = (0.0, 0.0)
    goal_point: tuple[float, float] = (0.0, 0.0)
    trajectory: Trajectory = field(default_factory=Trajectory)
    left_boundary_d: list[float] = field(default_factory=list)
    right_boundary_d: list[float] = field(default_factory=list)
    left_boundary_path: list[tuple[float, float]] = field(default_factory=list)
    right_boundary_path: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class GlobalHDPlan:
    start_point: tuple[float, float] = (0.0, 0.0)
    goal_point: tuple[float, float] = (0.0, 0.0)


#TODO still
@dataclass 
class LocalPlan:
    horizon: int = 1
    trajectory: Trajectory = field(default_factory=Trajectory)

    


