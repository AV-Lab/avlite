from dataclasses import dataclass, field

@dataclass
class GlobalPlan:
    start: tuple[float, float] = (0.0, 0.0)
    goal: tuple[float, float] = (0.0, 0.0)
    path: list[tuple[float, float]] = field(default_factory=list)
    velocity: list[float] = field(default_factory=list)
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
    path: list[tuple[float, float]] = field(default_factory=list)
    velocity: list[float] = field(default_factory=list)

    


