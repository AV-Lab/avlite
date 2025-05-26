from typing import Optional
from dataclasses import dataclass, field
import logging
import json

from c10_perception.c18_hdmap import HDMap
from c20_planning.c28_trajectory import Trajectory, convert_sd_path_to_xy_path

log = logging.getLogger(__name__)

@dataclass
class GlobalPlan:
    start_point: tuple[float, float] = (0.0, 0.0)
    goal_point: tuple[float, float] = (0.0, 0.0)
    path: list[tuple[float, float]] = field(default_factory=list)
    velocity: list[float] = field(default_factory=list)
    left_boundary_d: list[float] = field(default_factory=list)
    left_boundary_x: list[float] = field(default_factory=list)
    left_boundary_y: list[float] = field(default_factory=list)
    right_boundary_d: list[float] = field(default_factory=list)
    right_boundary_x: list[float] = field(default_factory=list)
    right_boundary_y: list[float] = field(default_factory=list)

    trajectory: Optional[Trajectory] = None
    hdmap: Optional[HDMap] = None  
    
    @classmethod
    def from_file(cls, path_to_track: str) -> "GlobalPlan":
        with open(path_to_track, "r") as f:
            data = json.load(f)
            path = [point[:2] for point in data["ReferenceLine"]]
            velocity=data["ReferenceSpeed"]
            left_boundary_d=data["LeftBound"]
            right_boundary_d=data["RightBound"]
            trajectory = Trajectory(path=path, velocity=velocity)
            left_boundary_x, left_boundary_y = convert_sd_path_to_xy_path(trajectory, trajectory.path_s, left_boundary_d)
            right_boundary_x, right_boundary_y = convert_sd_path_to_xy_path(trajectory, trajectory.path_s, right_boundary_d)
            return cls(
                start_point= path[0],
                goal_point=path[-1],
                path=path,
                velocity=velocity,
                left_boundary_d=left_boundary_d,
                right_boundary_d=right_boundary_d,
                trajectory=trajectory,
                left_boundary_x=left_boundary_x,
                left_boundary_y=left_boundary_y,
                right_boundary_x=right_boundary_x,
                right_boundary_y=right_boundary_y,
            )

# TODO:  
@dataclass 
class LocalPlan:
    path: list[tuple[float, float]] = field(default_factory=list)
    velocity: list[float] = field(default_factory=list)

    trajectory: Optional[Trajectory] = None

    


