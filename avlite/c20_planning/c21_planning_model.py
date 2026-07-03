from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
import logging
import json

from avlite.c10_perception.c11_perception_model import HDMap
from avlite.c20_planning.c29_settings import PlanningSettings
from avlite.c60_common.c63_trajectory_tracker import TrajectoryTracker, convert_sd_path_to_xy_path

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
    
    lane_left_boundary_d: list[float] = field(default_factory=list)
    lane_left_boundary_x: list[float] = field(default_factory=list)
    lane_left_boundary_y: list[float] = field(default_factory=list)
    lane_right_boundary_d: list[float] = field(default_factory=list)
    lane_right_boundary_x: list[float] = field(default_factory=list)
    lane_right_boundary_y: list[float] = field(default_factory=list)

    race_mode: bool = True
    trajectory: TrajectoryTracker = field(default_factory=lambda: TrajectoryTracker(path=[], velocity=[]))

    # Optional HDMap and lane path for global planning
    hdmap: Optional[HDMap] = None  
    lane_path: Optional[list[HDMap.Lane]] = None

    @staticmethod
    def is_loadable(path: str | Path) -> bool:
        path = Path(path)
        if path.suffix.lower() != ".json" or not path.is_file():
            return False
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        if not all(k in data for k in ("ReferenceLine", "ReferenceSpeed", "LeftBound", "RightBound")):
            return False
        ref_line = data["ReferenceLine"]
        ref_speed = data["ReferenceSpeed"]
        left = data["LeftBound"]
        right = data["RightBound"]
        if not ref_line or not ref_speed or not left or not right:
            return False
        if not isinstance(ref_line[0], list) or len(ref_line[0]) < 2:
            return False
        try:
            float(ref_line[0][0])
            float(ref_line[0][1])
        except (TypeError, ValueError, IndexError):
            return False
        if isinstance(left[0], list) or isinstance(right[0], list):
            return False
        try:
            float(left[0])
            float(right[0])
        except (TypeError, ValueError):
            return False
        return True
    
    @classmethod
    def from_file(cls, path_to_track: str) -> "GlobalPlan":
        with open(path_to_track, "r") as f:
            data = json.load(f)
            path = [point[:2] for point in data["ReferenceLine"]]
            velocity = data["ReferenceSpeed"]
            if velocity and velocity[0] <= 0:
                velocity = list(velocity)
                velocity[0] = PlanningSettings.c27_min_ramp_start_velocity
            left_boundary_d=data["LeftBound"]
            right_boundary_d=data["RightBound"]
            trajectory = TrajectoryTracker(path=path, velocity=velocity)
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

    def to_file(self, path_to_track: str) -> None:
        import os
        data = {
            "ReferenceLine": [[x, y, 0.0] for x, y in self.path],
            "ReferenceSpeed": list(self.velocity),
        }
        if self.left_boundary_d:
            data["LeftBound"] = list(self.left_boundary_d)
        if self.right_boundary_d:
            data["RightBound"] = list(self.right_boundary_d)
        dirname = os.path.dirname(path_to_track)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(path_to_track, "w") as f:
            json.dump(data, f, indent=2)
        log.info("Global plan saved to %s", path_to_track)

@dataclass
class LocalPlan:
    """Minimal local-planning output consumed by the control layer.

    A plan is defined either by an explicit ``trajectory`` (a fully built
    ``TrajectoryTracker``) or by raw ``path``/``velocity`` samples. When only
    raw samples are given, ``as_trajectory()`` builds the tracker on demand.
    """
    path: list[tuple[float, float]] = field(default_factory=list)
    velocity: list[float] = field(default_factory=list)

    trajectory: Optional[TrajectoryTracker] = None

    @classmethod
    def from_trajectory(cls, trajectory: TrajectoryTracker) -> "LocalPlan":
        """Wrap an existing trajectory in a LocalPlan."""
        return cls(path=list(trajectory.path), velocity=list(trajectory.velocity), trajectory=trajectory)

    def as_trajectory(self) -> Optional[TrajectoryTracker]:
        """Return the plan's trajectory, building one from path/velocity if needed."""
        if self.trajectory is not None:
            return self.trajectory
        if len(self.path) == 0:
            return None
        self.trajectory = TrajectoryTracker(path=self.path, velocity=self.velocity)
        return self.trajectory


