import logging

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

from avlite.c10_perception.c11_perception_model import RaceMap
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c20_planning.c29_settings import PlanningSettings
from avlite.c60_common.c63_trajectory_tracker import TrajectoryTracker
from avlite.c60_common.c67_paths import DataPaths

log = logging.getLogger(__name__)


class GlobalCenterlineRacePlanner(GlobalPlannerStrategy):
    """A global planner that reads a race-line JSON file with left/right
    boundary coordinates and produces a centre-line path with curvature-adapted
    target velocities.

    Expected JSON format::

        {
            "LeftBound":      [[x, y, z], ...],
            "RightBound":     [[x, y, z], ...],
            "ReferencePoint": [lat, lon, alt]   # required WGS84 degrees
        }

    The path is the corridor centre between the left and right boundary polylines,
    refined from an index-wise midpoint so tight corners stay equidistant to both sides.
    Target speed at each waypoint is capped by the lateral-acceleration limit:

        a_lat = v² · κ  →  v = min(v_max, sqrt(a_lat / κ))
    """

    def __init__(
        self,
        filepath: str | RaceMap,
        max_velocity: float = 10.0,
        max_lateral_accel: float = 5.0,
        margin: float | None = None,
    ):
        super().__init__()
        if isinstance(filepath, RaceMap):
            self._race_map = filepath
            self.filepath = filepath.source_path
        else:
            self._race_map = None
            self.filepath = filepath
        self.max_velocity = max_velocity
        self.max_lateral_accel = max_lateral_accel
        self.margin = margin

    def plan(self) -> GlobalPlan:
        margin = (
            self.margin
            if self.margin is not None
            else PlanningSettings.c20_boundary_margin
        )

        if self._race_map is not None:
            left = self._race_map.left_bound
            right = self._race_map.right_bound
        else:
            race_map = RaceMap.from_path(DataPaths.resolve_stored(self.filepath))
            left = race_map.left_bound
            right = race_map.right_bound

        if len(left) != len(right):
            raise ValueError(
                f"LeftBound ({len(left)}) and RightBound ({len(right)}) "
                "arrays must have equal length."
            )

        # Apply inward margin: shift each boundary toward the centreline.
        eps = 1e-6
        diff = right - left
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        dir_unit = diff / np.maximum(norms, eps)
        eff_left = left + margin * dir_unit
        eff_right = right - margin * dir_unit

        path_np = (eff_left + eff_right) / 2.0
        path_np = self._refine_centerline_to_corridor(path_np, eff_left, eff_right)
        path = [tuple(p) for p in path_np]
        velocity = self._curvature_velocity(path_np)
        trajectory = TrajectoryTracker(path=path, velocity=velocity)

        self.global_plan = GlobalPlan(
            start_point=path[0],
            goal_point=path[-1],
            path=path,
            velocity=velocity,
            left_boundary_d=[trajectory.convert_xy_to_sd(x, y)[1] for x, y in eff_left],
            right_boundary_d=[trajectory.convert_xy_to_sd(x, y)[1] for x, y in eff_right],
            left_boundary_x=eff_left[:, 0].tolist(),
            left_boundary_y=eff_left[:, 1].tolist(),
            right_boundary_x=eff_right[:, 0].tolist(),
            right_boundary_y=eff_right[:, 1].tolist(),
            trajectory=trajectory,
            race_mode=True,
        )
        log.debug(f"GlobalCenterlineRacePlanner: planned {len(path)} waypoints from {self.filepath}")
        return self.global_plan

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _refine_centerline_to_corridor(
        path_np: np.ndarray, left_np: np.ndarray, right_np: np.ndarray
    ) -> np.ndarray:
        """Re-center path so each point is midway between nearest boundary points."""
        left_ls = LineString(left_np)
        right_ls = LineString(right_np)
        refined = np.empty_like(path_np)
        for i, p in enumerate(path_np):
            pl = nearest_points(Point(p), left_ls)[1]
            pr = nearest_points(Point(p), right_ls)[1]
            refined[i] = [(pl.x + pr.x) / 2.0, (pl.y + pr.y) / 2.0]
        return refined

    def _curvature_velocity(self, path_np: np.ndarray) -> list[float]:
        """Compute per-waypoint speed limited by lateral acceleration.

        From circular motion: a_lat = v² · κ  →  v = sqrt(a_lat / κ).
        Speed is then capped by max_velocity on straights (κ ≈ 0).
        """
        eps = 1e-6
        x, y = path_np[:, 0], path_np[:, 1]
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        kappa = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2) ** 1.5
        kappa = np.maximum(kappa, eps)
        v = np.minimum(self.max_velocity, np.sqrt(self.max_lateral_accel / kappa))
        return v.tolist()
