import json
import logging

import numpy as np

from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c60_common.c63_trajectory_tracker import TrajectoryTracker

log = logging.getLogger(__name__)


class GlobalCenterlineRacePlanner(GlobalPlannerStrategy):
    """A global planner that reads a race-line JSON file with left/right
    boundary coordinates and produces a centre-line path with curvature-adapted
    target velocities.

    Expected JSON format::

        {
            "LeftBound":      [[x, y, z], ...],
            "RightBound":     [[x, y, z], ...],
            "ReferencePoint": [x, y, z]         # optional
        }

    The path is the element-wise midpoint of the two boundary arrays.
    Target speed at each waypoint is capped by the lateral-acceleration limit:

        a_lat = v² · κ  →  v = min(v_max, sqrt(a_lat / κ))
    """

    def __init__(
        self,
        filepath: str,
        max_velocity: float = 10.0,
        max_lateral_accel: float = 5.0,
        margin: float = 1.0,
    ):
        super().__init__()
        self.filepath = filepath
        self.max_velocity = max_velocity
        self.max_lateral_accel = max_lateral_accel
        self.margin = margin

    def plan(self) -> GlobalPlan:
        with open(self.filepath) as f:
            data = json.load(f)

        left = np.array(data["LeftBound"])[:, :2]
        right = np.array(data["RightBound"])[:, :2]

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
        eff_left = left + self.margin * dir_unit
        eff_right = right - self.margin * dir_unit

        path_np = (eff_left + eff_right) / 2.0
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
        return self.global_plan

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

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
