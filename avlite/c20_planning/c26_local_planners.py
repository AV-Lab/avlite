from __future__ import annotations

import logging
import math
from typing import Optional, TYPE_CHECKING

import numpy as np

from avlite.c10_perception.c12_perception_strategy import PerceptionModel
from avlite.c20_planning.c21_planning_model import GlobalPlan, LocalPlan
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
from avlite.c20_planning.c29_settings import PlanningSettings, PlanningSettingsSchema
from avlite.c60_common.c63_trajectory_tracker import TrajectoryTracker
from avlite.c60_common.c64_collision_checking import check_collision

if TYPE_CHECKING:
    from avlite.c30_control.c32_control_strategy import ControlStrategy

log = logging.getLogger(__name__)


class VelocityLocalPlanner(LocalPlanningStrategy):
    """Follow the global path geometry and adjust velocity for safe obstacle stopping."""

    def __init__(
        self,
        global_plan: GlobalPlan,
        env: PerceptionModel,
        controller: Optional["ControlStrategy"] = None,
        setting: PlanningSettingsSchema = PlanningSettings,
    ):
        super().__init__(global_plan=global_plan, pm=env, controller=controller, setting=setting)
        self._local_trajectory: Optional[TrajectoryTracker] = None
        self._stopping_decel_factor = setting.c26_stopping_decel_factor
        self._fallback_deceleration = setting.c26_fallback_deceleration
        self._stopping_safety_buffer = setting.c26_stopping_safety_buffer

    def set_global_plan(self, global_plan: GlobalPlan, ego_xy=None) -> None:
        super().set_global_plan(global_plan, ego_xy=ego_xy)
        self._local_trajectory = None

    def reset(self, wp: int = 0):
        super().reset(wp)
        self._local_trajectory = None

    def get_local_plan(self) -> LocalPlan:
        if self._local_trajectory is not None:
            return LocalPlan.from_trajectory(self._local_trajectory)
        return super().get_local_plan()

    def _advance_local_plan(self, state) -> None:
        if self._local_trajectory is not None:
            self._local_trajectory.update_waypoint_by_xy(state.x, state.y)

    def replan(self):
        tj = self.global_trajectory
        local_tj = TrajectoryTracker(path=list(tj.path), velocity=list(tj.velocity), name="Local Trajectory")
        local_tj.current_wp = tj.current_wp
        local_tj.next_wp = tj.next_wp

        collision, collision_idx, _ = check_collision(
            self.pm,
            local_tj,
            min_velocity_threshold=PlanningSettings.c20_min_velocity_threshold,
            collision_safety_margin=PlanningSettings.c20_collision_safety_margin,
            default_ego_velocity=PlanningSettings.c20_default_ego_velocity,
        )

        if collision:
            self._apply_stopping_profile(local_tj, collision_idx)

        self._local_trajectory = local_tj

    def _max_deceleration(self) -> float:
        if self.controller is not None:
            max_decel = abs(self.controller.ego_min_acceleration) * self._stopping_decel_factor
        else:
            max_decel = self._fallback_deceleration
        return max_decel if max_decel >= 0.1 else self._fallback_deceleration

    @staticmethod
    def _distance_to_index(path_x, path_y, end_idx: int) -> float:
        dist = 0.0
        for i in range(1, min(end_idx + 1, len(path_x))):
            dist += math.hypot(path_x[i] - path_x[i - 1], path_y[i] - path_y[i - 1])
        return dist

    def _apply_stopping_profile(self, tj: TrajectoryTracker, collision_idx: int) -> None:
        current_vel = self.pm.ego_vehicle.velocity if self.pm.ego_vehicle.velocity > 0 else (
            float(tj.velocity[tj.current_wp]) if len(tj.velocity) > tj.current_wp else 0.0
        )
        max_decel = self._max_deceleration()
        stopping_distance = current_vel ** 2 / (2 * max_decel)
        collision_distance = self._distance_to_index(tj.path_x, tj.path_y, collision_idx)

        if stopping_distance >= collision_distance - self._stopping_safety_buffer:
            tj.velocity = np.maximum(0.0, np.linspace(current_vel, 0.0, len(tj.path)))
            log.warning("Obstacle ahead — emergency stop profile applied")
            return

        cumulative_dist = 0.0
        brake_start_idx = 0
        target_brake_dist = collision_distance - stopping_distance - self._stopping_safety_buffer

        for i in range(1, len(tj.path_x)):
            cumulative_dist += math.hypot(tj.path_x[i] - tj.path_x[i - 1], tj.path_y[i] - tj.path_y[i - 1])
            if cumulative_dist >= target_brake_dist:
                brake_start_idx = i
                break

        new_velocity = np.empty(len(tj.path))
        for i in range(len(tj.path)):
            if i <= brake_start_idx:
                new_velocity[i] = current_vel
            else:
                progress = (i - brake_start_idx) / max(1, len(tj.path) - brake_start_idx - 1)
                new_velocity[i] = max(0.0, current_vel * (1.0 - progress))

        tj.velocity = new_velocity
        log.info(
            "Stop profile: hold %.1f m/s until idx %d, then ramp to 0 before collision at idx %d",
            current_vel,
            brake_start_idx,
            collision_idx,
        )
