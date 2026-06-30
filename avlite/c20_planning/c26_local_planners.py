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
from avlite.c60_common.c64_collision_checking import check_collision, precompute_obstacle_polygons

if TYPE_CHECKING:
    from avlite.c30_control.c32_control_strategy import ControlStrategy

log = logging.getLogger(__name__)

# Ego is treated as not closing on the lead above this margin (m/s).
_DECEL_EPS = 0.3


class VelocityLocalPlanner(LocalPlanningStrategy):
    """Follow global path geometry; adjust velocity via speed-match when obstacles block the path."""

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
        self._follow_gap_buffer = setting.c26_follow_gap_buffer
        self._follow_cruise_min_gap = setting.c26_follow_cruise_min_gap

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
        local_tj = self._clone_global_trajectory()
        ref_velocity = np.asarray(self.global_trajectory.velocity, dtype=float)
        self.profile_trajectory(local_tj, ref_velocity=ref_velocity)
        self._local_trajectory = local_tj

    def _large_gap_cruise_threshold(
        self, trajectory: TrajectoryTracker, collision_idx: int, target_vel: float
    ) -> tuple[float, float]:
        current_vel = self._current_ego_speed(trajectory)
        max_decel = self._max_deceleration()
        stopping_distance = max(0.0, current_vel ** 2 - target_vel ** 2) / (2 * max_decel)
        effective_distance = self._effective_stop_distance(trajectory, collision_idx)
        cruise_threshold = stopping_distance + max(self._follow_cruise_min_gap, 2 * stopping_distance)
        return effective_distance, cruise_threshold

    def apply_speed_match(
        self,
        trajectory: TrajectoryTracker,
        collision_idx: int,
        target_vel: float,
        ref_velocity: np.ndarray | None = None,
    ) -> None:
        """Apply speed-match profile and finalize in place on trajectory."""
        if target_vel > 0:
            effective_distance, cruise_threshold = self._large_gap_cruise_threshold(
                trajectory, collision_idx, target_vel
            )
            if effective_distance >= cruise_threshold:
                log.info(
                    "Large gap (%.1fm) — keeping global reference speed (lead %.1f m/s)",
                    effective_distance,
                    target_vel,
                )
                return
        self._apply_speed_match_profile(trajectory, collision_idx, target_vel)
        self._finalize_velocity(trajectory, ref_velocity, collision_idx, target_vel)

    def profile_trajectory(
        self,
        trajectory: TrajectoryTracker,
        *,
        collision_idx: int | None = None,
        agent_vel: float | None = None,
        ref_velocity: np.ndarray | None = None,
    ) -> None:
        """Profile any trajectory. Uses provided collision info, or runs collision check."""
        if ref_velocity is None:
            ref_velocity = np.asarray(trajectory.velocity, dtype=float)
        if collision_idx is None:
            hit, collision_idx, agent_vel = self._check_path_collision(trajectory)
            if not hit:
                return
        target_vel = max(0.0, agent_vel or 0.0)
        self.apply_speed_match(trajectory, collision_idx, target_vel, ref_velocity=ref_velocity)

    def _clone_global_trajectory(self) -> TrajectoryTracker:
        global_tj = self.global_trajectory
        local_tj = TrajectoryTracker(
            path=list(global_tj.path),
            velocity=list(global_tj.velocity),
            name="Local Trajectory",
        )
        local_tj.current_wp = global_tj.current_wp
        local_tj.next_wp = global_tj.next_wp
        return local_tj

    def _check_path_collision(self, local_tj: TrajectoryTracker) -> tuple[bool, int, float]:
        obstacle_polygons = None
        if len(self.pm.agent_vehicles) > 0:
            total_time = self._estimate_traversal_time(local_tj)
            obstacle_polygons = precompute_obstacle_polygons(
                self.pm,
                total_time=total_time,
                min_velocity_threshold=PlanningSettings.c20_min_velocity_threshold,
                obstacle_inflation_margin=PlanningSettings.c20_obstacle_inflation_margin,
            )
        return check_collision(
            self.pm,
            local_tj,
            obstacle_polygons=obstacle_polygons,
            min_velocity_threshold=PlanningSettings.c20_min_velocity_threshold,
            collision_safety_margin=PlanningSettings.c20_collision_safety_margin,
            default_ego_velocity=PlanningSettings.c20_default_ego_velocity,
        )

    def _estimate_traversal_time(self, local_tj: TrajectoryTracker) -> float:
        start_wp = local_tj.current_wp
        path_length = self._distance_between_indices(
            local_tj.path_x, local_tj.path_y, start_wp, len(local_tj.path_x) - 1
        )
        mean_vel = max(float(np.mean(local_tj.velocity[start_wp:])), PlanningSettings.c20_default_ego_velocity)
        return path_length / mean_vel

    @staticmethod
    def _distance_between_indices(path_x, path_y, start_idx: int, end_idx: int) -> float:
        total = 0.0
        for i in range(start_idx + 1, end_idx + 1):
            total += float(
                np.sqrt((path_x[i] - path_x[i - 1]) ** 2 + (path_y[i] - path_y[i - 1]) ** 2)
            )
        return total

    def _current_ego_speed(self, trajectory: TrajectoryTracker) -> float:
        if self.pm.ego_vehicle.velocity > 0:
            return float(self.pm.ego_vehicle.velocity)
        if len(trajectory.velocity) > trajectory.current_wp:
            return float(trajectory.velocity[trajectory.current_wp])
        return 0.0

    def _bumper_gap(self) -> float:
        ego_half = self.pm.ego_vehicle.length / 2
        if not self.pm.agent_vehicles:
            return ego_half + ego_half + self._follow_gap_buffer
        lead_half = max(agent.length for agent in self.pm.agent_vehicles) / 2
        return ego_half + lead_half + self._follow_gap_buffer

    def _remaining_path_distance(self, trajectory: TrajectoryTracker, collision_idx: int) -> float:
        ego = self.pm.ego_vehicle
        s_ego, _ = trajectory.convert_xy_to_sd(ego.x, ego.y)
        s_col = float(trajectory.path_s[min(collision_idx, len(trajectory.path_s) - 1)])
        return max(0.0, s_col - s_ego)

    def _effective_stop_distance(self, trajectory: TrajectoryTracker, collision_idx: int) -> float:
        remaining = self._remaining_path_distance(trajectory, collision_idx)
        return max(0.0, remaining - self._bumper_gap() - self._stopping_safety_buffer)

    def _apply_speed_match_profile(
        self,
        trajectory: TrajectoryTracker,
        collision_idx: int,
        target_vel: float,
    ) -> None:
        start_wp = trajectory.current_wp
        current_vel = self._current_ego_speed(trajectory)
        max_decel = self._max_deceleration()
        stopping_distance = max(0.0, current_vel ** 2 - target_vel ** 2) / (2 * max_decel)
        effective_distance = self._effective_stop_distance(trajectory, collision_idx)

        velocity = np.asarray(trajectory.velocity, dtype=float)
        n = len(velocity)
        upcoming = n - start_wp
        if upcoming <= 0:
            return

        if current_vel <= target_vel + _DECEL_EPS:
            end_idx = min(collision_idx + 1, n)
            if current_vel < target_vel - _DECEL_EPS and end_idx > start_wp:
                ramp_n = min(upcoming, 8, end_idx - start_wp)
                for k in range(ramp_n):
                    alpha = (k + 1) / ramp_n
                    velocity[start_wp + k] = current_vel + alpha * (target_vel - current_vel)
                velocity[start_wp + ramp_n:end_idx] = target_vel
            else:
                velocity[start_wp:end_idx] = target_vel
            trajectory.velocity = velocity
            log.info(
                "Following lead at %.1f m/s (collision idx %d, %.1fm gap)",
                target_vel,
                collision_idx,
                effective_distance,
            )
            return

        if stopping_distance >= effective_distance:
            immediate_cap = math.sqrt(target_vel ** 2 + 2 * max_decel * effective_distance)
            start_speed = min(current_vel, immediate_cap)
            velocity[start_wp:] = np.linspace(start_speed, target_vel, upcoming)
            trajectory.velocity = velocity
            log.warning(
                "Insufficient room to speed-match — ramping from %.1f to %.1f m/s over %d waypoints",
                start_speed,
                target_vel,
                upcoming,
            )
            return

        target_brake_dist = effective_distance - stopping_distance
        brake_start_idx = self._brake_start_index_from_s(trajectory, start_wp, target_brake_dist)

        for i in range(start_wp, n):
            if i <= brake_start_idx:
                velocity[i] = current_vel
            else:
                progress = (i - brake_start_idx) / max(1, n - brake_start_idx - 1)
                velocity[i] = max(target_vel, current_vel - progress * (current_vel - target_vel))

        trajectory.velocity = velocity
        log.info(
            "Speed-match profile: hold %.1f m/s until idx %d, then ramp to %.1f m/s (collision idx %d, %.1fm gap)",
            current_vel,
            brake_start_idx,
            target_vel,
            collision_idx,
            effective_distance,
        )

    def _finalize_velocity(
        self,
        trajectory: TrajectoryTracker,
        ref_velocity: np.ndarray | None,
        collision_idx: int,
        target_vel: float,
    ) -> None:
        velocity = np.asarray(trajectory.velocity, dtype=float)
        if ref_velocity is not None:
            n = min(len(velocity), len(ref_velocity))
            velocity[:n] = np.minimum(velocity[:n], ref_velocity[:n])

        collision_idx = min(max(collision_idx, 0), len(velocity) - 1)
        velocity[collision_idx:] = np.minimum(velocity[collision_idx:], target_vel)

        self._apply_kinematic_cap(velocity, trajectory, collision_idx, target_vel)
        trajectory.velocity = velocity

    def _apply_kinematic_cap(
        self,
        velocity: np.ndarray,
        trajectory: TrajectoryTracker,
        collision_idx: int,
        target_vel: float,
    ) -> None:
        start_wp = trajectory.current_wp
        max_decel = self._max_deceleration()
        if max_decel < 0.1:
            return

        capped = velocity[start_wp]
        s_col = float(trajectory.path_s[min(collision_idx, len(trajectory.path_s) - 1)])
        for i in range(start_wp, collision_idx):
            s_i = float(trajectory.path_s[i])
            dist_to_stop = max(0.0, s_col - s_i - self._bumper_gap() - self._stopping_safety_buffer)
            max_speed = math.sqrt(target_vel ** 2 + 2 * max_decel * dist_to_stop)
            capped = min(velocity[i], max_speed, capped)
            velocity[i] = capped

    def _max_deceleration(self) -> float:
        if self.controller is not None:
            max_decel = abs(self.controller.ego_min_acceleration) * self._stopping_decel_factor
        else:
            max_decel = self._fallback_deceleration
        return max_decel if max_decel >= 0.1 else self._fallback_deceleration

    def _brake_start_index_from_s(
        self, trajectory: TrajectoryTracker, start_wp: int, target_dist: float
    ) -> int:
        s_ego, _ = trajectory.convert_xy_to_sd(self.pm.ego_vehicle.x, self.pm.ego_vehicle.y)
        s_brake = s_ego + target_dist
        for i in range(start_wp, len(trajectory.path_s)):
            if float(trajectory.path_s[i]) >= s_brake:
                return i
        return start_wp
