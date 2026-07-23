"""Behavioral and velocity local-planning stages.

Holds the behavioral-stage planners (currently :class:`CruiseBehavioralPlanner`)
and the velocity-stage :class:`VelocityLocalPlanner`, which doubles as a
standalone local planner and as the velocity stage of ``LocalPlanningPipeline``.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from avlite.c10_perception.c12_perception_strategy import PerceptionModel
from avlite.c20_planning.c21_planning_model import GlobalPlan, LocalBehavior, LocalPlan
from avlite.c20_planning.c23_local_planning_strategy import (
    LocalBehavioralPlanningStrategy,
    LocalPlanningStrategy,
    LocalVelocityPlanningStrategy,
)
from avlite.c20_planning.c29_settings import PlanningSettings, PlanningSettingsSchema
from avlite.c50_common.c51_capabilities import MayUse, StackCapability
from avlite.c50_common.c54_trajectory_tracker import TrajectoryTracker, slice_trajectory_horizon
from avlite.c50_common.c55_collision_checking import check_collision, precompute_obstacle_polygons

log = logging.getLogger(__name__)

# Ego is treated as not closing on the lead above this margin (m/s).
_DECEL_EPS = 0.3


class CruiseBehavioralPlanner(LocalBehavioralPlanningStrategy):
    """Trivial behavioral planner: always cruise along the reference."""

    world_requirements = frozenset()
    stack_requirements = frozenset()
    stack_capabilities = frozenset()

    def plan_behavior(self, plan: LocalPlan) -> LocalPlan:
        plan.behavior = LocalBehavior.CRUISE
        return plan


class VelocityLocalPlanner(LocalPlanningStrategy, LocalVelocityPlanningStrategy):
    """Follow global path geometry; adjust velocity via speed-match when obstacles block the path.

    Dual-role: usable standalone as a :class:`LocalPlanningStrategy`, or as the
    velocity stage of :class:`LocalPlanningPipeline` via :meth:`plan_velocity`.
    """

    def __init__(
        self,
        global_plan: GlobalPlan,
        env: PerceptionModel,
        setting: PlanningSettingsSchema = PlanningSettings,
    ):
        super().__init__(global_plan=global_plan, pm=env, setting=setting)
        self._local_trajectory: Optional[TrajectoryTracker] = None
        self._max_decel = setting.c27_max_deceleration
        self._stopping_safety_buffer = setting.c27_stopping_safety_buffer
        self._follow_gap_buffer = setting.c27_follow_gap_buffer
        self._follow_cruise_min_gap = setting.c27_follow_cruise_min_gap
        self._follow_gap_gain = setting.c27_follow_gap_gain
        self._planning_horizon_points = setting.c27_planning_horizon_points

    world_requirements = frozenset()
    stack_requirements = frozenset({
        StackCapability.GLOBAL_PLAN,
        StackCapability.LOCALIZATION,
        MayUse(StackCapability.DETECTION, StackCapability.PREDICTION),
    })
    stack_capabilities = frozenset({StackCapability.LOCAL_PLAN})

    def set_global_plan(self, global_plan: GlobalPlan, ego_xy=None) -> None:
        super().set_global_plan(global_plan, ego_xy=ego_xy)
        self._local_trajectory = None

    def reset(self, wp: int = 0):
        super().reset(wp)
        self._local_trajectory = None

    def get_local_plan(self) -> LocalPlan:
        if self._local_trajectory is not None:
            return LocalPlan.from_trajectory(self._local_trajectory)
        return LocalPlan.from_trajectory(self._clone_global_trajectory())

    def _advance_local_plan(self, state) -> None:
        if self._local_trajectory is not None:
            self._local_trajectory.update_waypoint_by_xy(state.x, state.y)

    def replan(
        self,
        perception_model=None,
        sensors=None,
    ):
        if perception_model is not None:
            self.pm = perception_model
        local_tj = self._clone_global_trajectory()
        ref_velocity = np.asarray(local_tj.velocity, dtype=float)
        self.profile_trajectory(local_tj, ref_velocity=ref_velocity)
        self._local_trajectory = local_tj

    def plan_velocity(self, plan: LocalPlan) -> LocalPlan:
        """Velocity stage: profile the incoming plan's trajectory in place."""
        tj = plan.as_trajectory()
        if tj is not None:
            ref_velocity = np.asarray(tj.velocity, dtype=float)
            self.profile_trajectory(tj, ref_velocity=ref_velocity)
            plan.velocity = list(tj.velocity)
            plan.trajectory = tj
        return plan

    def apply_speed_match(
        self,
        trajectory: TrajectoryTracker,
        collision_idx: int,
        target_vel: float,
        ref_velocity: np.ndarray | None = None,
    ) -> None:
        """Apply speed-match profile and finalize in place on trajectory."""
        if target_vel > 0:
            # Large-gap cruise check: with ample room to the lead, keep reference speed.
            current_vel = self._current_ego_speed(trajectory)
            stopping_distance = max(0.0, current_vel ** 2 - target_vel ** 2) / (2 * self._max_deceleration())
            effective_distance = self._effective_stop_distance(trajectory, collision_idx)
            cruise_threshold = stopping_distance + max(self._follow_cruise_min_gap, 2 * stopping_distance)
            if effective_distance >= cruise_threshold:
                log.info(
                    "Large gap (%.1fm) — keeping global reference speed (lead %.1f m/s)",
                    effective_distance,
                    target_vel,
                )
                return
        profile_target = self._apply_speed_match_profile(trajectory, collision_idx, target_vel)
        self._finalize_velocity(trajectory, ref_velocity, collision_idx, profile_target)

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
            # Detect the first blocking agent along this trajectory, predicting movers
            # over the trajectory's estimated traversal time.
            obstacle_polygons = None
            if len(self.pm.agent_vehicles) > 0:
                start_wp = trajectory.current_wp
                px, py = trajectory.path_x, trajectory.path_y
                path_length = 0.0
                for i in range(start_wp + 1, len(px)):
                    path_length += float(np.sqrt((px[i] - px[i - 1]) ** 2 + (py[i] - py[i - 1]) ** 2))
                mean_vel = max(float(np.mean(trajectory.velocity[start_wp:])), PlanningSettings.c20_default_ego_velocity)
                obstacle_polygons = precompute_obstacle_polygons(
                    self.pm,
                    total_time=path_length / mean_vel,
                    min_velocity_threshold=PlanningSettings.c20_min_velocity_threshold,
                    obstacle_inflation_margin=PlanningSettings.c20_obstacle_inflation_margin,
                )
            hit, collision_idx, agent_vel, _ = check_collision(
                self.pm,
                trajectory,
                obstacle_polygons=obstacle_polygons,
                min_velocity_threshold=PlanningSettings.c20_min_velocity_threshold,
                collision_safety_margin=PlanningSettings.c20_collision_safety_margin,
                default_ego_velocity=PlanningSettings.c20_default_ego_velocity,
            )
            if not hit:
                return
        target_vel = max(0.0, agent_vel or 0.0)
        self.apply_speed_match(trajectory, collision_idx, target_vel, ref_velocity=ref_velocity)

    def _clone_global_trajectory(self) -> TrajectoryTracker:
        sliced = slice_trajectory_horizon(
            self.global_trajectory,
            max_points=self._planning_horizon_points,
        )
        sliced.name = "Local Trajectory"
        return sliced

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

    def _effective_stop_distance(self, trajectory: TrajectoryTracker, collision_idx: int) -> float:
        return max(0.0, self._standoff_margin(trajectory, collision_idx))

    def _standoff_margin(self, trajectory: TrajectoryTracker, collision_idx: int) -> float:
        """Signed follow-gap margin (m). Negative means the ego is inside the safe gap."""
        ego = self.pm.ego_vehicle
        s_ego, _ = trajectory.convert_xy_to_sd(ego.x, ego.y)
        s_col = float(trajectory.path_s[min(collision_idx, len(trajectory.path_s) - 1)])
        return (s_col - s_ego) - self._bumper_gap() - self._stopping_safety_buffer

    def _apply_speed_match_profile(
        self,
        trajectory: TrajectoryTracker,
        collision_idx: int,
        target_vel: float,
    ) -> float:
        """Write a speed-match profile onto ``trajectory``. Returns the finalize target speed."""
        start_wp = trajectory.current_wp
        current_vel = self._current_ego_speed(trajectory)
        max_decel = self._max_deceleration()
        stopping_distance = max(0.0, current_vel ** 2 - target_vel ** 2) / (2 * max_decel)
        effective_distance = self._effective_stop_distance(trajectory, collision_idx)
        # Same threshold as apply_speed_match large-gap gate (keep formula in one place here for profile).
        cruise_threshold = stopping_distance + max(self._follow_cruise_min_gap, 2 * stopping_distance)

        velocity = np.asarray(trajectory.velocity, dtype=float)
        n = len(velocity)
        upcoming = n - start_wp
        if upcoming <= 0:
            return target_vel

        matched = current_vel <= target_vel + _DECEL_EPS
        tight = stopping_distance >= effective_distance or (
            matched and effective_distance < cruise_threshold
        )

        if matched and not tight:
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
            return target_vel

        if tight:
            margin = self._standoff_margin(trajectory, collision_idx)
            recovery_target = max(0.0, target_vel + self._follow_gap_gain * min(0.0, margin))
            immediate_cap = math.sqrt(target_vel ** 2 + 2 * max_decel * max(0.0, margin))
            start_speed = min(current_vel, immediate_cap)
            col = min(max(collision_idx, start_wp), n - 1)

            # Seed flat, then kinematic cap pulls the path down at max decel.
            velocity[start_wp:col] = start_speed
            velocity[col:] = recovery_target
            if margin <= 0:
                velocity[start_wp] = recovery_target
            elif current_vel > recovery_target + _DECEL_EPS:
                # Envelope at the budget equals current speed; under async replan that never
                # brakes. Commit one path-step of max decel at current_wp.
                next_i = min(start_wp + 1, n - 1)
                brake_ds = max(1.0, float(trajectory.path_s[next_i] - trajectory.path_s[start_wp]))
                velocity[start_wp] = max(
                    recovery_target,
                    math.sqrt(max(0.0, current_vel ** 2 - 2 * max_decel * brake_ds)),
                )
            self._apply_kinematic_cap(velocity, trajectory, col, recovery_target)
            trajectory.velocity = velocity

            if recovery_target < target_vel - _DECEL_EPS:
                log.info(
                    "Inside follow gap (%.1fm deficit) — braking to %.1f m/s to re-open gap",
                    -margin,
                    recovery_target,
                )
            elif stopping_distance >= effective_distance:
                log.warning(
                    "Insufficient room to speed-match — braking from %.1f to %.1f m/s (%.1fm gap)",
                    float(velocity[start_wp]),
                    recovery_target,
                    effective_distance,
                )
            else:
                log.info(
                    "Following lead at %.1f m/s with gap-aware profile (collision idx %d, %.1fm gap)",
                    recovery_target,
                    collision_idx,
                    effective_distance,
                )
            return recovery_target

        # Hold current speed until the latest brake point, then ramp down to target.
        target_brake_dist = effective_distance - stopping_distance
        s_ego, _ = trajectory.convert_xy_to_sd(self.pm.ego_vehicle.x, self.pm.ego_vehicle.y)
        s_brake = s_ego + target_brake_dist
        brake_start_idx = start_wp
        for i in range(start_wp, len(trajectory.path_s)):
            if float(trajectory.path_s[i]) >= s_brake:
                brake_start_idx = i
                break

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
        return target_vel

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
        return self._max_decel if self._max_decel >= 0.1 else 3.0
