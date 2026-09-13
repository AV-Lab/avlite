"""Pure Pursuit path tracking and Follow the Gap (c35).

Two registered controllers share bicycle-model steering math via
``PurePursuitBase``:

* ``PurePursuitController`` — lookahead on a global/local path
* ``FollowTheGapController`` — LiDAR Follow-the-Gap target + Pure Pursuit steer
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
from avlite.c20_planning.c21_planning_model import GlobalPlan, LocalPlan
from avlite.c30_control.c31_control_model import ControlCommand
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c30_control.c39_settings import ControlSettings, ControlSettingsSchema
from avlite.c50_common.c51_capabilities import AnyOf, StackCapability, WorldCapability
from avlite.c50_common.c52_world_sensor_datatypes import SensorFrame
from avlite.c50_common.c54_trajectory_tracker import TrajectoryTracker

log = logging.getLogger(__name__)


class PurePursuitBase(ControlStrategy, abstract=True):
    """Shared Pure Pursuit steering and velocity PID for path and Follow-the-Gap."""

    def __init__(
        self,
        tj: Optional[TrajectoryTracker] = None,
        setting: ControlSettingsSchema = ControlSettings,
    ):
        super().__init__(tj)
        self._setting = setting
        self.lookahead_distance = setting.c35_lookahead_distance
        self.min_lookahead = setting.c35_min_lookahead
        self.max_lookahead = setting.c35_max_lookahead
        self.lookahead_speed_gain = setting.c35_lookahead_speed_gain
        self.valpha = setting.c35_valpha
        self.vbeta = setting.c35_vbeta
        self.vgamma = setting.c35_vgamma
        self.cte_v_sum = 0.0
        self.cte_velocity = 0.0

    def effective_lookahead(self, ego_velocity: float) -> float:
        """Return Ld: fixed distance, or speed-adaptive when gain > 0."""
        if self.lookahead_speed_gain > 0.0:
            return float(
                np.clip(
                    self.lookahead_speed_gain * max(ego_velocity, 0.0),
                    self.min_lookahead,
                    self.max_lookahead,
                )
            )
        return float(self.lookahead_distance)

    def steer_to_ego_target(self, target_ego_xy: tuple[float, float], ld: float) -> float:
        """Bicycle-model Pure Pursuit steer toward an ego-frame target (x forward, y left).

        delta = arctan(2 * L * sin(alpha) / Ld), with alpha = atan2(y, x).
        """
        tx, ty = target_ego_xy
        alpha = float(np.arctan2(ty, tx))
        # Guard against a degenerate Ld (should not happen with sane settings).
        denom = max(ld, 1e-3)
        steer = float(np.arctan2(2.0 * self.ego_distance_front_axle * np.sin(alpha), denom))
        return float(np.clip(steer, self.ego_min_steering, self.ego_max_steering))

    def velocity_pid(self, ego: EgoState, target_velocity: float) -> float:
        """PID acceleration toward target_velocity, with emergency brake and anti-windup."""
        prev_cte_v = self.cte_velocity
        self.cte_velocity = ego.velocity - target_velocity
        self.cte_v_sum += self.cte_velocity

        vP = -self.valpha * self.cte_velocity
        vI = -self.vbeta * self.cte_v_sum
        vD = -self.vgamma * (self.cte_velocity - prev_cte_v)
        acc = vP + vI + vD

        if (
            target_velocity < ControlSettings.c30_emergency_velocity_threshold
            and ego.velocity > ControlSettings.c30_emergency_min_moving_velocity
        ):
            emergency_acc = self.ego_min_acceleration * ControlSettings.c30_emergency_braking_factor
            if acc > emergency_acc:
                log.warning(
                    "Emergency braking: overriding PID acc %.2f with %.2f",
                    acc,
                    emergency_acc,
                )
                acc = emergency_acc

        acc = float(np.clip(acc, self.ego_min_acceleration, self.ego_max_acceleration))

        if ego.velocity <= 0 and self.cte_v_sum > 0:
            self.cte_v_sum = 0.0
        if ego.velocity <= 0 and acc < 0:
            acc = 0.0

        log.debug(
            "Acc  : %+6.2f [P=%+.3f, I=%+.3f, D=%+.3f] based on CTE: %+.2f "
            "(%.2f vs target: %.2f)",
            acc,
            vP,
            vI,
            vD,
            self.cte_velocity,
            ego.velocity,
            target_velocity,
        )
        return acc

    def find_path_lookahead(self, ego: EgoState, ld: float) -> tuple[float, float] | None:
        """Return ego-frame (x, y) of the path point at arc-length ego_s + Ld."""
        if self.tj is None or not self.tj.is_initialized:
            return None
        self.tj.update_waypoint_by_xy(ego.x, ego.y)
        s, cte = self.tj.convert_xy_to_sd(ego.x, ego.y)
        # Aim ahead along the path; clamp to the path's maximum arc-length.
        # Use max(path_s), not path_s[-1]: a corrupted/non-monotonic path_s (historically
        # path_s[-1]==0 on closed tracks) would otherwise pin every lookahead to s=0.
        s_end = float(np.max(self.tj.path_s)) if len(self.tj.path_s) else 0.0
        s_target = min(s + ld, s_end)
        gx, gy = self.tj.convert_sd_to_xy(s_target, 0.0)

        # World → ego: x forward, y left.
        dx = gx - ego.x
        dy = gy - ego.y
        c, s_th = np.cos(ego.theta), np.sin(ego.theta)
        ex = c * dx + s_th * dy
        ey = -s_th * dx + c * dy
        self.cte_steer = float(cte)
        return float(ex), float(ey)

    def reset(self):
        self.cte_v_sum = 0.0
        self.cte_velocity = 0.0


class PurePursuitController(PurePursuitBase):
    """Geometric Pure Pursuit on a global or local reference path."""

    def control(
        self,
        ego: EgoState,
        plan: GlobalPlan | LocalPlan | None = None,
        control_dt: float | None = None,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
    ) -> ControlCommand:
        if plan is not None:
            self.tj = plan.as_trajectory()
        elif self.tj is None:
            log.warning(
                "Trajectory is not provided. Steering and acceleration set to zero. "
                "Please provide a trajectory."
            )
            return ControlCommand(steer=0, acceleration=0)

        ld = self.effective_lookahead(ego.velocity)

        ##################################
        # Lookahead point on the path
        ##################################
        target_ego = self.find_path_lookahead(ego, ld)
        if target_ego is None:
            log.warning("Could not find path lookahead. Commanding zero.")
            return ControlCommand(steer=0, acceleration=0)

        ##################################
        # Steering: Pure Pursuit
        ##################################
        steer = self.steer_to_ego_target(target_ego, ld)
        log.debug("PurePursuit steer=%+.3f Ld=%.2f target_ego=(%.2f, %.2f)", steer, ld, *target_ego)

        ##################################
        # Velocity PID from trajectory
        ##################################
        if not self.tj.velocity:
            log.warning("Trajectory has no velocity profile. Acceleration set to zero.")
            cmd = ControlCommand(steer=steer, acceleration=0)
            self.cmd = cmd
            return cmd
        idx = min(max(self.tj.current_wp, 0), len(self.tj.velocity) - 1)
        target_velocity = self.tj.velocity[idx]
        acc = self.velocity_pid(ego, target_velocity)

        cmd = ControlCommand(steer=steer, acceleration=acc)
        self.cmd = cmd
        return cmd


class FollowTheGapController(PurePursuitBase):
    """Follow the Gap: aim Pure Pursuit at a forward LiDAR free gap (path-biased)."""

    def __init__(
        self,
        tj: Optional[TrajectoryTracker] = None,
        setting: ControlSettingsSchema = ControlSettings,
    ):
        super().__init__(tj, setting)
        self.cruise_velocity = setting.c35_cruise_velocity
        self.lidar_z_min = setting.c35_lidar_z_min
        self.lidar_z_max = setting.c35_lidar_z_max
        self.bubble_radius = setting.c35_bubble_radius
        self.min_gap_width = setting.c35_min_gap_width

    world_requirements = frozenset({AnyOf(WorldCapability.LIDAR_2D, WorldCapability.LIDAR_3D)})
    # LiDAR steering does not need a plan; localization provides ego pose.
    stack_requirements = frozenset({StackCapability.LOCALIZATION})

    def control(
        self,
        ego: EgoState,
        plan: GlobalPlan | LocalPlan | None = None,
        control_dt: float | None = None,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
    ) -> ControlCommand:
        if plan is not None:
            self.tj = plan.as_trajectory()

        ld = self.effective_lookahead(ego.velocity)

        ##################################
        # Lookahead: path-biased free gap
        ##################################
        lidar_data = sensors.lidar if sensors is not None else None
        ego_pts = self.to_ego_frame(lidar_data, ego)
        if ego_pts is None or len(ego_pts) == 0:
            log.warning("No LiDAR points available. Commanding zero.")
            return ControlCommand(steer=0, acceleration=0)

        preferred = None
        path_pt = self.find_path_lookahead(ego, ld)
        if path_pt is not None:
            preferred = float(np.arctan2(path_pt[1], path_pt[0]))

        target_ego = self.largest_gap_target(ego_pts, ld, preferred_bearing=preferred)
        if target_ego is None:
            log.warning("Could not derive Follow-the-Gap lookahead. Commanding zero.")
            return ControlCommand(steer=0, acceleration=0)

        ##################################
        # Steering: Pure Pursuit
        ##################################
        steer = self.steer_to_ego_target(target_ego, ld)
        log.debug(
            "FollowTheGap steer=%+.3f Ld=%.2f target_ego=(%.2f, %.2f)",
            steer,
            ld,
            *target_ego,
        )

        ##################################
        # Velocity: plan waypoint or cruise
        ##################################
        if self.tj is not None and self.tj.is_initialized and len(self.tj.velocity) > 0:
            self.tj.update_waypoint_by_xy(ego.x, ego.y)
            idx = min(max(self.tj.current_wp, 0), len(self.tj.velocity) - 1)
            target_velocity = self.tj.velocity[idx]
        else:
            target_velocity = self.cruise_velocity

        acc = self.velocity_pid(ego, target_velocity)
        cmd = ControlCommand(steer=steer, acceleration=acc)
        self.cmd = cmd
        return cmd

    def to_ego_frame(self, lidar_data, ego: EgoState) -> np.ndarray | None:
        """Squash LiDAR to 2D and transform world-frame hits into the ego frame."""
        if lidar_data is None:
            return None
        pts = np.asarray(lidar_data, dtype=float)
        if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
            return None

        # Optional z-band filter for 3D clouds (N, 3+) or (N, 4).
        if pts.shape[1] >= 3:
            z = pts[:, 2]
            mask = (z >= self.lidar_z_min) & (z <= self.lidar_z_max)
            pts = pts[mask]
            if len(pts) == 0:
                return None

        xy = pts[:, :2]
        dx = xy[:, 0] - ego.x
        dy = xy[:, 1] - ego.y
        c, s_th = np.cos(ego.theta), np.sin(ego.theta)
        ex = c * dx + s_th * dy
        ey = -s_th * dx + c * dy
        return np.column_stack([ex, ey])

    def largest_gap_target(
        self,
        ego_pts: np.ndarray,
        ld: float,
        preferred_bearing: float | None = None,
    ) -> tuple[float, float] | None:
        """Aim at a forward free-gap mid-bearing (path-biased when preferred is set)."""
        # Safety bubble: ignore returns inside the ego footprint.
        r = np.hypot(ego_pts[:, 0], ego_pts[:, 1])
        ego_pts = ego_pts[r >= self.bubble_radius]
        forward = ego_pts[ego_pts[:, 0] > 0]
        if len(forward) == 0:
            return None

        # Bearings in (-pi/2, pi/2) for x>0; sort ascending (right → left).
        bearings = np.arctan2(forward[:, 1], forward[:, 0])
        bearings = np.sort(bearings)

        edge_lo, edge_hi = -0.5 * np.pi, 0.5 * np.pi
        interior: list[tuple[float, float]] = []
        for i in range(len(bearings) - 1):
            gap = float(bearings[i + 1] - bearings[i])
            mid = float(0.5 * (bearings[i] + bearings[i + 1]))
            interior.append((gap, mid))
        edges = [
            (float(bearings[0] - edge_lo), float(0.5 * (bearings[0] + edge_lo))),
            (float(edge_hi - bearings[-1]), float(0.5 * (bearings[-1] + edge_hi))),
        ]
        # Prefer interior gaps (corridor ahead); fall back to ±90° edges only if needed.
        pool = interior if interior else edges

        if preferred_bearing is not None:
            wide = [c for c in pool if c[0] >= self.min_gap_width] or pool
            _width, mid_bearing = min(wide, key=lambda t: abs(t[1] - preferred_bearing))
        else:
            _width, mid_bearing = max(pool, key=lambda t: t[0])

        return float(ld * np.cos(mid_bearing)), float(ld * np.sin(mid_bearing))
