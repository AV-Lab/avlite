import math
from typing import Optional

import numpy as np

from avlite.c10_perception.c11_perception_model import AgentState, Map, PerceptionModel, RaceMap
from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c30_control.c31_control_model import ControlCommand
from avlite.c40_execution.c41_world_bridge import WorldBridge
from avlite.c50_common.c51_capabilities import StackCapability, WorldCapability
from avlite.c40_execution.c49_settings import ExecutionSettings, ExecutionSettingsSchema
from avlite.c30_control.c34_stanley import StanleyController
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c50_common.c52_world_sensor_datatypes import LidarCloud
from avlite.c50_common.c54_trajectory_tracker import TrajectoryTracker


import logging

log = logging.getLogger(__name__)



class BasicSim(WorldBridge):
    world_capabilities = frozenset({
        WorldCapability.LIDAR_2D,
        WorldCapability.AGENT_SPAWN,
    })
    stack_capabilities = frozenset({
        StackCapability.DETECTION,
        StackCapability.TRACKING,
        StackCapability.LOCALIZATION,
    })
    stack_requirements = frozenset({StackCapability.CONTROL})

    def __init__(self, ego_state: EgoState, pm: Optional[PerceptionModel] = None,
                 controller: Optional[ControlStrategy] = None,
                 setting: ExecutionSettingsSchema = ExecutionSettings,
                 reference_point: tuple[float, float] | None = None,
                 map: Map | None = None):
        self.setting = setting
        self.ego_state = ego_state
        self.pm = pm
        self.reference_point = reference_point
        self.map = map
        self.ego_controller = controller
        self.supports_ground_truth_detection = True
        self.supports_ground_truth_localization = True
        self.npc_control = setting.c46_npc_control
        self.speed_factor = setting.c46_npc_speed_factor
        self.npc_controllers = {}

        # Road boundary polylines as raycasting segments, shape (M, 2, 2):
        #   axis 0 — segment index (M = total segments from LeftBound + RightBound)
        #   axis 1 — endpoint: 0 = start, 1 = end
        #   axis 2 — coordinate: 0 = x, 1 = y
        # Used by __collect_segments() for LiDAR raycasting and by c57 for visualization.
        self.boundary_segments: np.ndarray = boundary_segments_from_map(map)


    def control_ego_state(self, cmd:ControlCommand, dt=0.01):
        acceleration = cmd.acceleration
        steering_angle = cmd.steer

        self.ego_state.x += self.ego_state.velocity * math.cos(self.ego_state.theta) * dt
        self.ego_state.y += self.ego_state.velocity * math.sin(self.ego_state.theta) * dt
        self.ego_state.velocity += acceleration * dt
        self.ego_state.theta += self.ego_state.velocity / (self.ego_controller.ego_distance_front_axle if self.ego_controller is not None else 2.5) * steering_angle * dt

        if self.npc_control:
            self.__control_npc_agents(dt)

    def __control_npc_agents(self, dt: float):
        """ Control NPC agents in the simulation. """
        max_dsteer = 3.0 * dt
        for agent in self.pm.agent_vehicles:
            ctrl = self.npc_controllers.get(agent.agent_id)
            if ctrl is None:
                continue

            cmd = ctrl.control(
                agent, control_dt=dt,
                perception_model=PerceptionModel(ego_vehicle=agent),
            )

            prev_steer = getattr(ctrl, "_npc_steer", cmd.steer)
            steer = float(np.clip(cmd.steer, prev_steer - max_dsteer, prev_steer + max_dsteer))
            ctrl._npc_steer = steer

            agent.velocity += cmd.acceleration * dt
            agent.theta += agent.velocity / ctrl.ego_distance_front_axle * steer * dt
            agent.x += agent.velocity * math.cos(agent.theta) * dt
            agent.y += agent.velocity * math.sin(agent.theta) * dt


        
    def spawn_agent(
        self,
        agent_state: AgentState,
        global_plan: Optional[GlobalPlan] = None,
    ):
        id = self.pm.add_agent_vehicle(agent_state)

        ref = global_plan.trajectory if global_plan is not None else None
        if self.npc_control and ref is not None and len(ref.path) > 0:
            tj = TrajectoryTracker(
                path=list(ref.path),
                velocity=[v * self.speed_factor for v in ref.velocity],
            )
            tj.update_waypoint_by_xy(agent_state.x, agent_state.y)
            agent_state.velocity = tj.velocity[tj.current_wp]

            controller = StanleyController(tj=tj)
            controller.reset()
            self.npc_controllers[id] = controller
        elif self.npc_control:
            log.warning("spawn_agent: no global plan available; NPC will not be controlled")

        agent_state.set_start()

    def get_ego_state(self):

        return self.ego_state

    def teleport_ego(self, x: float, y: float, theta: Optional[float] = None):
        self.ego_state.x = x
        self.ego_state.y = y
        if theta is not None:
            self.ego_state.theta = theta


    def get_ground_truth_perception_model(self) -> PerceptionModel:
        return self.pm

    def reset(self):
        """Restore the ego and simulated NPCs to their start poses."""
        self.ego_state.reset()
        for agent in self.pm.agent_vehicles if self.pm is not None else []:
            agent.reset()
            if agent.agent_id in self.npc_controllers:
                self.npc_controllers[agent.agent_id].reset()

    # ------------------------------------------------------------------
    # 2D LiDAR simulation
    # ------------------------------------------------------------------

    def __collect_segments(self) -> np.ndarray:
        """All obstacle segments to raycast: agent bounding boxes + boundaries."""
        segments = [self.boundary_segments]
        if self.pm is not None:
            for agent in self.pm.agent_vehicles:
                corners = agent.get_bb_corners()  # (4, 2) CCW
                rolled = np.roll(corners, -1, axis=0)
                segments.append(np.stack([corners, rolled], axis=1))
        return np.concatenate(segments, axis=0) if segments else np.empty((0, 2, 2))

    def get_lidar_data(self) -> Optional[LidarCloud]:
        """Simulate a 2D LiDAR scan, returning world-frame hits as (N, 4) float32.

        Casts ``num_beams`` rays over ``fov_deg`` (centred on the ego heading)
        against agent bounding boxes and road boundaries, keeping the nearest
        intersection per beam within ``range``.  Beams that hit nothing are
        skipped.  z and intensity columns are zero (2D scanner).
        """
        points_2d = self._simulate_lidar_2d()
        return lidar_2d_to_4(points_2d)

    def _simulate_lidar_2d(self) -> np.ndarray:
        """Return ordered world-frame 2D hits (N, 2)."""
        segments = self.__collect_segments()
        if len(segments) == 0:
            return np.empty((0, 2))

        n = self.setting.c46_lidar_num_beams
        fov = math.radians(self.setting.c46_lidar_fov_deg)
        max_range = self.setting.c46_lidar_range
        origin = np.array([self.ego_state.x, self.ego_state.y])

        if fov >= 2 * math.pi:
            angles = self.ego_state.theta + np.linspace(0, 2 * math.pi, n, endpoint=False)
        else:
            angles = self.ego_state.theta + np.linspace(-fov / 2, fov / 2, n)
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (n, 2)

        # Segment endpoints: p = seg[:,0], q = seg[:,1]; edge e = q - p
        p = segments[:, 0, :]                  # (m, 2)
        e = segments[:, 1, :] - p              # (m, 2)

        # Solve origin + t*d = p + u*e  (per beam/segment pair), via 2x2 system.
        # d cross e in denominator; broadcast beams (n,1) against segments (1,m).
        d = directions[:, None, :]             # (n, 1, 2)
        denom = d[..., 0] * e[None, :, 1] - d[..., 1] * e[None, :, 0]  # (n, m)
        diff = p[None, :, :] - origin          # (1, m, 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (diff[..., 0] * e[None, :, 1] - diff[..., 1] * e[None, :, 0]) / denom
            u = (diff[..., 0] * d[..., 1] - diff[..., 1] * d[..., 0]) / denom

        valid = (np.abs(denom) > 1e-12) & (t > 0) & (t <= max_range) & (u >= 0) & (u <= 1)
        t = np.where(valid, t, np.inf)
        nearest = t.min(axis=1)                # (n,)

        hit = np.isfinite(nearest)
        if not hit.any():
            return np.empty((0, 2))
        ranges = nearest[hit]
        dirs = directions[hit]
        return origin + ranges[:, None] * dirs


def lidar_2d_to_4(points_2d: np.ndarray) -> LidarCloud:
    """Convert (N, 2) world-frame hits to canonical (N, 4) lidar format."""
    n = points_2d.shape[0]
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)
    pts = np.asarray(points_2d, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"expected (N, 2) lidar, got shape {pts.shape}")
    return np.c_[pts, np.zeros((n, 2), dtype=np.float32)]


def boundary_segments_from_map(map: Map | None) -> np.ndarray:
    """Build (M, 2, 2) LiDAR raycast segments from a RaceMap; empty for other maps."""
    if not isinstance(map, RaceMap):
        return np.empty((0, 2, 2))
    segments = []
    for pts in (map.left_bound, map.right_bound):
        pts = np.asarray(pts, dtype=float)
        if pts.ndim != 2 or len(pts) < 2:
            continue
        pts = pts[:, :2]
        segments.append(np.stack([pts[:-1], pts[1:]], axis=1))
    if not segments:
        return np.empty((0, 2, 2))
    return np.concatenate(segments, axis=0)


def boundary_segments_from_global_plan(plan) -> np.ndarray:
    """Build (M, 2, 2) line segments from global plan left/right boundary polylines."""
    segments = []
    for xs, ys in (
        (getattr(plan, "left_boundary_x", None), getattr(plan, "left_boundary_y", None)),
        (getattr(plan, "right_boundary_x", None), getattr(plan, "right_boundary_y", None)),
    ):
        if not xs or not ys or len(xs) != len(ys):
            continue
        pts = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])
        if len(pts) >= 2:
            segments.append(np.stack([pts[:-1], pts[1:]], axis=1))
    if not segments:
        return np.empty((0, 2, 2))
    return np.concatenate(segments, axis=0)

