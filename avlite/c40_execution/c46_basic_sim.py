import math
from typing import Optional

import numpy as np

from avlite.c10_perception.c11_perception_model import AgentState, PerceptionModel
from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c30_control.c32_control_strategy import ControlComand
from avlite.c40_execution.c41_execution_model import WorldBridge, WorldCapability, ExecutionSettings
from avlite.c30_control.c34_stanley import StanleyController
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c60_common.c61_setting_utils import get_absolute_path


import logging

log = logging.getLogger(__name__)

class BasicSim(WorldBridge):
    @property
    def capabilities(self) -> set[WorldCapability]:
        return {
            WorldCapability.GT_DETECTION,
            WorldCapability.GT_TRACKING,
            WorldCapability.GT_LOCALIZATION,
            WorldCapability.LIDAR_2D,
        }

    def __init__(self,ego_state:EgoState, pm:Optional[PerceptionModel] = None,
                 npc_control=ExecutionSettings.basic_sim_npc_control,
                 speed_factor=ExecutionSettings.basic_sim_npc_speed_factor,
                 default_trajectory=ExecutionSettings.basic_sim_default_trajectory,
                 controller: Optional[ControlStrategy] = None):
        self.ego_state = ego_state
        self.pm = pm
        self.ego_controller = controller
        self.supports_ground_truth_detection = True
        self.supports_ground_truth_localization = True
        self.npc_control = npc_control
        self.speed_factor = speed_factor
        self.npc_controllers = {}

        log.info(f"Loading default trajectory from {default_trajectory}")
        if pm is not None and default_trajectory:
            try:  
                from avlite.c20_planning.c21_planning_model import GlobalPlan
                self.default_global_plan = GlobalPlan.from_file(default_trajectory)
                self.npc_control = True
            except Exception as e:
                log.error(f"Failed to load default trajectory {default_trajectory}: {e}")

        self._boundary_segments = self.__load_boundary_segments(
            ExecutionSettings.basic_sim_lidar_boundary_file
        )


    def control_ego_state(self, cmd:ControlComand, dt=0.01):
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
        for agent in self.pm.agent_vehicles:

            cmd = self.npc_controllers[agent.agent_id].control(agent,control_dt=dt) 

            agent_acceleration = cmd.acceleration
            agent_steering_angle = cmd.steer
            agent.x += agent.velocity * math.cos(agent.theta) * dt
            agent.y += agent.velocity * math.sin(agent.theta) * dt
            agent.velocity += agent_acceleration * dt
            agent.theta += agent.velocity / self.npc_controllers[agent.agent_id].ego_distance_front_axle * agent_steering_angle * dt


        
    def spawn_agent(self, agent_state:AgentState):
        id = self.pm.add_agent_vehicle(agent_state)

        if self.npc_control:
            controllable_agent = EgoState(agent_state.x, agent_state.y, agent_state.theta, agent_state.velocity)

            controller = StanleyController(tj=self.default_global_plan.trajectory)
            controller.tj.velocity = [v* self.speed_factor for v in controller.tj.velocity] 
            controller.set_trajectory(self.default_global_plan.trajectory)

            self.npc_controllers[id] = controller

        

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
        """Clear simulated NPC agents and their controllers."""
        if self.pm is not None:
            self.pm.reset()
        self.npc_controllers = {}

    # ------------------------------------------------------------------
    # 2D LiDAR simulation
    # ------------------------------------------------------------------

    @staticmethod
    def __load_boundary_segments(boundary_file: Optional[str]) -> np.ndarray:
        """Load road boundaries as line segments of shape (M, 2, 2)."""
        if not boundary_file:
            return np.empty((0, 2, 2))
        try:
            import json
            with open(get_absolute_path(boundary_file)) as f:
                data = json.load(f)
            segments = []
            for key in ("LeftBound", "RightBound"):
                pts = np.asarray(data.get(key, []), dtype=float)[:, :2]
                if len(pts) >= 2:
                    segments.append(np.stack([pts[:-1], pts[1:]], axis=1))
            if not segments:
                return np.empty((0, 2, 2))
            return np.concatenate(segments, axis=0)
        except Exception as e:
            log.error(f"Failed to load lidar boundary file {boundary_file}: {e}")
            return np.empty((0, 2, 2))

    def __collect_segments(self) -> np.ndarray:
        """All obstacle segments to raycast: agent bounding boxes + boundaries."""
        segments = [self._boundary_segments]
        if self.pm is not None:
            for agent in self.pm.agent_vehicles:
                corners = agent.get_bb_corners()  # (4, 2) CCW
                rolled = np.roll(corners, -1, axis=0)
                segments.append(np.stack([corners, rolled], axis=1))
        return np.concatenate(segments, axis=0) if segments else np.empty((0, 2, 2))

    def get_lidar_data(self) -> Optional[np.ndarray]:
        """Simulate a 2D LiDAR scan, returning ordered world-frame hits (N, 2).

        Casts ``num_beams`` rays over ``fov_deg`` (centred on the ego heading)
        against agent bounding boxes and road boundaries, keeping the nearest
        intersection per beam within ``range``.  Beams that hit nothing are
        skipped, producing the angular gaps that segmentation relies on.
        """
        segments = self.__collect_segments()
        if len(segments) == 0:
            return np.empty((0, 2))

        n = ExecutionSettings.basic_sim_lidar_num_beams
        fov = math.radians(ExecutionSettings.basic_sim_lidar_fov_deg)
        max_range = ExecutionSettings.basic_sim_lidar_range
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

