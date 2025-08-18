import math
from typing import Optional

from avlite.c10_perception.c11_perception_model import AgentState, PerceptionModel
from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c30_control.c32_control_strategy import ControlComand
from avlite.c40_execution.c41_execution_model import WorldBridge, WorldCapability

import logging
log = logging.getLogger(__name__)

class BasicSim(WorldBridge):
    @property
    def capabilities(self) -> frozenset[WorldCapability]:
        return frozenset({
            WorldCapability.GT_DETECTION,
            WorldCapability.GT_TRACKING,
            WorldCapability.GT_LOCALIZATION,
        })

    def __init__(self,ego_state:EgoState, pm:PerceptionModel = None,):
        self.ego_state = ego_state
        self.pm = pm
        self.supports_ground_truth_detection = True
        self.supports_ground_truth_localization = True
    

    def control_ego_state(self, cmd:ControlComand, dt=0.01):
        acceleration = cmd.acceleration
        steering_angle = cmd.steer

        self.ego_state.x += self.ego_state.velocity * math.cos(self.ego_state.theta) * dt
        self.ego_state.y += self.ego_state.velocity * math.sin(self.ego_state.theta) * dt
        self.ego_state.velocity += acceleration * dt
        self.ego_state.theta += self.ego_state.velocity / self.ego_state.L_f * steering_angle * dt
        
    def spawn_agent(self, agent_state:AgentState):
        self.pm.add_agent_vehicle(agent_state)

    def get_ego_state(self):
        return self.ego_state

    def teleport_ego(self, x: float, y: float, theta: Optional[float] = None):
        self.ego_state.x = x
        self.ego_state.y = y
        if theta is not None:
            self.ego_state.theta = theta


    def get_ground_truth_perception_model(self) -> PerceptionModel:
        return self.pm
    
