from c10_perception.c12_state import AgentState
from c10_perception.c12_state import EgoState
from c30_control.c31_base_controller import ControlComand
from c40_execution.c41_base_executer import WorldInterface
import math
import copy

import logging
log = logging.getLogger(__name__)

class BasicSim(WorldInterface):
    def __init__(self,ego_state:EgoState):
        self.ego_state = ego_state
    

    def update_ego_state(self, cmd:ControlComand, dt=0.01):
        acceleration = cmd.acceleration
        steering_angle = cmd.steer

        self.ego_state.x += self.ego_state.velocity * math.cos(self.ego_state.theta) * dt
        self.ego_state.y += self.ego_state.velocity * math.sin(self.ego_state.theta) * dt
        self.ego_state.velocity += acceleration * dt
        self.ego_state.theta += self.ego_state.velocity / self.ego_state.L_f * steering_angle * dt
        
    def spawn_agent(self, agent_state:AgentState):
        pass

    def get_ego_state(self):
        return self.ego_state

