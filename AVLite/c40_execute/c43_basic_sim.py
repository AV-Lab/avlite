from c10_perceive.c12_state import AgentState
from c10_perceive.c12_state import EgoState
from c30_control.c31_base_controller import ControlComand
from c40_execute.c41_executer import WorldInterface
import math
import copy

import logging
log = logging.getLogger(__name__)

class BasicSim(WorldInterface):
    def __init__(self,ego_state:EgoState):
        self.ego_state = ego_state
    

    def update_ego_state(self, state:EgoState, cmd:ControlComand, dt=0.01):
        acceleration = cmd.acceleration
        steering_angle = cmd.steer

        state.x += state.velocity * math.cos(state.theta) * dt
        state.y += state.velocity * math.sin(state.theta) * dt
        state.velocity += acceleration * dt
        state.theta += state.velocity / state.L_f * steering_angle * dt

        self.ego_state = state
        
    def spawn_agent(self, agent_state:AgentState):
        pass

    def get_ego_state(self):
        return self.ego_state

    def get_copy(self):
        return copy.deepcopy(self)
