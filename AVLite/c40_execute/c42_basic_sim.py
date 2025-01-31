from c10_perceive.c12_state import AgentState
from c10_perceive.c12_state import EgoState
from c30_control.c31_base_controller import ControlComand
from c40_execute.c41_executer import WorldInterface
import math

import logging
log = logging.getLogger(__name__)

class BasicSim(WorldInterface):
    def __init__(self):
        pass
    

    def update_ego_state(self, ego_state:EgoState, cmd:ControlComand, dt=0.01):
        acceleration = cmd.acceleration
        steering_angle = cmd.steer

        ego_state.x += ego_state.velocity * math.cos(ego_state.theta) * dt
        ego_state.y += ego_state.velocity * math.sin(ego_state.theta) * dt
        ego_state.velocity += acceleration * dt
        ego_state.theta += ego_state.velocity / ego_state.L_f * steering_angle * dt
        
    def spawn_agent(self, agent_state:AgentState):
        pass
