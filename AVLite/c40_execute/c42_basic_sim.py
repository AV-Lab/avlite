from c10_perceive.c12_state import AgentState
from c10_perceive.c12_state import EgoState
from c40_execute.c41_executer import WorldInterface
import math

import logging
log = logging.getLogger(__name__)

class BasicSim(WorldInterface):
    def __init__(self):
        pass
    

    def update_ego_state(self, ego_state:EgoState, dt=0.01, acceleration=0, steering_angle=0):
        ego_state.x += ego_state.speed * math.cos(ego_state.theta) * dt
        ego_state.y += ego_state.speed * math.sin(ego_state.theta) * dt
        ego_state.speed += acceleration * dt
        ego_state.theta += ego_state.speed / ego_state.L_f * steering_angle * dt
        return ego_state

    
    def spawn_agent(self, agent_state:AgentState):
        pass
