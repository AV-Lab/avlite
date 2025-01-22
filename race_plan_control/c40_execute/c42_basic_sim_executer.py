from c40_execute.c41_executer import Executer
from c10_perceive.c11_environment import Environment
from c10_perceive.c12_state import EgoState
from c20_plan.c21_planner import Planner
from c30_control.c31_controller import Controller
import math

class BasicSimExecuter(Executer):
    def __init__(self, env: Environment, ego_state: EgoState, pl: Planner, cn: Controller):
        super().__init__(env, ego_state, pl, cn)
    

    def update_ego_state(self, dt=0.01, acceleration=0, steering_angle=0) -> EgoState:
        self.ego_state.x += self.ego_state.speed * math.cos(self.ego_state.theta) * dt
        self.ego_state.y += self.ego_state.speed * math.sin(self.ego_state.theta) * dt
        self.ego_state.speed += acceleration * dt
        self.ego_state.theta += self.ego_state.speed / self.ego_state.L_f * steering_angle * dt
        return self.ego_state

