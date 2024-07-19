
from race_plan_control.plan.planner import Planner
from race_plan_control.control.controller import Controller
from race_plan_control.execute.executer import Executer, VehicleState
from math import cos, sin

class SimpleSim(Executer):
    def __init__(self, state: VehicleState, pl: Planner, cn: Controller):
        super().__init__(state, pl, cn)

    def update_state(self, dt=0.01, acceleration=0, steering_angle=0):
        self.state.x += self.state.speed * cos(self.state.theta) * dt
        self.state.y += self.state.speed * sin(self.state.theta) * dt
        self.state.speed += acceleration * dt
        self.state.theta += self.state.speed/self.state.L_f * steering_angle * dt
        super().update_state()
