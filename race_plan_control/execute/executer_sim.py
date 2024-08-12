from race_plan_control.plan.planner import Planner
from race_plan_control.control.controller import Controller
from race_plan_control.perceive.vehicle_state import VehicleState
from race_plan_control.execute.executer import Executer
from math import cos, sin


class SimpleSim(Executer):
    def __init__(self, state: VehicleState, pl: Planner, cn: Controller):
        super().__init__(state, pl, cn)

    def update_state(self, dt=0.01, acceleration=0, steering_angle=0):
        super().update_state(dt, acceleration, steering_angle)
        self.ego_state.x += self.ego_state.speed * cos(self.ego_state.theta) * dt
        self.ego_state.y += self.ego_state.speed * sin(self.ego_state.theta) * dt
        self.ego_state.speed += acceleration * dt
        self.ego_state.theta += (
            self.ego_state.speed / self.ego_state.L_f * steering_angle * dt
        )
        return self.ego_state
