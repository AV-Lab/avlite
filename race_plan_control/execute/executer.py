from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plan.planner import Planner

from control.controller import Controller
import logging 

import numpy as np

class Executer:
    def __init__(self, state: VehicleState, pl: Planner, cn: Controller):
        self.state = state
        self.pl = pl
        self.cn = cn
    

    def run(self, dt=0.01):
        # update planner location

        self.pl.update_state(self.state)
        cte = self.pl.past_d[-1]

        steering_angle = self.cn.control(cte)
        self.update(dt=dt, steering_angle=steering_angle)
    
    def reset(self):
        self.pl.reset()
        self.state.x = self.pl.reference_x[0]
        self.state.y = self.pl.reference_y[0]


        
    def update(self, dt=0.01, acceleration=0, steering_angle=0):
        pass


class VehicleState:
    def __init__(self, x=0.0, y=0.0, theta=-np.pi/4, speed=0, max_speed=30, max_acceleration=10, max_deceleration=10, l_f=2.5, width=2.0, length=4.5):
        self.x = x
        self.y = y
        self.theta = theta
        self.speed = speed
        
        
        # TODO this should be read from a config file
        # car parameters
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.max_deceleration = max_deceleration
        self.L_f = l_f # Distance from center of mass to front axle
        self.width = width
        self.length = length
    
    def print_state(self):
        print(f"x: {self.x}, y: {self.y}, theta: {self.theta}, speed: {self.speed}")
    
    def __repr__(self):
        return f"VehicleState(x={self.x}, y={self.y}, speed={self.speed}, theta={self.theta})"