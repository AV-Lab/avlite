from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC,abstractmethod
if TYPE_CHECKING:
    from plan.planner import Planner
from control.controller import Controller
import logging 
import numpy as np
import time 

class Executer(ABC):
    def __init__(self, state: VehicleState, pl: Planner, cn: Controller):
        self.state = state
        self.pl = pl
        self.cn = cn
    

    def run(self, dt=0.01):
        # update planner location
        self.pl.update_state(self.state)
        
        cte = self.pl.past_d[-1]
        t1 = time.time()
        steering_angle = self.cn.control(cte)
        t2 = time.time()
        t3 = time.time()
        self.update(dt=dt, steering_angle=steering_angle)
        t4 = time.time()
        logging.info(f"Control Time: {t2-t1},  Plan Update Time: {(t4-t3)}")
    
    def reset(self):
        self.state.reset()
        self.pl.reset()
        self.cn.reset()

        
    @abstractmethod
    def update(self, dt=0.01, acceleration=0, steering_angle=0):
        pass


class VehicleState:
    def __init__(self, x=0.0, y=0.0, theta=-np.pi/4, speed=0, max_speed=30, max_acceleration=10, max_deceleration=10, l_f=2.5, width=2.0, length=4.5):
        self.x = x
        self.y = y
        self.theta = theta
        self.speed = speed

        self.init_x = x
        self.init_y = y
        self.init_theta = theta
        self.init_speed = speed
        
        # TODO this should be read from a config file
        # car parameters
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.max_deceleration = max_deceleration
        self.L_f = l_f # Distance from center of mass to front axle
        self.width = width
        self.length = length

    def reset(self):
        self.x = self.init_x
        self.y = self.init_y
        self.theta = self.init_theta
        self.speed = self.init_speed
    
    def __repr__(self):
        return f"VehicleState(x={self.x}, y={self.y}, speed={self.speed}, theta={self.theta})"
