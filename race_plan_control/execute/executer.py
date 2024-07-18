from race_plan_control.plan.planner import Planner
from race_plan_control.control.controller import Controller
from race_plan_control.execute.vehicle_state import VehicleState

from abc import ABC,abstractmethod
import logging 
import numpy as np
import time 
log = logging.getLogger(__name__)

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
        log.info(f"Control Time: {t2-t1},  Plan Update Time: {(t4-t3)}")
    
    def reset(self):
        self.state.reset()
        self.pl.reset()
        self.cn.reset()

        
    @abstractmethod
    def update(self, dt=0.01, acceleration=0, steering_angle=0):
        pass

