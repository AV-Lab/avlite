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
        self.prev_time = None
    

    def run(self, dt=0.01):
        # update planner location

        self.pl.update_state(self.state)

        global_cte = self.pl.past_d[-1] # this one is with respect to global trajectory
        local_tj = self.pl.get_local_plan()
        _,cte = local_tj.convert_point_to_frenet(self.state.x, self.state.y)


        t1 = time.time()
        steering_angle = self.cn.control(cte)
        t2 = time.time()
        t3 = time.time()
        self.update_state(dt=dt, steering_angle=steering_angle)
        t4 = time.time()
        
        if self.prev_time is not None:
            log.info(f"Exec Step Time:{(time.time() - self.prev_time):.3f}  | Control Time: {(t2-t1):.4g},  Plan Update Time: {(t4-t3):.4g}")
        else:
            log.info(f"Control Time: {t2-t1},  Plan Update Time: {(t4-t3)}")
        self.prev_time = time.time()
    
    def reset(self):
        self.state.reset()
        self.pl.reset()
        self.cn.reset()

        
    @abstractmethod
    def update_state(self, dt=0.01, acceleration=0, steering_angle=0): # perhaps move_base is a better name
        pass

