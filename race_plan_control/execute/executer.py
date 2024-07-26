from race_plan_control.plan.planner import Planner
from race_plan_control.control.controller import Controller
from race_plan_control.perceive.vehicle_state import VehicleState

from abc import ABC,abstractmethod
import logging 
import numpy as np
import time 
from math import cos, sin

log = logging.getLogger(__name__)

class Executer(ABC):
    def __init__(self, state: VehicleState, pl: Planner, cn: Controller):
        self.state = state
        self.pl = pl
        self.cn = cn
        self._prev_exec_time = None
        self.time_since_last_replan = 0
        self.elapsed_real_time = 0
        self.elapsed_sim_time = 0

    

    def run(self, control_dt=0.01, replan_dt=None):
        if replan_dt is not None:
            self.time_since_last_replan += control_dt
            if self.time_since_last_replan > replan_dt:
                self.time_since_last_replan = 0
                self.pl.replan()
        # update planner location
        self.pl.step(self.state)

        global_cte = self.pl.past_d[-1] # this one is with respect to global trajectory
        local_tj = self.pl.get_local_plan()
        _,cte = local_tj._convert_point_to_frenet(self.state.x, self.state.y)


        t1 = time.time()
        steering_angle = self.cn.control(cte)
        steering_angle = np.clip(steering_angle, -self.state.max_steering, self.state.max_steering)
        t2 = time.time()
        t3 = time.time()
        self.update_state(dt=control_dt, steering_angle=steering_angle)
        t4 = time.time()
        
        
        if self._prev_exec_time is not None:
            log.info(f"Exec Step Time:{(time.time() - self._prev_exec_time):.3f}  | Control Time: {(t2-t1):.4f},  Plan Update Time: {(t4-t3):.4f}")
        else:
            log.info(f"Control Time: {(t2-t1):.4f},  Plan Update Time: {(t4-t3):.4f}")
        
        
        self._prev_exec_time = time.time()
        self.elapsed_real_time += self._prev_exec_time
    
    def reset(self):
        self.state.reset()
        self.pl.reset()
        self.cn.reset()

        
    @abstractmethod
    def update_state(self, dt=0.01, acceleration=0.0, steering_angle=0.0): # perhaps move_base is a better name
        pass


if __name__ == "__main__":
    import race_plan_control.main as main
    main.run()
