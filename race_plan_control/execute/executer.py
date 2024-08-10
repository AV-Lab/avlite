from race_plan_control.plan.planner import Planner
from race_plan_control.control.controller import Controller
from race_plan_control.perceive.vehicle_state import VehicleState

from abc import ABC,abstractmethod
import logging 
import numpy as np
import time 

log = logging.getLogger(__name__)

class Executer(ABC):
    def __init__(self, state: VehicleState, pl: Planner, cn: Controller):
        self.ego_state = state
        self.planner = pl
        self.controller = cn
        self.__prev_exec_time = None
        self.__time_since_last_replan = 0
        self.elapsed_real_time = 0
        self.elapsed_sim_time = 0

    

    def run(self, control_dt=0.01, replan_dt=None):
        if replan_dt is not None:
            self.__time_since_last_replan += control_dt
            if self.__time_since_last_replan > replan_dt:
                self.__time_since_last_replan = 0
                self.planner.replan()
        # update planner location
        self.planner.step(self.ego_state)

        global_cte = self.planner.traversed_d[-1] # this one is with respect to global trajectory
        local_tj = self.planner.get_local_plan()
        _,cte = local_tj.convert_xy_to_sd(self.ego_state.x, self.ego_state.y)


        t1 = time.time()
        steering_angle = self.controller.control(cte)
        steering_angle = np.clip(steering_angle, -self.ego_state.max_steering, self.ego_state.max_steering)
        t2 = time.time()
        t3 = time.time()
        self.update_state(dt=control_dt, steering_angle=steering_angle)
        t4 = time.time()
        
        self.elapsed_sim_time += control_dt
        delta_t_exec = time.time() - self.__prev_exec_time if self.__prev_exec_time is not None else 0
        self.__prev_exec_time = time.time()
        self.elapsed_real_time += delta_t_exec 
        
        log.info(f"Exec Step Time:{delta_t_exec:.3f}  | Control Time: {(t2-t1):.4f},  Plan Update Time: {(t4-t3):.4f}")
        log.info(f"Elapsed Real Time: {self.elapsed_real_time:.3f} | Elapsed Sim Time: {self.elapsed_sim_time:.3f}")

    def run_loop(self, control_dt=0.01, replan_dt=None, max_time=100):
        self.reset()
        while self.elapsed_sim_time < max_time:
            self.run(control_dt=control_dt, replan_dt=replan_dt)
        
    
    def reset(self):
        self.ego_state.reset()
        self.planner.reset()
        self.controller.reset()
        self.__prev_exec_time = None
        self.__time_since_last_replan = 0
        self.elapsed_real_time = 0
        self.elapsed_sim_time = 0

        
    @abstractmethod
    def update_state(self, dt=0.01, acceleration=0.0, steering_angle=0.0)-> VehicleState: # perhaps move_base is a better name
        pass


if __name__ == "__main__":
    import race_plan_control.main as main
    main.run()
