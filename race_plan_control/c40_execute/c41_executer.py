from c10_perceive.c11_environment import Environment
from c20_plan.c21_planner import Planner
from c30_control.c31_controller import Controller
from c10_perceive.c12_state import EgoState

from abc import ABC, abstractmethod

import logging
import numpy as np
import time
from c10_perceive.c12_state import AgentState

log = logging.getLogger(__name__)


class Executer(ABC):
    def __init__(self, env: Environment, ego_state: EgoState, pl: Planner, cn: Controller):
        self.env = env
        self.ego_state = ego_state
        self.planner = pl
        self.controller = cn
        self.elapsed_real_time = 0
        self.elapsed_sim_time = 0

        self.__prev_exec_time = None
        self.__time_since_last_replan = 0

    def step(self, control_dt=0.01, replan_dt=None):
        t0 = time.time()
        if replan_dt is not None:
            self.__time_since_last_replan += control_dt
            if self.__time_since_last_replan > replan_dt:
                self.__time_since_last_replan = 0
                self.planner.replan()

        # update planner location
        self.planner.step(self.ego_state)

        local_tj = self.planner.get_local_plan()
        _, cte = local_tj.convert_xy_to_sd(self.ego_state.x, self.ego_state.y)

        t1 = time.time()
        steering_angle = self.controller.control(cte)
        steering_angle = np.clip(steering_angle, -self.ego_state.max_steering, self.ego_state.max_steering)
        t2 = time.time()
        t3 = time.time()
        self.update_ego_state(dt=control_dt, steering_angle=steering_angle)
        t4 = time.time()

        self.elapsed_sim_time += control_dt
        delta_t_exec = time.time() - self.__prev_exec_time if self.__prev_exec_time is not None else 0
        self.__prev_exec_time = time.time()
        self.elapsed_real_time += delta_t_exec

        log.info(
            f"Exec Step Time: {(t4-t0)*1000:.2f} ms | Plan Tim: {(t1 - t0)*1000:.2} ms, Control Time: {(t2-t1)*1000:.2f} ms,  Sim Step Time: {(t4-t3)*1000:.2f} ms"
        )
        log.info(f"Elapsed Real Time: {self.elapsed_real_time:.3f} | Elapsed Sim Time: {self.elapsed_sim_time:.3f}")

    def run(self, control_dt=0.01, replan_dt=None, max_time=100):
        self.reset()
        while self.elapsed_sim_time < max_time:
            self.step(control_dt=control_dt, replan_dt=replan_dt)

    def reset(self):
        self.env.reset()
        self.ego_state.reset()
        self.planner.reset()
        self.controller.reset()
        self.__prev_exec_time = None
        self.__time_since_last_replan = 0
        self.elapsed_real_time = 0
        self.elapsed_sim_time = 0

    def spawn_agent(self, x=None, y=None, s=None, d=None, theta=None):
        if x is not None and y is not None:
            t = self.ego_state.theta if theta is None else theta
            agent = AgentState(x=x, y=y, theta=t, speed=0)
        elif s is not None and d is not None:
            # Convert (s, d) to (x, y) using some transformation logic
            x, y = self.planner.global_trajectory.convert_sd_to_xy(s, d)
            log.info(f"Spawning agent at (x, y) = ({x}, {y}) from (s, d) = ({s}, {d})")
            log.info(f"Ego State: {self.ego_state}")
            t = self.ego_state.theta if theta is None else theta
            agent = AgentState(x=x, y=y, theta=t, speed=0)
        else:
            raise ValueError("Either (x, y) or (s, d) must be provided")

        self.env.add_agent_vehicle(agent)

    @abstractmethod
    def update_ego_state(self, dt=0.01, acceleration=0, steering_angle=0) -> EgoState:
        pass


if __name__ == "__main__":
    import race_plan_control.main as main

    main.run()
