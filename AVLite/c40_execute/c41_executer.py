from c10_perceive.c11_perception_model import PerceptionModel
from c20_plan.c21_base_planner import BasePlanner
from c30_control.c31_base_controller import BaseController, ControlComand
from c10_perceive.c12_state import EgoState

from abc import ABC, abstractmethod

import logging
import time
from c10_perceive.c12_state import AgentState

log = logging.getLogger(__name__)


class WorldInterface(ABC):
    ego_state: EgoState
    
    @abstractmethod
    def update_ego_state(self, state: EgoState, cmd: ControlComand, dt=0.01):
        """
        Update the ego state.

        Parameters:
        state (EgoState): A mutable object representing the ego state.
        cmd (ControlCommand): The control command containing acceleration and steering angle.
        dt (float): Time delta for the update. Default is 0.01.
        """
        pass

    @abstractmethod
    def spawn_agent(self, agent_state: AgentState):
        """
        Spawn an agent vehicled in a (simulated) world. Its optional if the world allows that.

        """
        pass
    @abstractmethod
    def get_copy(self):
        pass

class Executer:
    pm: PerceptionModel
    ego_state: EgoState
    planner: BasePlanner
    controller: BaseController
    world: WorldInterface

    planner_fps: int
    control_fps: int 

    def __init__(
        self,
        pm: PerceptionModel,
        pl: BasePlanner,
        cn: BaseController,
        world: WorldInterface,
        replan_dt=0.5,
        control_dt=0.01
    ):
        self.pm = pm
        self.ego_state = pm.ego_vehicle
        self.planner = pl
        self.controller = cn
        self.world = world
        self.replan_dt = replan_dt
        self.control_dt = control_dt

        self.elapsed_real_time = 0
        self.elapsed_sim_time = 0

        self.__prev_exec_time = None
        self.__time_since_last_replan = 0

        self.planner_fps = 0
        self.control_fps = 0
        

    def step(
        self,
        control_dt=0.01,
        replan_dt=0.01,
        call_replan=True,
        call_control=True,
        call_perceive=True,
    ):
        pln_time, cn_time, sim_time = "", "", ""
        t0 = time.time()
        if call_replan:
            self.__time_since_last_replan += control_dt
            if self.__time_since_last_replan > replan_dt:
                self.__time_since_last_replan = 0
                self.planner.replan()
                pln_time = f" P: {(time.time() - t0):.2} sec,"
        self.planner.step(self.ego_state)

        t1 = time.time() 
        if call_control:
            local_tj = self.planner.get_local_plan()
            cmd = self.controller.control(self.ego_state, local_tj)
            cn_time = f"C: {(time.time() - t1):.4f} sec,"
            t2 = time.time()
            self.world.update_ego_state(self.ego_state, cmd, dt=control_dt)
            self.ego_state = self.world.ego_state
            sim_time = f"Sim: {(time.time() - t2):.4f} sec"


        self.elapsed_sim_time += control_dt
        delta_t_exec = time.time() - self.__prev_exec_time if self.__prev_exec_time is not None else 0
        self.__prev_exec_time = time.time()
        self.elapsed_real_time += delta_t_exec

        log.info(f"Real Step time: {delta_t_exec:.4f} sec | {pln_time} {cn_time} {sim_time}")

        # if call_perceive and call_replan and call_control:
        #     log.info(
        #         f"Exec Step Time: {(t4-t0)*1000:.2f} ms | P: {(t1 - t0)*1000:.2} ms, C: {(t2-t1)*1000:.2f} ms,  Sim: {(t4-t3)*1000:.2f} ms"
        #     )
        log.debug(f"Elapsed Real Time: {self.elapsed_real_time:.3f} | Elapsed Sim Time: {self.elapsed_sim_time:.3f}")

    def run(self, replan_dt=0.5, control_dt=0.01, call_replan=True, call_control=True, call_perceive=False):
        self.reset()
        while True:
            self.step(control_dt=control_dt)
            time.sleep(control_dt)

    def stop(self):
        pass

    def reset(self):
        self.pm.reset()
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
            self.world.spawn_agent(agent)
        elif s is not None and d is not None:
            # Convert (s, d) to (x, y) using some transformation logic
            x, y = self.planner.global_trajectory.convert_sd_to_xy(s, d)
            log.info(f"Spawning agent at (x, y) = ({x}, {y}) from (s, d) = ({s}, {d})")
            log.info(f"Ego State: {self.ego_state}")
            t = self.ego_state.theta if theta is None else theta
            agent = AgentState(x=x, y=y, theta=t, speed=0)
            self.world.spawn_agent(agent)
        else:
            raise ValueError("Either (x, y) or (s, d) must be provided")

        self.pm.add_agent_vehicle(agent)

