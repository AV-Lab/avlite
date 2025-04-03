from c10_perceive.c11_base_perception import PerceptionModel
from c20_plan.c24_base_local_planner import BaseLocalPlanner
from c20_plan.c21_base_global_planner import BaseGlobalPlanner
from c30_control.c31_base_controller import BaseController, ControlComand
from c10_perceive.c12_state import EgoState
from  c20_plan.c21_base_global_planner import PlannerTypeEnum

from abc import ABC, abstractmethod

import logging
import time
from c10_perceive.c12_state import AgentState

log = logging.getLogger(__name__)


class WorldInterface(ABC):
    ego_state: EgoState

    @abstractmethod
    def update_ego_state(self, cmd: ControlComand, dt=0.01):
        """
        Update the ego state.

        Parameters
        state (EgoState): A mutable object representing the ego state.
        cmd (ControlCommand): The control command containing acceleration and steering angle.
        dt (float): Time delta for the update. Default is 0.01.
        """
        pass

    def get_ego_state(self) -> EgoState:
        return self.ego_state

    @abstractmethod
    def spawn_agent(self, agent_state: AgentState):
        """
        Spawn an agent vehicled in a (simulated) world. Its optional if the world allows that.

        """
        pass

    def reset(self):
        pass



class BaseExecuter:
    pm: PerceptionModel
    ego_state: EgoState
    global_planner: BaseGlobalPlanner
    local_planner: BaseLocalPlanner
    controller: BaseController
    world: WorldInterface

    planner_fps: int
    control_fps: int

    def __init__(
        self,
        pm: PerceptionModel,
        glob_pl: BaseGlobalPlanner,
        pl: BaseLocalPlanner,
        cn: BaseController,
        world: WorldInterface,
        replan_dt=0.5,
        control_dt=0.01,
    ):
        self.pm = pm
        self.ego_state = pm.ego_vehicle
        self.global_planner = glob_pl
        self.local_planner = pl
        self.controller = cn
        self.world = world
        self.replan_dt = replan_dt
        self.control_dt = control_dt

        self.elapsed_real_time = 0
        self.elapsed_sim_time = 0

        self.__prev_exec_time = None
        self.__planner_last_time = 0.0
        self.__controller_last_time = 0.0

        self.planner_fps = 0.0
        self.control_fps = 0.0

    def step(
        self,
        control_dt=0.01,
        replan_dt=0.01,
        sim_dt=0.01,
        call_replan=True,
        call_control=True,
        call_perceive=True,
        ) -> None:
        log.info("Global Plan Registry %s",  BaseGlobalPlanner.registry.keys())
        pln_time_txt, cn_time_txt, sim_time_txt = "", "", ""
        t0 = time.time()

        if call_replan:
            dt_p = self.elapsed_sim_time - self.__planner_last_time
            if dt_p >= replan_dt:
                self.local_planner.replan()
                self.__planner_last_time = self.elapsed_sim_time
                self.planner_fps = 1.0 / dt_p
                pln_time_txt = f" P: {(time.time() - t0):.2} sec,"
                # log.info(f"DT Planner: {dt_p:.4f} sec")
        self.local_planner.step(self.ego_state)

        t1 = time.time()
        if call_control:
            dt_c = self.elapsed_sim_time - self.__controller_last_time
            if dt_c >= control_dt:
                self.__controller_last_time = self.elapsed_sim_time
                self.control_fps = 1.0 / dt_c
                local_tj = self.local_planner.get_local_plan()
                cmd = self.controller.control(self.ego_state, local_tj)
                cn_time_txt = f"C: {(time.time() - t1):.4f} sec,"

                self.world.update_ego_state(cmd, dt=sim_dt)
        self.ego_state = self.world.get_ego_state()

        self.elapsed_sim_time += control_dt
        delta_t_exec = time.time() - self.__prev_exec_time if self.__prev_exec_time is not None else 0
        self.__prev_exec_time = time.time()
        self.elapsed_real_time += delta_t_exec

        log.debug(f"Real Step time: {delta_t_exec:.4f} sec | {pln_time_txt} {cn_time_txt} {sim_time_txt}")
        log.debug(
            f"Elapsed Real Time: {self.elapsed_real_time:.3f} sec | Elapsed Sim Time: {self.elapsed_sim_time:.3f} sec"
        )

    def run(self, replan_dt=0.5, control_dt=0.01, call_replan=True, call_control=True, call_perceive=False):
        self.reset()
        while True:
            self.step(
                control_dt=control_dt,
                replan_dt=replan_dt,
                call_replan=call_replan,
                call_control=call_control,
                call_perceive=call_perceive,
            )
            time.sleep(control_dt)

    def stop(self):
        pass

    def reset(self):
        self.pm.reset()
        self.ego_state.reset()
        self.local_planner.reset()
        self.controller.reset()
        self.world.reset
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
            x, y = self.local_planner.global_trajectory.convert_sd_to_xy(s, d)
            log.info(f"Spawning agent at (x, y) = ({x}, {y}) from (s, d) = ({s}, {d})")
            log.info(f"Ego State: {self.ego_state}")
            t = self.ego_state.theta if theta is None else theta
            agent = AgentState(x=x, y=y, theta=t, speed=0)
            self.world.spawn_agent(agent)
        else:
            raise ValueError("Either (x, y) or (s, d) must be provided")

        self.pm.add_agent_vehicle(agent)
