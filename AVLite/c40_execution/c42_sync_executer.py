from ast import Global
import logging
import time
from typing import Optional, Union

from c20_planning.c21_planning_model import GlobalPlan
from c10_perception.c11_perception_model import PerceptionModel, EgoState, AgentState
from c10_perception.c12_perception_strategy import PerceptionStrategy
from c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from c20_planning.c23_local_planning_strategy import LocalPlannerStrategy
from c30_control.c32_control_strategy import ControlStrategy
from c40_execution.c41_execution_model import WorldBridge
from c40_execution.c49_settings import ExecutionSettings

from c40_execution.c41_execution_model import Executer

log = logging.getLogger(__name__)


class SyncExecuter(Executer):
    def __init__(
        self,
        perception_model: PerceptionModel,
        perception:PerceptionStrategy,
        global_planner: GlobalPlannerStrategy,
        local_planner: LocalPlannerStrategy,
        controller: ControlStrategy,
        world: WorldBridge,
        replan_dt=ExecutionSettings.replan_dt,
        control_dt=ExecutionSettings.control_dt,
    ):
        """
        Initializes the SyncExecuter with the given perception model, global planner, local planner, control strategy, and world interface.
        """
        super().__init__(perception_model,perception, global_planner, local_planner, controller, world, replan_dt=replan_dt, control_dt=control_dt)

        self.elapsed_real_time = 0
        self.elapsed_sim_time = 0

        self.__prev_exec_time = None
        self.__planner_last_time = 0.0
        self.__controller_last_time = 0.0


    def step(self, control_dt=0.01, replan_dt=0.01, sim_dt=0.01, call_replan=True, call_control=True, call_perceive=True,) -> None:

        pln_time_txt, cn_time_txt, pr_time_txt, sim_time_txt = "", "", "", ""
        t0 = time.time()

        self.ego_state = self.world.get_ego_state()

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
                cmd = self.controller.control(self.ego_state, local_tj, control_dt=sim_dt)
                cn_time_txt = f"C: {(time.time() - t1):.4f} sec,"

                self.world.control_ego_state(cmd, dt=sim_dt)
        self.elapsed_sim_time += control_dt
        
        t2 = time.time()
        if call_perceive:
            if not self.perception:
                log.error("Perception strategy is not set. Skipping perception step.")

            elif self.perception.supports_detection == False and self.world.supports_ground_truth_detection:
                self.pm = self.world.get_ground_truth_perception_model()
                perception_output = self.perception.perceive(perception_model=self.pm)
                log.debug(f"[Executer] Perception output: {perception_output.shape if not isinstance(perception_output, list) else len(perception_output)}")
                log.debug(f"type of perception_output: {type(perception_output)}")
                # log.warning(f"occupancy grid: {self.pm.occupancy_flow}")
                log.debug(f"occupancy grid sizes: {self.pm.grid_bounds}")

            else:
                perception_output = self.perception.perceive( rgb_img=self.world.get_rgb_image() 
                    if self.world.supports_rgb_image else None, depth_img=self.world.get_depth_image() 
                    if self.world.supports_depth_image else None, lidar_data=self.world.get_lidar_data() 
                    if self.world.supports_lidar_data else None)
            pr_time_txt = f" PR: {(time.time() - t2):.4f} sec,"



        delta_t_exec = time.time() - self.__prev_exec_time if self.__prev_exec_time is not None else 0
        self.__prev_exec_time = time.time()
        self.elapsed_real_time += delta_t_exec

        log.debug(f"Real Step time: {delta_t_exec:.4f} sec | {pln_time_txt} {cn_time_txt} {pr_time_txt} {sim_time_txt}")
        log.debug( f"Elapsed Real Time: {self.elapsed_real_time:.3f} sec | Elapsed Sim Time: {self.elapsed_sim_time:.3f} sec")


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
        super().reset()
        self.__prev_exec_time = None
        self.__time_since_last_replan = 0


