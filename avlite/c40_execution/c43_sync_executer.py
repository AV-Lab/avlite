import logging
import time

from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c10_perception.c11_perception_model import PerceptionModel, EgoState, AgentState
from avlite.c10_perception.c12_perception_strategy import PerceptionStrategy
from avlite.c10_perception.c13_localization_strategy import LocalizationStrategy
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c20_planning.c23_local_planning_strategy import LocalPlannerStrategy
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c40_execution.c41_execution_model import Executer
from avlite.c40_execution.c41_execution_model import WorldBridge
from avlite.c60_common.c62_capabilities import WorldCapability, PerceptionCapability

log = logging.getLogger(__name__)

class SyncExecuter(Executer):
    def __init__(
        self,
        perception_model: PerceptionModel,
        perception: PerceptionStrategy = None,
        global_planner: GlobalPlannerStrategy = None,
        local_planner: LocalPlannerStrategy = None,
        controller: ControlStrategy = None,
        world: WorldBridge = None,
        localization: LocalizationStrategy = None,
        perception_dt=ExecutionSettings.perception_dt,
        replan_dt=ExecutionSettings.replan_dt,
        control_dt=ExecutionSettings.control_dt,
        localization_dt=ExecutionSettings.localization_dt,
    ):
        """
        Initializes the SyncExecuter with the given perception model, global planner, local planner, control strategy, and world interface.
        """
        super().__init__(perception_model,perception, global_planner, local_planner, controller, world,
                         localization=localization, perception_dt=perception_dt, replan_dt=replan_dt,
                         control_dt=control_dt, localization_dt=localization_dt)

        self.elapsed_real_time = 0
        self.elapsed_sim_time = 0

        self.__prev_exec_time = None
        self.__perception_last_time = 0.0
        self.__planner_last_time = 0.0
        self.__controller_last_time = 0.0
        self.__localization_last_time = 0.0


    def step(self, perception_dt = 0.01,  control_dt=0.01, replan_dt=0.01, localization_dt=0.01, sim_dt=0.01, call_replan=True, call_control=True, call_perceive=True, call_localize=True,) -> None:
        """ Executes a single step of the simulation, including planning, control, and perception. """

        pln_time_txt, cn_time_txt, pr_time_txt, loc_time_txt, sim_time_txt = "", "", "", "", ""
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
        
        # ---- Localization step ----
        t_loc = time.time()
        if call_localize and self.localization:
            if self.localization.requirements.issubset(self.world.capabilities):
                dt_loc = self.elapsed_sim_time - self.__localization_last_time
                if dt_loc >= localization_dt:
                    self.__localization_last_time = self.elapsed_sim_time
                    self.localization.localize(
                        lidar=self.world.get_lidar_data() if ExecutionSettings.provide_lidar else None,
                        rgb_img=self.world.get_rgb_image() if ExecutionSettings.provide_rgb else None,
                    )
                    self.localization_fps = 1.0 / dt_loc
                    loc_time_txt = f" LOC: {(time.time() - t_loc):.4f} sec,"
            else:
                log.error(f"Localization strategy {self.localization.__class__.__name__} requirements {self.localization.requirements} not satisfied by capabilities: {self.world.capabilities}. Skipping localization step.")

        t2 = time.time()
        if call_perceive:
            if not self.perception:
                log.error("Perception strategy is not set. Skipping perception step.")

            # elif self.perception.supports_detection == False and self.world.supports_ground_truth_detection:
            elif self.perception.requirements.issubset(self.world.capabilities): 
                # log.warning(f"[Executer] Perception step started at {t2:.4f} sec")
                if ExecutionSettings.provide_ground_truth:
                    self.pm = self.world.get_ground_truth_perception_model()
                else:
                    self.pm.agent_vehicles = []
                perception_output = self.perception.perceive(
                    perception_model=self.pm,
                    rgb_img=self.world.get_rgb_image() if ExecutionSettings.provide_rgb else None,
                    depth_img=self.world.get_depth_image(),
                    lidar_data=self.world.get_lidar_data() if ExecutionSettings.provide_lidar else None,
                )

                # log.debug(f"[Executer] Perception output: {perception_output.shape if not isinstance(perception_output, list) else len(perception_output)}")
                log.debug(f"type of perception_output: {type(perception_output)}")
                # log.warning(f"occupancy grid: {self.pm.occupancy_flow}")
                log.debug(f"occupancy grid sizes: {self.pm.grid_bounds}")

            else:
                log.error(f"Perception strategy {self.perception.__class__.__name__} requirements {self.perception.requirements} not satisfied by capabilities: {self.world.capabilities}. Skipping perception step.")

            pr_time_txt = f" PR: {(time.time() - t2):.4f} sec,"



        delta_t_exec = time.time() - self.__prev_exec_time if self.__prev_exec_time is not None else 0
        self.__prev_exec_time = time.time()
        self.elapsed_real_time += delta_t_exec

        log.debug(f"Real Step time: {delta_t_exec:.4f} sec | {pln_time_txt} {cn_time_txt} {loc_time_txt} {pr_time_txt} {sim_time_txt}")
        log.debug( f"Elapsed Real Time: {self.elapsed_real_time:.3f} sec | Elapsed Sim Time: {self.elapsed_sim_time:.3f} sec")


    def reset(self):
        super().reset()
        self.__prev_exec_time = None
        self.__time_since_last_replan = 0


