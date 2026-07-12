import logging
import time

from avlite.c10_perception.c11_perception_model import PerceptionModel, EgoState
from avlite.c10_perception.c12_perception_strategy import PerceptionStrategy
from avlite.c10_perception.c13_localization_strategy import LocalizationStrategy
from avlite.c10_perception.c14_mapping_strategy import MappingStrategy
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c40_execution.c41_world_bridge import WorldBridge
from avlite.c40_execution.c42_execution_strategy import ExecutionStrategy
from avlite.c40_execution.c43_task_strategy import TaskStrategy
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c50_common.c51_capabilities import WorldCapability

log = logging.getLogger(__name__)

class SyncExecuter(ExecutionStrategy):
    def __init__(
        self,
        perception_model: PerceptionModel,
        perception: PerceptionStrategy = None,
        global_planner: GlobalPlannerStrategy = None,
        local_planner: LocalPlanningStrategy = None,
        controller: ControlStrategy = None,
        world: WorldBridge = None,
        localization: LocalizationStrategy = None,
        mapping: MappingStrategy = None,
        perception_dt=ExecutionSettings.c40_perception_dt,
        replan_dt=ExecutionSettings.c40_replan_dt,
        control_dt=ExecutionSettings.c40_control_dt,
        localization_dt=ExecutionSettings.c40_localization_dt,
        tasks: list[TaskStrategy] | None = None,
    ):
        """
        Initializes the SyncExecuter with the given perception model, global planner, local planner, control strategy, and world interface.
        """
        super().__init__(perception_model,perception, global_planner, local_planner, controller, world,
                         localization=localization, mapping=mapping, perception_dt=perception_dt, replan_dt=replan_dt,
                         control_dt=control_dt, localization_dt=localization_dt, tasks=tasks)

        self.elapsed_real_time = 0
        self.elapsed_sim_time = 0

        self.__prev_exec_time = None
        self.__planner_last_time = 0.0
        self.__controller_last_time = 0.0
        self.__localization_last_time = 0.0


    def step(self, perception_dt = 0.01,  control_dt=0.01, replan_dt=0.01, localization_dt=0.01, sim_dt=0.01, call_replan=True, call_control=True, call_perceive=True, call_localize=True,) -> None:
        """ Executes a single step of the simulation, including planning, control, and perception. """

        pln_time_txt, cn_time_txt, pr_time_txt, loc_time_txt, sim_time_txt = "", "", "", "", ""
        t0 = time.time()

        self.ego_state = self.world.get_ego_state()

        # Perceive first so that planning and the visualization both operate on the
        # same perception snapshot. Running perception after replan caused the planner
        # to react to the previous frame's obstacles while the UI rendered the new
        # perception model, making obstacles appear "detected but not visualized".
        t2 = time.time()
        if call_perceive:
            self._perception_step()
            pr_time_txt = f" PR: {(time.time() - t2):.4f} sec,"

        if call_replan:
            dt_p = self.elapsed_sim_time - self.__planner_last_time
            if dt_p >= replan_dt:
                self.__planner_last_time = self.elapsed_sim_time
                self._replan_step()
                pln_time_txt = f" P: {(time.time() - t0):.2} sec,"

        if self.local_planner:
            self.local_planner.step(self.ego_state)

        t1 = time.time()
        if call_control:
            dt_c = self.elapsed_sim_time - self.__controller_last_time
            if dt_c >= control_dt:
                self.__controller_last_time = self.elapsed_sim_time
                self._control_step(sim_dt)
                cn_time_txt = f"C: {(time.time() - t1):.4f} sec,"
        self.elapsed_sim_time += control_dt
        
        # ---- Localization step ----
        t_loc = time.time()
        if call_localize and self.localization:
            dt_loc = self.elapsed_sim_time - self.__localization_last_time
            if dt_loc >= localization_dt:
                self.__localization_last_time = self.elapsed_sim_time
                self._localization_step()
                loc_time_txt = f" LOC: {(time.time() - t_loc):.4f} sec,"

        delta_t_exec = time.time() - self.__prev_exec_time if self.__prev_exec_time is not None else 0
        self.__prev_exec_time = time.time()
        self.elapsed_real_time += delta_t_exec

        self._task_runner.step(self)

        log.debug(f"Real Step time: {delta_t_exec:.4f} sec | {pln_time_txt} {cn_time_txt} {loc_time_txt} {pr_time_txt} {sim_time_txt}")
        log.debug( f"Elapsed Real Time: {self.elapsed_real_time:.3f} sec | Elapsed Sim Time: {self.elapsed_sim_time:.3f} sec")


    def reset(self):
        super().reset()
        self.__prev_exec_time = None
        self.__time_since_last_replan = 0

