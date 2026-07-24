import logging
import time

from avlite.c10_perception.c11_perception_model import PerceptionModel
from avlite.c10_perception.c12_perception_strategy import PerceptionStrategy
from avlite.c10_perception.c13_localization_strategy import LocalizationStrategy
from avlite.c10_perception.c14_mapping_strategy import MappingStrategy
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c40_execution.c41_world_bridge import WorldBridge, is_world_stack_capability_enabled
from avlite.c40_execution.c42_execution_strategy import ExecutionStrategy
from avlite.c40_execution.c43_task_strategy import TaskStrategy
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c50_common.c51_capabilities import StackCapability

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
        # Negative so the first paced gate (elapsed - last >= period) fires immediately.
        self.__planner_last_time = -1e9
        self.__controller_last_time = -1e9
        self.__localization_last_time = -1e9
        self.__perception_last_time = -1e9


    def step(
        self,
        perception_dt=0.01,
        control_dt=0.01,
        replan_dt=0.01,
        localization_dt=0.01,
        sim_dt=0.01,
        call_replan=True,
        call_control=True,
        call_perceive=True,
        call_localize=True,
        pace_perception=True,
        pace_replan=True,
        pace_control=True,
        pace_sim=True,
    ) -> None:
        """ Executes a single step of the simulation, including planning, control, and perception. """

        pln_time_txt, cn_time_txt, pr_time_txt, loc_time_txt, sim_time_txt = "", "", "", "", ""
        t0 = time.time()

        # Pose update: GT world → PM, or localization strategy → PM (mutually exclusive).
        t_loc = time.time()
        if is_world_stack_capability_enabled(StackCapability.LOCALIZATION):
            self.pm.ego_vehicle.copy_from(self.world.get_ego_state())
        elif call_localize and self.localization:
            if self.elapsed_sim_time - self.__localization_last_time >= localization_dt:
                self.__localization_last_time = self.elapsed_sim_time
                self._localization_step()
                loc_time_txt = f" LOC: {(time.time() - t_loc):.4f} sec,"

        # Perceive first so that planning and the visualization both operate on the
        # same perception snapshot. Running perception after replan caused the planner
        # to react to the previous frame's obstacles while the UI rendered the new
        # perception model, making obstacles appear "detected but not visualized".
        t2 = time.time()
        if call_perceive:
            if (not pace_perception) or (
                self.elapsed_sim_time - self.__perception_last_time >= perception_dt
            ):
                self.__perception_last_time = self.elapsed_sim_time
                self._perception_step()
                pr_time_txt = f" PR: {(time.time() - t2):.4f} sec,"

        if call_replan:
            if (not pace_replan) or (
                self.elapsed_sim_time - self.__planner_last_time >= replan_dt
            ):
                self.__planner_last_time = self.elapsed_sim_time
                self._replan_step()
                pln_time_txt = f" P: {(time.time() - t0):.2} sec,"

        if self.local_planner:
            self.local_planner.step(self.pm.ego_vehicle)

        t1 = time.time()
        if call_control:
            if (not pace_control) or (
                self.elapsed_sim_time - self.__controller_last_time >= control_dt
            ):
                self.__controller_last_time = self.elapsed_sim_time
                self._control_step(sim_dt)
                cn_time_txt = f"C: {(time.time() - t1):.4f} sec,"

        # Free-run: advance sim and real by the same wall interval so the UI clocks match.
        # Paced: fixed sim_dt; real time is measured separately over the full step.
        # Stall cap is 1s (debugger pauses) — do not clamp to ~0.05s or UI redraws make sim lag real.
        now = time.time()
        if pace_sim:
            dt = sim_dt
            self._simulate_step(dt)
            self.elapsed_sim_time += dt
            t_end = time.time()
            if self.__prev_exec_time is not None:
                self.elapsed_real_time += t_end - self.__prev_exec_time
            self.__prev_exec_time = t_end
        else:
            if self._last_sim_wall_t is None:
                self._last_sim_wall_t = now
            else:
                dt = max(1e-4, min(now - self._last_sim_wall_t, 1.0))
                self._last_sim_wall_t = now
                self._simulate_step(dt)
                self.elapsed_sim_time += dt
                self.elapsed_real_time += dt

        self.task_runner.step(self)

        log.debug(
            "Step | %s %s %s %s %s | real=%.3f sim=%.3f",
            pln_time_txt, cn_time_txt, loc_time_txt, pr_time_txt, sim_time_txt,
            self.elapsed_real_time, self.elapsed_sim_time,
        )


    def reset(self):
        super().reset()
        self.__prev_exec_time = None
        self.__planner_last_time = -1e9
        self.__controller_last_time = -1e9
        self.__localization_last_time = -1e9
        self.__perception_last_time = -1e9
        self.__time_since_last_replan = 0
