from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing import Optional

from avlite.c10_perception.c11_perception_model import AgentState, EgoState, PerceptionModel
from avlite.c10_perception.c12_perception_strategy import PerceptionStrategy
from avlite.c10_perception.c13_localization_strategy import LocalizationStrategy
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c40_execution.c41_world_bridge import WorldBridge
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c50_common.c51_capabilities import satisfies_requirements
from avlite.c50_common.c55_fps_tracker import FpsTracker
from avlite.c50_common.c52_sensor_data import SensorFrame

log = logging.getLogger(__name__)


class ExecutionStrategy(ABC):
    registry = {}

    def __init__(
        self,
        perception_model: PerceptionModel,
        perception: Optional[PerceptionStrategy],
        global_planner: GlobalPlannerStrategy,
        local_planner: LocalPlanningStrategy,
        controller: ControlStrategy,
        world: WorldBridge,
        localization: Optional[LocalizationStrategy] = None,
        perception_dt=0.5,
        replan_dt=0.5,
        control_dt=0.01,
        localization_dt=0.1,
    ):
        """
        Initializes the SyncExecuter with the given perception model, global planner, local planner, control strategy, and world interface.
        """
        self.pm: PerceptionModel = perception_model
        self.perception: Optional[PerceptionStrategy] = perception
        self.localization: Optional[LocalizationStrategy] = localization
        self.ego_state: EgoState = perception_model.ego_vehicle
        self.global_planner: GlobalPlannerStrategy = global_planner
        self.local_planner: LocalPlanningStrategy = local_planner
        self.controller: ControlStrategy = controller
        self.world: WorldBridge = world

        self.perception_fps: float = 0.0
        self.planner_fps: float = 0.0
        self.control_fps: float = 0.0
        self.localization_fps: float = 0.0

        self.perception_dt: float = perception_dt
        self.replan_dt: float = replan_dt
        self.control_dt: float = control_dt
        self.localization_dt: float = localization_dt

        self.elapsed_real_time = 0
        self.elapsed_sim_time = 0

        self._perception_fps_tracker = FpsTracker()
        self._planner_fps_tracker = FpsTracker()
        self._control_fps_tracker = FpsTracker()
        self._localization_fps_tracker = FpsTracker()

        self._stop_event = threading.Event()

    @abstractmethod
    def step(self, perception_dt=0.01, control_dt=0.01, replan_dt=0.01, localization_dt=0.01, sim_dt=0.01, call_replan=True, call_control=True, call_perceive=True, call_localize=True,) -> None:
        """ Steps the executer for one time step. This method should be implemented by the specific executer class. """
        pass

    def run(self, replan_dt=None, control_dt=None, call_replan=True, call_control=True, call_perceive=False, call_localize=True):
        """Generic run loop that repeatedly calls step() until stop() is called.

        Subclasses may override for specialized scheduling.
        """
        self.reset()
        self._stop_event.clear()
        rdt = replan_dt if replan_dt is not None else self.replan_dt
        cdt = control_dt if control_dt is not None else self.control_dt
        pdt = self.perception_dt
        ldt = self.localization_dt
        sdt = ExecutionSettings.c40_sim_dt
        while not self._stop_event.is_set():
            try:
                self.step(
                    perception_dt=pdt,
                    control_dt=cdt,
                    replan_dt=rdt,
                    localization_dt=ldt,
                    sim_dt=sdt,
                    call_replan=call_replan,
                    call_control=call_control,
                    call_perceive=call_perceive,
                    call_localize=call_localize,
                )
            except Exception as e:
                log.error(f"ExecutionStrategy step error: {e}", exc_info=True)
            if self._stop_event.wait(timeout=cdt):
                break

    def stop(self):
        """Signal run() to exit. Subclasses may override to also tear down threads/resources."""
        self._stop_event.set()

    @property
    def ui_poll_delay(self) -> Optional[float]:
        """Suggested interval (seconds) for the UI to wait between calls to step().

        Return a fixed value when step() is lightweight (background workers handle heavy
        computation). Return None to let the UI derive the delay from sim_dt adaptively.
        The default is None (adaptive), which suits synchronous executers.
        """
        return None

    def reset(self):
        self.pm.reset()
        self.world.reset()
        self.ego_state.reset()
        if self.perception:
            self.perception.reset()
        if self.localization:
            self.localization.reset()
        self.local_planner.reset()
        self.controller.reset()
        self.world.reset()
        self.elapsed_real_time = 0
        self.elapsed_sim_time = 0
        self.perception_fps = 0.0
        self.planner_fps = 0.0
        self.control_fps = 0.0
        self.localization_fps = 0.0
        self._perception_fps_tracker.reset()
        self._planner_fps_tracker.reset()
        self._control_fps_tracker.reset()
        self._localization_fps_tracker.reset()

    def _fetch_sensor_frame(self) -> SensorFrame:
        """Build SensorFrame from world bridge, respecting c41_provide_* flags."""
        frame = self.world.get_sensor_frame()
        if not ExecutionSettings.c41_provide_rgb:
            frame.rgb = None
        if not ExecutionSettings.c41_provide_lidar:
            frame.lidar = None
        if not ExecutionSettings.c41_provide_depth:
            frame.depth = None
        return frame

    def _localization_step(self) -> None:
        """Run one localization iteration using the current world capabilities."""
        if not self.localization:
            return

        if satisfies_requirements(self.localization.requirements, self.world.capabilities):
            sensors = self._fetch_sensor_frame()
            self.localization.localize(sensors=sensors)
            self.localization_fps = self._localization_fps_tracker.tick()
        else:
            log.warning(
                f"Localization strategy {self.localization.__class__.__name__} requirements "
                f"{self.localization.requirements} not satisfied by capabilities: "
                f"{self.world.capabilities}. Skipping."
            )

    def _perception_step(self) -> None:
        """Run one perception iteration and update fps tracking."""
        if not self.perception:
            log.error("Perception strategy is not set. Skipping perception step.")
            return

        if not satisfies_requirements(self.perception.requirements, self.world.capabilities):
            log.error(
                f"Perception strategy {self.perception.__class__.__name__} requirements "
                f"{self.perception.requirements} not satisfied by capabilities: "
                f"{self.world.capabilities}. Skipping perception step."
            )
            return

        if ExecutionSettings.c41_provide_ground_truth:
            gt = self.world.get_ground_truth_perception_model()
            # Copy the world's authoritative agents into the executer's own
            # perception model instead of aliasing to it, so that clearing the
            # executer's model (when ground truth is off) never wipes the
            # simulator's spawned agents.
            self.pm.agent_vehicles = list(gt.agent_vehicles)
            self.pm.static_obstacles = list(gt.static_obstacles)
        else:
            self.pm.agent_vehicles = []

        sensors = self._fetch_sensor_frame()
        self.perception.perceive(perception_model=self.pm, sensors=sensors)

        self.perception_fps = self._perception_fps_tracker.tick()

    def _replan_step(self) -> None:
        """Run one planning iteration (replan) and update FPS."""
        self.local_planner.replan()
        self.planner_fps = self._planner_fps_tracker.tick()
    
    def _control_step(self, sim_dt: float) -> None:
        """Run one control iteration, apply to world, and update FPS."""
        local_tj = self.local_planner.get_local_plan()
        cmd = self.controller.control(self.ego_state, local_tj, control_dt=sim_dt)
        self.world.control_ego_state(cmd, dt=sim_dt)
        self.control_fps = self._control_fps_tracker.tick(floor_dt=sim_dt)

    def replan_global(self) -> None:
        """Recompute the global plan from the current ego pose and hand it to the local planner.

        Keeps the existing goal, moves the start to the ego's current position,
        re-runs the global planner, then pushes the resulting plan to the local
        planner and controller so subsequent ticks follow the new route.
        """
        goal = self.global_planner.goal_point
        if goal is not None:
            self.global_planner.set_start_goal((self.ego_state.x, self.ego_state.y), goal)

        new_plan = self.global_planner.plan()
        if new_plan is None or new_plan.trajectory is None:
            log.error("Global replan failed: planner returned no valid plan.")
            return

        self.apply_global_plan(new_plan)
        log.info(f"Global replan complete; {len(new_plan.left_boundary_d)} boundary pts")

    def apply_global_plan(
        self,
        global_plan: GlobalPlan,
        ego_xy: Optional[tuple[float, float]] = None,
    ) -> None:
        """Push a global plan to the local planner and controller."""
        if global_plan is None or global_plan.trajectory is None:
            log.error("apply_global_plan: plan or trajectory is None")
            return
        if len(global_plan.trajectory.path_s) == 0:
            log.error("apply_global_plan: trajectory is empty")
            return
        ego_xy = ego_xy if ego_xy is not None else (self.ego_state.x, self.ego_state.y)
        self.local_planner.set_global_plan(global_plan, ego_xy=ego_xy)
        self.controller.set_trajectory(global_plan.trajectory)
        self.controller.reset()

    def spawn_agent(self, agent_state: AgentState) -> None:
        """Spawn an agent in the world using the ego's current global plan."""
        self.world.spawn_agent(agent_state, global_plan=self.local_planner.global_plan)


    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:
            ExecutionStrategy.registry[cls.__name__] = cls
