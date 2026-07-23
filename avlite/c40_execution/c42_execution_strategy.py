from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
from avlite.c10_perception.c12_perception_strategy import PerceptionStrategy
from avlite.c10_perception.c13_localization_strategy import LocalizationStrategy
from avlite.c10_perception.c14_mapping_strategy import MappingStrategy
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c40_execution.c41_world_bridge import (
    WorldBridge,
    is_world_stack_capability_enabled,
)
from avlite.c40_execution.c43_task_strategy import (
    StackEvent,
    TaskPlacement,
    TaskRunner,
    TaskStrategy,
)
from avlite.c50_common.c51_capabilities import StackCapability, satisfies_requirements
from avlite.c50_common.c56_fps_tracker import FpsTracker

log = logging.getLogger(__name__)


class ExecutionStrategy(ABC):
    registry = {}

    def __init__(
        self,
        perception_model: PerceptionModel,
        perception: Optional[PerceptionStrategy],
        global_planner: Optional[GlobalPlannerStrategy],
        local_planner: Optional[LocalPlanningStrategy],
        controller: Optional[ControlStrategy],
        world: WorldBridge,
        localization: Optional[LocalizationStrategy] = None,
        mapping: Optional[MappingStrategy] = None,
        perception_dt=0.5,
        replan_dt=0.5,
        control_dt=0.01,
        localization_dt=0.1,
        tasks: Optional[list[TaskStrategy]] = None,
    ):
        """
        Initializes the SyncExecuter with the given perception model, global planner, local planner, control strategy, and world interface.
        """
        self.pm: PerceptionModel = perception_model
        self.perception: Optional[PerceptionStrategy] = perception
        self.localization: Optional[LocalizationStrategy] = localization
        self.mapping: Optional[MappingStrategy] = mapping
        self.ego_state: EgoState = perception_model.ego_vehicle
        self.global_planner: Optional[GlobalPlannerStrategy] = global_planner
        self.local_planner: Optional[LocalPlanningStrategy] = local_planner
        self.controller: Optional[ControlStrategy] = controller
        self.world: WorldBridge = world
        self.task_runner = TaskRunner(tasks or [], executer=self)

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

        self.stopped = False

        self._localization_missing_warned = False

        self._validate_stack()

    # --- public API ---

    def dispatch_task(self, task: TaskStrategy, event: StackEvent | None = None,) -> None:
        """Run a due task, honoring :attr:`TaskStrategy.placement` when possible.

        Base implementation always runs ``INLINE``. Non-INLINE placements log a
        warning and fall back to INLINE until a concrete executer overrides this.
        """
        if task.placement is not TaskPlacement.INLINE:
            log.warning(
                "Task %s requested placement %s; running INLINE",
                type(task).__name__,
                task.placement.name,
            )
        task.execute(self, event=event)

    def available_stack_capabilities(self) -> set:
        """StackCapabilities provided by the assembled stack plus world ground truth."""
        caps = {c for c in self.world.stack_capabilities if is_world_stack_capability_enabled(c)}
        for _, module in self._stack_modules():
            if module is not None:
                caps |= module.stack_capabilities
        return caps

    @abstractmethod
    def step(self, perception_dt=0.01, control_dt=0.01, replan_dt=0.01, localization_dt=0.01, sim_dt=0.01, call_replan=True, call_control=True, call_perceive=True, call_localize=True,) -> None:
        """ Steps the executer for one time step. This method should be implemented by the specific executer class. """
        pass

    def stop(self):
        """Request cooperative shutdown. Subclasses may override to tear down threads/resources."""
        if self.stopped:
            return
        self.stopped = True
        self.task_runner.notify(StackEvent.EXECUTION_STOPPED)

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
        if self.local_planner:
            self.local_planner.reset()
        if self.controller:
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
        self.task_runner.reset()
        self.stopped = False
        self.task_runner.notify(StackEvent.EXECUTION_RESET)

    # --- stack helpers ---

    def _stack_modules(self):
        """Yield ``(label, module)`` for each assembled stack strategy (may be None)."""
        yield "perception", self.perception
        yield "localization", self.localization
        yield "mapping", self.mapping
        yield "global planner", self.global_planner
        yield "local planner", self.local_planner
        yield "controller", self.controller

    def _validate_stack(self) -> None:
        """Raise on unmet module stack_requirements; warn on world deps and duplicates."""
        available = self.available_stack_capabilities()
        if not satisfies_requirements(self.world.stack_requirements, available):
            log.warning(
                "world bridge %s stack_requirements not satisfied: required %s "
                "(available: %s).",
                type(self.world).__name__,
                self.world.stack_requirements,
                available,
            )
        for label, module in self._stack_modules():
            if module is None:
                continue
            if not satisfies_requirements(module.stack_requirements, available):
                raise ValueError(
                    f"{label} strategy {module.__class__.__name__} stack_requirements "
                    f"not satisfied: required {module.stack_requirements} "
                    f"(available: {available})."
                )

        providers: dict = {}
        world_caps = {c for c in self.world.stack_capabilities if is_world_stack_capability_enabled(c)}
        for cap in world_caps:
            providers.setdefault(cap, []).append(f"world/{type(self.world).__name__}")
        for label, module in self._stack_modules():
            if module is None:
                continue
            for cap in module.stack_capabilities:
                providers.setdefault(cap, []).append(f"{label}/{module.__class__.__name__}")
        for cap, sources in providers.items():
            if len(sources) > 1:
                log.warning(
                    "StackCapability.%s provided by multiple sources: %s.",
                    cap.name,
                    ", ".join(sources),
                )

    def _can_actuate(self) -> bool:
        """Whether the ego may be actuated this tick.

        Actuation requires an available ego pose source: either a localization
        strategy that provides ``LOCALIZATION`` or ground-truth localization from
        the world. Without it there is no trustworthy ego pose, so the vehicle
        must not move (warns once per state transition).
        """
        if StackCapability.LOCALIZATION in self.available_stack_capabilities():
            self._localization_missing_warned = False
            return True
        if not self._localization_missing_warned:
            log.warning(
                "LOCALIZATION unavailable (no localization strategy or ground-truth "
                "localization provided); halting ego control."
            )
            self._localization_missing_warned = True
        return False

    # --- tick helpers ---

    def _localization_step(self) -> None:
        """Run one localization iteration using the current world capabilities."""
        if not self.localization:
            return

        world_ok = satisfies_requirements(self.localization.world_requirements, self.world.world_capabilities)
        stack_ok = satisfies_requirements(self.localization.stack_requirements, self.available_stack_capabilities())
        if world_ok and stack_ok:
            sensors = self.world.get_sensor_frame()
            self.localization.localize(perception_model=self.pm, sensors=sensors)
            self.localization_fps = self._localization_fps_tracker.tick()
            # Harvest optional stack_event stamp on PerceptionModel (notify once, then clear).
            if self.pm.stack_event is not None:
                event = self.pm.stack_event
                self.pm.stack_event = None
                self.task_runner.notify(event)
        else:
            log.warning(
                f"Localization strategy {self.localization.__class__.__name__} requirements not satisfied "
                f"(world_requirements {self.localization.world_requirements} vs {self.world.world_capabilities}; "
                f"stack_requirements {self.localization.stack_requirements} vs {self.available_stack_capabilities()}). "
                f"Skipping."
            )

    def _perception_step(self) -> None:
        """Run one perception iteration and update fps tracking."""
        if not self.perception:
            log.debug("Perception strategy is not set. Skipping perception step.")
            return

        world_ok = satisfies_requirements(self.perception.world_requirements, self.world.world_capabilities)
        stack_ok = satisfies_requirements(self.perception.stack_requirements, self.available_stack_capabilities())
        if not (world_ok and stack_ok):
            log.debug(
                f"Perception strategy {self.perception.__class__.__name__} requirements not satisfied "
                f"(world_requirements {self.perception.world_requirements} vs {self.world.world_capabilities}; "
                f"stack_requirements {self.perception.stack_requirements} vs {self.available_stack_capabilities()}). "
                f"Skipping perception step."
            )
            return

        if is_world_stack_capability_enabled(StackCapability.DETECTION):
            gt = self.world.get_ground_truth_perception_model()
            # Copy the world's authoritative agents into the executer's own
            # perception model instead of aliasing to it, so that clearing the
            # executer's model (when ground truth is off) never wipes the
            # simulator's spawned agents.
            self.pm.agent_vehicles = list(gt.agent_vehicles)
            self.pm.static_obstacles = list(gt.static_obstacles)
        else:
            self.pm.agent_vehicles = []

        sensors = self.world.get_sensor_frame()
        self.perception.perceive(perception_model=self.pm, sensors=sensors)

        self.perception_fps = self._perception_fps_tracker.tick()
        # Harvest optional stack_event stamp on PerceptionModel (notify once, then clear).
        if self.pm.stack_event is not None:
            event = self.pm.stack_event
            self.pm.stack_event = None
            self.task_runner.notify(event)

    def _replan_step(self) -> None:
        """Run one planning iteration (replan) and update FPS."""
        if not self.local_planner:
            return
        sensors = self.world.get_sensor_frame()
        self.local_planner.replan(perception_model=self.pm, sensors=sensors)
        self.planner_fps = self._planner_fps_tracker.tick()
        # Harvest optional stack_event stamps from plan artifacts (notify once, then clear).
        try:
            local_plan = self.local_planner.get_local_plan()
        except Exception:
            local_plan = None
        if local_plan is not None:
            event = getattr(local_plan, "stack_event", None)
            if event is not None:
                local_plan.stack_event = None
                self.task_runner.notify(event)
        gp = getattr(self.local_planner, "global_plan", None)
        if gp is not None:
            event = getattr(gp, "stack_event", None)
            if event is not None:
                gp.stack_event = None
                self.task_runner.notify(event)

    def _control_step(self, sim_dt: float) -> None:
        """Run one control iteration, apply to world, and update FPS."""
        if not self.controller or not self.local_planner:
            return
        if not self._can_actuate():
            return
        sensors = self.world.get_sensor_frame()
        local_plan = self.local_planner.get_local_plan()
        cmd = self.controller.control(
            self.ego_state, local_plan, control_dt=sim_dt,
            perception_model=self.pm, sensors=sensors,
        )
        self.world.control_ego_state(cmd, dt=sim_dt)
        self.control_fps = self._control_fps_tracker.tick(floor_dt=sim_dt)
        # Harvest optional stack_event stamp on the control command (notify once, then clear).
        if getattr(cmd, "stack_event", None) is not None:
            event = cmd.stack_event
            cmd.stack_event = None
            self.task_runner.notify(event)

    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:
            ExecutionStrategy.registry[cls.__name__] = cls
