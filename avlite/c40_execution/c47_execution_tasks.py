"""Built-in concrete :class:`~avlite.c40_execution.c43_task_strategy.TaskStrategy` tasks."""

from __future__ import annotations

import logging
import math
from typing import ClassVar

from avlite.c40_execution.c43_task_strategy import StackEvent, TaskSchedule, TaskStrategy
from avlite.c50_common.c51_capabilities import satisfies_requirements

log = logging.getLogger(__name__)


class GoalArrivalMonitor(TaskStrategy):
    """Detect rising-edge arrival at the global goal and notify listeners."""

    schedule = TaskSchedule.EVERY_CYCLE
    arrive_radius_m: ClassVar[float] = 3.0

    def __init__(self) -> None:
        self._was_arrived = False

    def reset(self) -> None:
        self._was_arrived = False

    def execute(self, executer, event=None) -> None:
        arrived = self._ego_near_goal(executer, self.arrive_radius_m)
        if arrived and not self._was_arrived:
            executer.task_runner.notify(StackEvent.GOAL_ARRIVED)
        self._was_arrived = arrived

    @staticmethod
    def _ego_near_goal(executer, radius_m: float) -> bool:
        lp = executer.local_planner
        if lp is None:
            return False
        goal = getattr(getattr(lp, "global_plan", None), "goal_point", None)
        if goal is None or len(goal) < 2:
            return False
        ego = executer.ego_state
        return math.hypot(float(ego.x) - float(goal[0]), float(ego.y) - float(goal[1])) <= radius_m


class StopExecAtGoalTask(TaskStrategy):
    schedule = TaskSchedule.ON_EVENT
    listen_events = frozenset({StackEvent.GOAL_ARRIVED})

    def execute(self, executer, event=None) -> None:
        executer.stop()


class TelemetryTask(TaskStrategy):
    schedule = TaskSchedule.INTERVAL
    interval_s = 0.5

    def execute(self, executer, event=None) -> None:
        ego = executer.ego_state
        log.info(
            "sim_t=%.2f ego=(%.1f, %.1f)",
            executer.elapsed_sim_time,
            ego.x,
            ego.y,
        )


class MappingTask(TaskStrategy):
    """Mapping tick. Copies the assembled mapper's ``MAP_*`` caps so the stack
    advertises them once (via this task), not via the mapping module.
    """

    schedule = TaskSchedule.EVERY_CYCLE

    def __init__(self, mapping=None) -> None:
        self.stack_capabilities = (
            frozenset(mapping.stack_capabilities) if mapping is not None else frozenset()
        )

    def execute(self, executer, event=None) -> None:
        mapping = executer.mapping
        if not mapping:
            return
        sensors = None
        ego = None
        if mapping.world_requirements:
            sensors = executer.world.get_sensor_frame()
            ego = executer.world.get_ego_state()
        world_ok = satisfies_requirements(mapping.world_requirements, executer.world.world_capabilities)
        stack_ok = satisfies_requirements(mapping.stack_requirements, executer.available_stack_capabilities())
        if world_ok and stack_ok:
            mapping.update(perception_model=executer.pm, sensors=sensors, ego=ego)
        else:
            log.warning(
                f"Mapping strategy {mapping.__class__.__name__} requirements not satisfied "
                f"(world_requirements {mapping.world_requirements} vs {executer.world.world_capabilities}; "
                f"stack_requirements {mapping.stack_requirements} vs {executer.available_stack_capabilities()}). "
                f"Skipping."
            )


