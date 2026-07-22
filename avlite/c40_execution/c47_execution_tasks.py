"""Built-in concrete :class:`~avlite.c40_execution.c43_task_strategy.TaskStrategy` tasks."""

from __future__ import annotations

import logging
import math
from typing import ClassVar

from avlite.c40_execution.c43_task_strategy import StackEvent, TaskSchedule, TaskStrategy

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
            executer.notify(StackEvent.GOAL_ARRIVED)
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


