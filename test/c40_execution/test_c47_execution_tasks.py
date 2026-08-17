"""GoalArrivalMonitor / StopExecAtGoalTask guards (c47).

These live in a dedicated module so they do not collide with open drafts that
already append to test_c43_task_strategy.py.
"""

from __future__ import annotations

from types import SimpleNamespace

from avlite.c40_execution.c43_task_strategy import StackEvent, TaskRunner
from avlite.c40_execution.c47_execution_tasks import GoalArrivalMonitor, StopExecAtGoalTask


class _Exec:
    def __init__(self, *, x=0.0, y=0.0, local_planner=None):
        self.ego_state = SimpleNamespace(x=x, y=y)
        self.local_planner = local_planner
        self.stopped = False
        self.task_runner = None
        self.elapsed_sim_time = 0.0

    def dispatch_task(self, task, event=None):
        task.execute(self, event=event)

    def stop(self):
        self.stopped = True


def _planner_with_goal(goal):
    return SimpleNamespace(global_plan=SimpleNamespace(goal_point=goal))


def test_goal_monitor_does_not_fire_without_local_planner():
    monitor = GoalArrivalMonitor()
    executer = _Exec(x=0.0, y=0.0, local_planner=None)
    runner = TaskRunner([monitor, StopExecAtGoalTask()], executer=executer)
    executer.task_runner = runner
    runner.step(executer)
    assert executer.stopped is False


def test_goal_monitor_does_not_fire_without_goal_point():
    monitor = GoalArrivalMonitor()
    executer = _Exec(x=0.0, y=0.0, local_planner=_planner_with_goal(None))
    runner = TaskRunner([monitor, StopExecAtGoalTask()], executer=executer)
    executer.task_runner = runner
    runner.step(executer)
    assert executer.stopped is False


def test_goal_monitor_does_not_fire_on_short_goal_point():
    monitor = GoalArrivalMonitor()
    executer = _Exec(x=0.0, y=0.0, local_planner=_planner_with_goal((1.0,)))
    runner = TaskRunner([monitor, StopExecAtGoalTask()], executer=executer)
    executer.task_runner = runner
    runner.step(executer)
    assert executer.stopped is False


def test_goal_monitor_reset_allows_rising_edge_again():
    monitor = GoalArrivalMonitor()
    planner = _planner_with_goal((10.0, 0.0))
    executer = _Exec(x=10.0, y=0.0, local_planner=planner)
    runner = TaskRunner([monitor, StopExecAtGoalTask()], executer=executer)
    executer.task_runner = runner

    runner.step(executer)
    assert executer.stopped is True

    executer.stopped = False
    runner.step(executer)
    assert executer.stopped is False  # still inside radius; edge already consumed

    monitor.reset()
    runner.step(executer)
    assert executer.stopped is True


def test_stop_exec_at_goal_listens_only_for_goal_arrived():
    assert StopExecAtGoalTask.listen_events == frozenset({StackEvent.GOAL_ARRIVED})
