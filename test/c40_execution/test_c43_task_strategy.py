"""Tests for TaskStrategy scheduling and monitor/response notify flow."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
from avlite.c20_planning.c21_planning_model import LocalPlan
from avlite.c40_execution.c43_task_strategy import (
    StackEvent,
    TaskPlacement,
    TaskRunner,
    TaskSchedule,
    TaskStrategy,
)
from avlite.c40_execution.c44_sync_executer import SyncExecuter
from avlite.c40_execution.c46_basic_sim import BasicSim
from avlite.c40_execution.c47_execution_tasks import (
    GoalArrivalMonitor,
    StopExecAtGoalTask,
    TelemetryTask,
)
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c50_common.c51_capabilities import StackCapability
from avlite.c60_apps.c62_factory import executor_factory


class EveryCycleTask(TaskStrategy):
    schedule = TaskSchedule.EVERY_CYCLE
    calls: list = []

    def execute(self, executer, event=None) -> None:
        EveryCycleTask.calls.append(event)


class IntervalTask(TaskStrategy):
    schedule = TaskSchedule.INTERVAL
    interval_s = 0.5
    calls: list = []

    def execute(self, executer, event=None) -> None:
        IntervalTask.calls.append(float(executer.elapsed_sim_time))


class StartedEventTask(TaskStrategy):
    schedule = TaskSchedule.ON_EVENT
    listen_events = frozenset({StackEvent.EXECUTION_STARTED})
    calls: list = []

    def execute(self, executer, event=None) -> None:
        StartedEventTask.calls.append(event)


class LocalPlanFailedListener(TaskStrategy):
    schedule = TaskSchedule.ON_EVENT
    listen_events = frozenset({StackEvent.LOCAL_PLAN_FAILED})
    calls: list = []

    def execute(self, executer, event=None) -> None:
        LocalPlanFailedListener.calls.append(event)


class NotifyDuringCycleTask(TaskStrategy):
    schedule = TaskSchedule.EVERY_CYCLE
    fired = False

    def execute(self, executer, event=None) -> None:
        if not NotifyDuringCycleTask.fired:
            NotifyDuringCycleTask.fired = True
            executer.task_runner.notify(StackEvent.GOAL_ARRIVED)


class ThreadPlacementTask(TaskStrategy):
    placement = TaskPlacement.THREAD
    schedule = TaskSchedule.EVERY_CYCLE
    ran = False

    def execute(self, executer, event=None) -> None:
        ThreadPlacementTask.ran = True


@pytest.fixture(autouse=True)
def _reset_task_call_state():
    EveryCycleTask.calls = []
    IntervalTask.calls = []
    StartedEventTask.calls = []
    LocalPlanFailedListener.calls = []
    NotifyDuringCycleTask.fired = False
    ThreadPlacementTask.ran = False
    yield


class _FakeExecuter:
    def __init__(self, *, x=0.0, y=0.0, goal=(100.0, 0.0), caps=None):
        self.elapsed_sim_time = 0.0
        self.ego_state = SimpleNamespace(x=x, y=y)
        self.stopped = False
        self.local_planner = SimpleNamespace(
            global_plan=SimpleNamespace(
                goal_point=goal,
                path=[(0.0, 0.0), goal],
                trajectory=SimpleNamespace(path=[(0.0, 0.0), goal]),
                stack_event=None,
            ),
            global_trajectory=SimpleNamespace(
                is_traversed=lambda: False,
                path=[(0.0, 0.0), goal],
            ),
            get_local_plan=lambda: SimpleNamespace(
                as_trajectory=lambda: SimpleNamespace(path=[(0.0, 0.0), goal]),
                stack_event=None,
            ),
            selected_local_plan=None,
        )
        self._caps = set(caps or {StackCapability.LOCALIZATION})
        self.task_runner = None

    def available_stack_capabilities(self):
        return self._caps

    def dispatch_task(self, task, event=None):
        task.execute(self, event=event)

    def stop(self):
        self.stopped = True


def test_every_cycle_runs_once_per_step():
    runner = TaskRunner([EveryCycleTask()])
    executer = _FakeExecuter()
    runner.step(executer)
    runner.step(executer)
    assert len(EveryCycleTask.calls) == 2


def test_interval_fires_only_when_due():
    runner = TaskRunner([IntervalTask()])
    executer = _FakeExecuter()

    runner.step(executer)
    assert IntervalTask.calls == [0.0]

    executer.elapsed_sim_time = 0.4
    runner.step(executer)
    assert IntervalTask.calls == [0.0]

    executer.elapsed_sim_time = 0.5
    runner.step(executer)
    assert IntervalTask.calls == [0.0, 0.5]


def test_goal_monitor_notifies_stop_at_goal_once():
    monitor = GoalArrivalMonitor()
    executer = _FakeExecuter(x=0.0, y=0.0, goal=(10.0, 0.0))
    runner = TaskRunner([monitor, StopExecAtGoalTask()], executer=executer)
    executer.task_runner = runner

    runner.step(executer)
    assert executer.stopped is False

    executer.ego_state.x = 9.0
    runner.step(executer)
    assert executer.stopped is True

    executer.stopped = False
    runner.step(executer)
    assert executer.stopped is False  # edge already consumed


def test_notify_during_step_flushes_to_on_event():
    executer = _FakeExecuter()
    runner = TaskRunner([NotifyDuringCycleTask(), StopExecAtGoalTask()], executer=executer)
    executer.task_runner = runner
    runner.step(executer)
    assert executer.stopped is True


def test_notify_dispatches_lifecycle_listeners():
    runner = TaskRunner([StartedEventTask()])
    executer = _FakeExecuter()
    runner.notify(StackEvent.EXECUTION_STARTED, executer=executer)
    assert StartedEventTask.calls == [StackEvent.EXECUTION_STARTED]


def test_harvest_plan_stack_event_notifies_once():
    listener = LocalPlanFailedListener()
    ego = EgoState(x=0.0, y=0.0)
    pm = PerceptionModel(ego_vehicle=ego)
    world = BasicSim(ego_state=ego, pm=PerceptionModel(ego_vehicle=ego))
    plan = LocalPlan(stack_event=StackEvent.LOCAL_PLAN_FAILED)
    local_planner = SimpleNamespace(
        replan=lambda **kwargs: None,
        get_local_plan=lambda: plan,
        global_plan=SimpleNamespace(stack_event=None),
        step=lambda state: None,
        stack_capabilities=frozenset(),
        stack_requirements=frozenset(),
        world_requirements=frozenset(),
    )
    executer = SyncExecuter(
        perception_model=pm,
        world=world,
        tasks=[listener],
        perception=None,
        global_planner=None,
        local_planner=local_planner,
        controller=None,
    )
    executer._replan_step()
    assert LocalPlanFailedListener.calls == [StackEvent.LOCAL_PLAN_FAILED]
    assert plan.stack_event is None

    LocalPlanFailedListener.calls = []
    executer._replan_step()
    assert LocalPlanFailedListener.calls == []


def test_non_inline_placement_falls_back_to_inline():
    ego = EgoState(x=0.0, y=0.0)
    pm = PerceptionModel(ego_vehicle=ego)
    world = BasicSim(ego_state=ego, pm=PerceptionModel(ego_vehicle=ego))
    executer = SyncExecuter(
        perception_model=pm,
        world=world,
        tasks=[ThreadPlacementTask()],
        perception=None,
        global_planner=None,
        local_planner=None,
        controller=None,
    )
    executer.dispatch_task(executer.task_runner.tasks[0])
    assert ThreadPlacementTask.ran is True
    assert executer.task_runner.tasks[0].placement is TaskPlacement.THREAD


def test_builtin_tasks_are_registered():
    assert "GoalArrivalMonitor" in TaskStrategy.registry
    assert "StopExecAtGoalTask" in TaskStrategy.registry
    assert "TelemetryTask" in TaskStrategy.registry
    assert TaskStrategy.registry["GoalArrivalMonitor"] is GoalArrivalMonitor
    assert TaskStrategy.registry["StopExecAtGoalTask"] is StopExecAtGoalTask
    assert TaskStrategy.registry["TelemetryTask"] is TelemetryTask


def test_factory_rejects_unknown_task_name(minimal_corridor_map_path):
    ExecutionSettings.c40_map = str(minimal_corridor_map_path.resolve())
    with pytest.raises(ValueError, match="execution task"):
        executor_factory(
            load_plugins=False,
            executer_type=SyncExecuter.__name__,
            bridge="BasicSim",
            execution_task_names=["DefinitelyNotARegisteredTask"],
            mapping_strategy_name="",
            global_planner_strategy_name="",
            local_planner_strategy_name="",
            controller_strategy_name="",
            perception_strategy_name="",
            localization_strategy_name="",
        )


def test_factory_wires_builtin_stop_at_goal(minimal_corridor_map_path):
    ExecutionSettings.c40_map = str(minimal_corridor_map_path.resolve())
    executer = executor_factory(
        load_plugins=False,
        executer_type=SyncExecuter.__name__,
        bridge="BasicSim",
        execution_task_names=["GoalArrivalMonitor", "StopExecAtGoalTask"],
        mapping_strategy_name="",
        global_planner_strategy_name="",
        local_planner_strategy_name="",
        controller_strategy_name="",
        perception_strategy_name="",
        localization_strategy_name="",
    )
    assert len(executer.task_runner.tasks) == 2
    assert isinstance(executer.task_runner.tasks[0], GoalArrivalMonitor)
    assert isinstance(executer.task_runner.tasks[1], StopExecAtGoalTask)
