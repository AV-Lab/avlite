"""Regression: toggling Control/Plan off must not orphan or duplicate async workers."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
from avlite.c20_planning.c21_planning_model import LocalPlan
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
from avlite.c30_control.c31_control_model import ControlCommand
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c40_execution.c41_world_bridge import WorldBridge
from avlite.c40_execution.c45_async_threaded_executer import AsyncThreadedExecuter
from avlite.c50_common.c51_capabilities import StackCapability


@dataclass
class _StubWorldBridge(WorldBridge):
    ego_state: EgoState = field(default_factory=lambda: EgoState(x=0, y=0, theta=0, velocity=0))
    perception_model: Optional[PerceptionModel] = None

    world_capabilities = frozenset()
    stack_capabilities = frozenset({StackCapability.LOCALIZATION})

    def control_ego_state(self, cmd: ControlCommand, dt: float = 0.01):
        pass

    def get_ego_state(self) -> EgoState:
        return self.ego_state


class _StubLocalPlanner(LocalPlanningStrategy):
    world_requirements = frozenset()
    stack_requirements = frozenset()
    stack_capabilities = frozenset({StackCapability.LOCAL_PLAN})

    def __init__(self):
        self.lap = 0
        self._plan = LocalPlan(
            path=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            velocity=[1.0, 1.0, 1.0],
        )

    def replan(self, perception_model=None, sensors=None):
        pass

    def step(self, ego_state):
        pass

    def get_local_plan(self):
        return self._plan

    def reset(self):
        pass

    def __init_subclass__(cls, **kwargs):
        pass


class _StubController(ControlStrategy, abstract=True):
    def control(
        self, ego, plan=None, control_dt=None, perception_model=None, sensors=None,
    ) -> ControlCommand:
        return ControlCommand()

    def reset(self):
        pass


def _make_async_executer() -> AsyncThreadedExecuter:
    return AsyncThreadedExecuter(
        perception_model=PerceptionModel(
            ego_vehicle=EgoState(x=0, y=0, theta=0, velocity=0)
        ),
        perception=None,
        global_planner=None,
        local_planner=_StubLocalPlanner(),
        controller=_StubController(),
        world=_StubWorldBridge(),
        control_dt=0.05,
        replan_dt=0.05,
    )


def _count_named(name: str) -> int:
    return sum(1 for t in threading.enumerate() if t.name == name and t.is_alive())


def _step(exec_: AsyncThreadedExecuter, *, call_replan: bool, call_control: bool) -> None:
    exec_.step(
        call_replan=call_replan,
        call_control=call_control,
        call_perceive=False,
        call_localize=False,
        replan_dt=0.05,
        control_dt=0.05,
        sim_dt=0.05,
    )


def test_control_toggle_off_does_not_duplicate_planner():
    """UI uncheck Control must exit the controller without orphaning/duplicating Planner."""
    exec_ = _make_async_executer()
    try:
        _step(exec_, call_replan=True, call_control=True)
        time.sleep(0.15)
        assert _count_named("Planner") == 1
        assert _count_named("Controller") == 1

        for _ in range(3):
            _step(exec_, call_replan=True, call_control=False)
            time.sleep(0.12)

        assert _count_named("Planner") == 1
        assert _count_named("Controller") == 0
        assert exec_.planner_thread is not None and exec_.planner_thread.is_alive()
    finally:
        exec_.stop()
        time.sleep(0.2)


def test_control_toggle_on_restarts_single_controller():
    """Re-checking Control must start exactly one Controller without duplicating Planner."""
    exec_ = _make_async_executer()
    try:
        _step(exec_, call_replan=True, call_control=True)
        time.sleep(0.15)

        _step(exec_, call_replan=True, call_control=False)
        time.sleep(0.15)
        assert _count_named("Controller") == 0

        _step(exec_, call_replan=True, call_control=True)
        time.sleep(0.15)

        assert _count_named("Planner") == 1
        assert _count_named("Controller") == 1
    finally:
        exec_.stop()
        time.sleep(0.2)


def test_plan_toggle_off_on_keeps_single_workers():
    """Same lifecycle for the Planning stage checkbox."""
    exec_ = _make_async_executer()
    try:
        _step(exec_, call_replan=True, call_control=True)
        time.sleep(0.15)

        for _ in range(2):
            _step(exec_, call_replan=False, call_control=True)
            time.sleep(0.12)
        assert _count_named("Planner") == 0
        assert _count_named("Controller") == 1

        _step(exec_, call_replan=True, call_control=True)
        time.sleep(0.15)
        assert _count_named("Planner") == 1
        assert _count_named("Controller") == 1
    finally:
        exec_.stop()
        time.sleep(0.2)
