"""Async combined planner worker must localize/perceive before replan."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
from avlite.c20_planning.c21_planning_model import LocalPlan
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
from avlite.c30_control.c31_control_model import ControlCommand
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c40_execution.c41_world_bridge import WorldBridge
from avlite.c40_execution.c45_async_threaded_executer import AsyncThreadedExecuter
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c50_common.c51_capabilities import StackCapability
from avlite.c50_common.c52_world_sensor_datatypes import SensorFrame


@dataclass
class _StubWorld(WorldBridge):
    ego_state: EgoState = field(default_factory=lambda: EgoState(x=0, y=0, theta=0))
    perception_model: Optional[PerceptionModel] = None
    world_capabilities = frozenset()
    stack_capabilities = frozenset()

    def control_ego_state(self, cmd: ControlCommand, dt: float = 0.01):
        pass

    def get_sensor_frame(self, agent_id: int = 0) -> SensorFrame:
        return SensorFrame(stamp=0.0)


class _RecordingPlanner(LocalPlanningStrategy):
    world_requirements = frozenset()
    stack_requirements = frozenset()
    stack_capabilities = frozenset({StackCapability.LOCAL_PLAN})

    def __init__(self, order: list[str]):
        self.order = order
        self.lap = 0

    def replan(self, perception_model=None, sensors=None):
        self.order.append("replan")

    def step(self, ego_state):
        pass

    def get_local_plan(self):
        return LocalPlan()

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


def test_combined_worker_localizes_and_perceives_before_replan():
    order: list[str] = []
    prev = ExecutionSettings.c41_world_stack_capabilities
    ExecutionSettings.c41_world_stack_capabilities = []  # force localization stage
    try:
        localization = SimpleNamespace(
            world_requirements=frozenset(),
            stack_requirements=frozenset(),
            stack_capabilities=frozenset({StackCapability.LOCALIZATION}),
            localize=lambda **kwargs: order.append("localize"),
            reset=lambda: None,
        )
        perception = SimpleNamespace(
            world_requirements=frozenset(),
            stack_requirements=frozenset(),
            stack_capabilities=frozenset(),
            perceive=lambda **kwargs: order.append("perceive"),
            reset=lambda: None,
        )
        exec_ = AsyncThreadedExecuter(
            perception_model=PerceptionModel(),
            perception=perception,
            localization=localization,
            global_planner=None,
            local_planner=_RecordingPlanner(order),
            controller=_StubController(),
            world=_StubWorld(),
            combined_perception_planning=True,
            control_dt=0.05,
            replan_dt=0.01,
            perception_dt=0.01,
            localization_dt=0.01,
        )
        done = threading.Event()

        def _watch():
            while "replan" not in order and not exec_.stopped:
                time.sleep(0.005)
            done.set()

        watcher = threading.Thread(target=_watch, daemon=True)
        watcher.start()
        exec_.step(
            call_replan=True,
            call_control=False,
            call_perceive=True,
            call_localize=True,
            pace_replan=False,
            pace_perception=False,
            pace_control=False,
            pace_sim=False,
            replan_dt=0.01,
            perception_dt=0.01,
            localization_dt=0.01,
            control_dt=0.05,
            sim_dt=0.01,
        )
        assert done.wait(timeout=2.0), f"timed out; order={order}"
        exec_.stop()
        watcher.join(timeout=1.0)

        # First combined iteration must update ego/obstacles before planning.
        assert "localize" in order
        assert "perceive" in order
        assert "replan" in order
        assert order.index("localize") < order.index("replan")
        assert order.index("perceive") < order.index("replan")
    finally:
        ExecutionSettings.c41_world_stack_capabilities = prev
