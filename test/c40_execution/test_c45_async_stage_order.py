"""Async combined planner worker must localize/perceive before replan."""

from __future__ import annotations

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


def _has_ordered_triple(order: list[str]) -> bool:
    """True if some localize→perceive→replan subsequence appears in that order."""
    try:
        i_loc = order.index("localize")
        i_pr = order.index("perceive", i_loc)
        i_rp = order.index("replan", i_pr)
    except ValueError:
        return False
    return i_loc < i_pr < i_rp


def test_combined_worker_localizes_and_perceives_before_replan():
    order: list[str] = []
    prev = ExecutionSettings.c41_world_stack_capabilities
    # Empty filter disables world GT localization so the localization stage runs.
    ExecutionSettings.c41_world_stack_capabilities = []
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
            localization_dt=0.0,
        )
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
            localization_dt=0.0,
            control_dt=0.05,
            sim_dt=0.01,
        )
        deadline = time.time() + 2.0
        while time.time() < deadline and not _has_ordered_triple(order):
            time.sleep(0.01)
        exec_.stop()

        assert _has_ordered_triple(order), f"missing localize→perceive→replan; order={order[:30]}"
        # Whenever perceive and replan both fire, perceive must not follow replan
        # in the same iteration. After the FPS warm-up skip, the stable pattern is
        # localize, perceive, replan (possibly with an initial localize, replan).
        for i in range(len(order) - 1):
            if order[i] == "perceive":
                # Next stage in the same iteration is replan (localize already ran).
                assert order[i + 1] == "replan", order[i : i + 3]
    finally:
        ExecutionSettings.c41_world_stack_capabilities = prev
