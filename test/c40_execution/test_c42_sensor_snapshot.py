"""One sensor snapshot per executer tick, shared by every stage."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import pytest

from avlite.c10_perception.c11_perception_model import EGO_AGENT_ID, EgoState, PerceptionModel
from avlite.c20_planning.c21_planning_model import LocalPlan
from avlite.c30_control.c31_control_model import AckermannControlCommand
from avlite.c40_execution.c44_sync_executer import SyncExecuter
from avlite.c40_execution.c45_async_threaded_executer import AsyncThreadedExecuter
from avlite.c40_execution.c41_world_bridge import WorldBridge
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c50_common.c51_capabilities import StackCapability
from avlite.c50_common.c52_world_sensor_datatypes import SensorFrame


@dataclass
class _CountingWorld(WorldBridge):
    """Hands out a distinct SensorFrame per fetch, so sharing is observable by identity."""

    ego_state: EgoState = field(default_factory=lambda: EgoState(x=0.0, y=0.0, theta=0.0))
    perception_model: Optional[PerceptionModel] = None
    world_capabilities = frozenset()
    stack_capabilities = frozenset()

    def __post_init__(self):
        self.fetches: list[SensorFrame] = []

    def control_ego_state(self, cmd, dt: Optional[float] = 0.01):
        pass

    def get_sensor_frame(self, agent_id: int = EGO_AGENT_ID) -> SensorFrame:
        frame = SensorFrame(stamp=float(len(self.fetches)))
        self.fetches.append(frame)
        return frame


def _make_exec(world: _CountingWorld, seen: dict) -> SyncExecuter:
    """Full stack of stubs that record the SensorFrame each stage was handed."""

    def perceive(*, perception_model=None, sensors=None):
        seen["perceive"] = sensors

    def localize(*, perception_model=None, sensors=None):
        seen["localize"] = sensors

    def replan(*, perception_model=None, sensors=None):
        seen["replan"] = sensors

    def control(ego, plan=None, control_dt=None, perception_model=None, sensors=None):
        seen["control"] = sensors
        return AckermannControlCommand()

    return SyncExecuter(
        perception_model=PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0)),
        world=world,
        perception=SimpleNamespace(
            world_requirements=frozenset(),
            stack_requirements=frozenset(),
            stack_capabilities=frozenset(),
            perceive=perceive,
        ),
        localization=SimpleNamespace(
            world_requirements=frozenset(),
            stack_requirements=frozenset(),
            stack_capabilities=frozenset({StackCapability.LOCALIZATION}),
            localize=localize,
        ),
        global_planner=None,
        local_planner=SimpleNamespace(
            world_requirements=frozenset(),
            stack_requirements=frozenset(),
            stack_capabilities=frozenset(),
            replan=replan,
            get_local_plan=lambda: LocalPlan(),
            step=lambda state: None,
            global_plan=None,
        ),
        controller=SimpleNamespace(
            world_requirements=frozenset(),
            stack_requirements=frozenset(),
            stack_capabilities=frozenset({StackCapability.CONTROL}),
            control=control,
        ),
    )


@pytest.fixture(autouse=True)
def _restore_stack_cap_filter():
    prev = ExecutionSettings.c41_world_stack_capabilities
    # Disable world ground truth so the localization stage runs from sensors.
    ExecutionSettings.c41_world_stack_capabilities = []
    yield
    ExecutionSettings.c41_world_stack_capabilities = prev


def test_one_tick_fetches_one_frame_shared_by_every_stage():
    world = _CountingWorld()
    seen: dict = {}
    exec_ = _make_exec(world, seen)

    exec_.step(
        sim_dt=0.01, perception_dt=0.0, replan_dt=0.0, control_dt=0.0, localization_dt=0.0,
    )

    assert len(world.fetches) == 1
    frame = world.fetches[0]
    assert set(seen) == {"localize", "perceive", "replan", "control"}
    assert all(s is frame for s in seen.values())


def test_tick_with_no_stage_due_skips_the_fetch():
    world = _CountingWorld()
    seen: dict = {}
    exec_ = _make_exec(world, seen)

    exec_.step(
        sim_dt=0.01, perception_dt=0.0, replan_dt=0.0, control_dt=0.0, localization_dt=0.0,
    )
    assert len(world.fetches) == 1

    # Every period now far exceeds the elapsed sim time, so nothing is due.
    exec_.step(
        sim_dt=0.01, perception_dt=1e6, replan_dt=1e6, control_dt=1e6, localization_dt=1e6,
    )
    assert len(world.fetches) == 1


def test_async_combined_worker_shares_one_snapshot_per_iteration():
    """Planner+perception combined mode must fetch once and share the frame."""
    world = _CountingWorld()
    seen: dict = {}

    def perceive(*, perception_model=None, sensors=None):
        seen["perceive"] = sensors
        # Perception runs after replan in the same iteration; stop once both saw the frame.
        exec_.stopped = True

    def replan(*, perception_model=None, sensors=None):
        seen["replan"] = sensors

    exec_ = AsyncThreadedExecuter(
        perception_model=PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0)),
        world=world,
        perception=SimpleNamespace(
            world_requirements=frozenset(),
            stack_requirements=frozenset(),
            stack_capabilities=frozenset(),
            perceive=perceive,
        ),
        localization=None,
        global_planner=None,
        local_planner=SimpleNamespace(
            world_requirements=frozenset(),
            stack_requirements=frozenset(),
            stack_capabilities=frozenset(),
            replan=replan,
            get_local_plan=lambda: LocalPlan(),
            step=lambda state: None,
            global_plan=None,
        ),
        controller=None,
        combined_perception_planning=True,
        perception_dt=0.01,
        replan_dt=0.01,
    )
    # Avoid the cold-start perception stall gate (dt_p from last=0 looks huge).
    exec_._perception_fps_tracker.last = time.time()
    exec_.call_perceive = True
    exec_.call_replan = True
    exec_.call_localize = False
    exec_.pace_perception = False
    exec_.pace_replan = False

    exec_.worker_planning()

    assert "replan" in seen and "perceive" in seen
    assert seen["replan"] is seen["perceive"]
    assert len(world.fetches) == 1
    assert seen["replan"] is world.fetches[0]


def test_async_idle_gates_skip_sensor_fetch():
    """When no stage module is active, the planner worker must not fetch sensors."""
    import threading

    world = _CountingWorld()
    exec_ = AsyncThreadedExecuter(
        perception_model=PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0)),
        world=world,
        perception=None,
        localization=None,
        global_planner=None,
        # Presence gates: without a local planner, do_replan stays false.
        local_planner=None,
        controller=None,
        combined_perception_planning=True,
        replan_dt=0.01,
    )
    exec_.call_perceive = False
    exec_.call_replan = True
    exec_.call_localize = False
    exec_.pace_replan = False  # 1 ms free-run sleep so stop is observed quickly

    worker = threading.Thread(target=exec_.worker_planning, daemon=True)
    worker.start()
    time.sleep(0.05)
    exec_.stopped = True
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert world.fetches == []
