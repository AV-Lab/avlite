"""Regression: AsyncThreadedExecuter.stop() must be safe from a worker thread."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
from avlite.c30_control.c31_control_model import ControlCommand
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c40_execution.c41_world_bridge import WorldBridge
from avlite.c40_execution.c45_async_threaded_executer import AsyncThreadedExecuter
from avlite.c50_common.c51_capabilities import StackCapability


@dataclass
class _StubWorldBridge(WorldBridge):
    ego_state: EgoState = field(default_factory=lambda: EgoState(x=0, y=0, theta=0))
    perception_model: Optional[PerceptionModel] = None

    world_capabilities = frozenset()
    stack_capabilities = frozenset({StackCapability.LOCALIZATION})

    def control_ego_state(self, cmd: ControlCommand, dt: float = 0.01):
        pass


class _StubLocalPlanner(LocalPlanningStrategy):
    world_requirements = frozenset()
    stack_requirements = frozenset()
    stack_capabilities = frozenset({StackCapability.LOCAL_PLAN})

    def __init__(self):
        self.lap = 0
        self._tj = None

    def replan(self, perception_model=None, sensors=None):
        pass

    def step(self, ego_state):
        pass

    def get_local_plan(self):
        return None

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
        perception_model=PerceptionModel(),
        perception=None,
        global_planner=None,
        local_planner=_StubLocalPlanner(),
        controller=_StubController(),
        world=_StubWorldBridge(),
        control_dt=0.05,
    )


def test_stop_from_worker_thread_does_not_raise():
    """stop() must skip joining the calling worker (StopExecAtGoalTask path)."""
    exec_ = _make_async_executer()

    result = {"error": None}
    peer_exited = threading.Event()

    def peer():
        while not exec_._stop_event.is_set():
            time.sleep(0.01)
        peer_exited.set()

    def worker():
        try:
            exec_.stop()
        except Exception as exc:  # noqa: BLE001 — capture for assertion
            result["error"] = exc

    peer_thread = threading.Thread(target=peer, name="Peer", daemon=True)
    worker_thread = threading.Thread(target=worker, name="Controller", daemon=True)

    exec_.threads = [peer_thread, worker_thread]
    exec_.planner_thread = peer_thread
    exec_.controller_thread = worker_thread
    exec_.threads_started = True

    peer_thread.start()
    worker_thread.start()
    worker_thread.join(timeout=2.0)
    peer_exited.wait(timeout=2.0)
    peer_thread.join(timeout=2.0)

    assert result["error"] is None
    assert exec_._stop_event.is_set()
    assert exec_.threads_started is False
    assert exec_.threads == []
    assert not peer_thread.is_alive()
    assert not worker_thread.is_alive()


def test_create_threads_recreates_dead_planner():
    exec_ = _make_async_executer()
    dead = threading.Thread(target=lambda: None, name="DeadPlanner", daemon=True)
    dead.start()
    dead.join(timeout=1.0)
    assert not dead.is_alive()

    exec_.planner_thread = dead
    exec_.controller_thread = None
    exec_.create_threads()

    assert exec_.planner_thread is not dead
    assert exec_.planner_thread in exec_.threads
    assert exec_.planner_thread.name == "Planner"
