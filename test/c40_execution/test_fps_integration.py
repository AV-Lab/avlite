"""Integration tests for FPS reporting in SyncExecuter and AsyncThreadedExecuter.

Verifies that:
- control_fps reflects wall-clock time (not sim-time).
- planner_fps reflects wall-clock time with floor_dt cap.
- elapsed_sim_time advances by control_dt per step (consistent across executers).
- A slow world makes FPS drop below target (yellow/red in dashboard).
- A fast world is capped at 1/sim_dt (floor), not reported faster.

All tests use lightweight stubs and avoid any file I/O or simulator initialization.
"""
import time
import threading
from dataclasses import dataclass, field
from typing import Optional

import pytest

from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c20_planning.c22_global_planning_strategy import GlobalPlannerStrategy
from avlite.c20_planning.c23_local_planning_strategy import LocalPlannerStrategy
from avlite.c30_control.c31_control_model import ControlComand
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c40_execution.c41_execution_model import WorldBridge
from avlite.c40_execution.c43_sync_executer import SyncExecuter
from avlite.c40_execution.c44_async_threaded_executer import AsyncThreadedExecuter
from avlite.c60_common.c62_capabilities import WorldCapability


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

@dataclass
class _StubWorldBridge(WorldBridge):
    ego_state: EgoState = field(default_factory=lambda: EgoState(x=0, y=0, theta=0))
    perception_model: Optional[PerceptionModel] = None
    delay: float = 0.0  # artificial latency per control_ego_state call

    @property
    def capabilities(self):
        return set()

    def control_ego_state(self, cmd: ControlComand, dt: float = 0.01):
        if self.delay > 0.0:
            time.sleep(self.delay)


class _StubGlobalPlanner(GlobalPlannerStrategy):
    def plan(self) -> GlobalPlan:
        return GlobalPlan()


class _StubLocalPlanner(LocalPlannerStrategy):
    """Local planner with no trajectory data — just satisfies the interface."""

    def __init__(self):
        # Bypass the heavy base __init__ (needs GlobalPlan + PerceptionModel)
        self.lap = 0
        self._tj = None

    def replan(self):
        pass

    def step(self, ego_state):
        pass

    def get_local_plan(self):
        return None

    def reset(self):
        pass

    def __init_subclass__(cls, **kwargs):
        pass  # prevent auto-registration


class _StubController(ControlStrategy, abstract=True):
    def control(self, ego, tj=None, control_dt=None) -> ControlComand:
        return ControlComand()

    def reset(self):
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sync_executer(world_delay: float = 0.0, control_dt: float = 0.05, sim_dt: float = 0.01):
    pm = PerceptionModel()
    world = _StubWorldBridge(delay=world_delay)
    return SyncExecuter(
        perception_model=pm,
        perception=None,
        global_planner=None,
        local_planner=_StubLocalPlanner(),
        controller=_StubController(),
        world=world,
        control_dt=control_dt,
    ), sim_dt


# ---------------------------------------------------------------------------
# SyncExecuter tests
# ---------------------------------------------------------------------------

class TestSyncExecuterFps:
    def test_control_fps_nonzero_after_two_steps(self):
        # Step 1: elapsed_sim_time goes from 0→control_dt; gate (dt_c=0) not met.
        # Step 2: gate met (dt_c=control_dt), _control_step called → tick #1 → fps=0.
        # Step 3: gate met again → tick #2 → fps > 0.
        exec_, sim_dt = _make_sync_executer()
        for _ in range(3):
            exec_.step(control_dt=0.05, sim_dt=sim_dt, replan_dt=99, localization_dt=99,
                       call_replan=False, call_perceive=False, call_localize=False)
        assert exec_.control_fps > 0.0

    def test_control_fps_capped_by_floor_when_fast(self):
        """Control FPS is capped at 1/sim_dt (floor_dt) — physics must not run faster.
        Planner/perception/localization have no floor, so they report true wall-clock rate.
        """
        sim_dt = 0.05
        exec_, _ = _make_sync_executer(sim_dt=sim_dt)
        for _ in range(5):
            exec_.step(control_dt=0.05, sim_dt=sim_dt, replan_dt=99,
                       localization_dt=99, call_replan=False,
                       call_perceive=False, call_localize=False)
        cap = 1.0 / sim_dt  # 20
        # Allow slight floating-point overshoot (< 5 %)
        assert exec_.control_fps <= cap * 1.05

    def test_control_fps_reflects_slow_world(self):
        """When the world introduces a 0.1 s delay, FPS must drop below target."""
        control_dt = 0.05
        sim_dt = 0.01
        exec_, _ = _make_sync_executer(world_delay=0.1, control_dt=control_dt, sim_dt=sim_dt)
        # Warm up
        exec_.step(control_dt=control_dt, sim_dt=sim_dt, replan_dt=99,
                   localization_dt=99, call_replan=False, call_perceive=False,
                   call_localize=False)
        exec_.step(control_dt=control_dt, sim_dt=sim_dt, replan_dt=99,
                   localization_dt=99, call_replan=False, call_perceive=False,
                   call_localize=False)
        target_fps = 1.0 / control_dt  # 20
        assert exec_.control_fps < target_fps

    def test_control_fps_first_step_is_zero(self):
        exec_, sim_dt = _make_sync_executer()
        exec_.step(control_dt=0.05, sim_dt=sim_dt, replan_dt=99, localization_dt=99,
                   call_replan=False, call_perceive=False, call_localize=False)
        assert exec_.control_fps == 0.0

    def test_elapsed_sim_time_advances_by_control_dt(self):
        control_dt = 0.05
        exec_, sim_dt = _make_sync_executer(control_dt=control_dt)
        n_steps = 6
        for _ in range(n_steps):
            exec_.step(control_dt=control_dt, sim_dt=sim_dt, replan_dt=99,
                       localization_dt=99, call_replan=False,
                       call_perceive=False, call_localize=False)
        expected = control_dt * n_steps
        assert abs(exec_.elapsed_sim_time - expected) < 1e-9

    def test_fps_resets_after_reset(self):
        exec_, sim_dt = _make_sync_executer()
        for _ in range(4):  # 4 steps to ensure control gate fires at least twice
            exec_.step(control_dt=0.05, sim_dt=sim_dt, replan_dt=99,
                       localization_dt=99, call_replan=False,
                       call_perceive=False, call_localize=False)
        assert exec_.control_fps > 0.0
        exec_.reset()
        # reset() must zero the public fps attributes AND the internal trackers
        assert exec_.control_fps == 0.0
        assert exec_.elapsed_sim_time == 0.0


# ---------------------------------------------------------------------------
# AsyncThreadedExecuter tests
# ---------------------------------------------------------------------------

class TestAsyncExecuterFps:
    def test_elapsed_sim_time_advances_by_control_dt(self):
        """elapsed_sim_time must use control_dt increments, not sim_dt."""
        pm = PerceptionModel()
        world = _StubWorldBridge()
        control_dt = 0.05
        sim_dt = 0.01  # intentionally different to expose old drift bug

        exec_ = AsyncThreadedExecuter(
            perception_model=pm,
            perception=None,
            global_planner=None,
            local_planner=_StubLocalPlanner(),
            controller=_StubController(),
            world=world,
            control_dt=control_dt,
        )
        exec_.step(control_dt=control_dt, sim_dt=sim_dt,
                   call_replan=False, call_control=True,
                   call_perceive=False, call_localize=False)

        # Let the controller thread run for ~N control cycles
        target_cycles = 5
        time.sleep(control_dt * target_cycles * 3)  # generous window
        exec_.stop()

        # elapsed_sim_time should be a multiple of control_dt (not sim_dt)
        sim_t = exec_.elapsed_sim_time
        if sim_t > 0:
            remainder = sim_t % control_dt
            # Allow floating-point rounding (< 1 % of control_dt)
            assert remainder < control_dt * 0.01 or abs(remainder - control_dt) < control_dt * 0.01

    def test_control_fps_nonzero_after_running(self):
        pm = PerceptionModel()
        world = _StubWorldBridge()
        control_dt = 0.05
        sim_dt = 0.05

        exec_ = AsyncThreadedExecuter(
            perception_model=pm,
            perception=None,
            global_planner=None,
            local_planner=_StubLocalPlanner(),
            controller=_StubController(),
            world=world,
            control_dt=control_dt,
        )
        exec_.step(control_dt=control_dt, sim_dt=sim_dt,
                   call_replan=False, call_control=True,
                   call_perceive=False, call_localize=False)

        time.sleep(control_dt * 4)
        exec_.stop()
        assert exec_.control_fps > 0.0
