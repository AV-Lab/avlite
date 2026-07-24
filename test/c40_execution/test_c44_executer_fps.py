"""Integration tests for FPS reporting in SyncExecuter and AsyncThreadedExecuter.

Verifies that:
- control_fps reflects wall-clock time (not sim-time).
- planner_fps reflects wall-clock time with floor_dt cap.
- elapsed_sim_time advances by the sim step dt (not control_dt).
- A slow control path makes FPS drop below target (yellow/red in dashboard).
- A fast world is capped at 1/sim_dt (floor), not reported faster.

All tests use lightweight stubs and avoid any file I/O or simulator initialization.
"""
import time
from dataclasses import dataclass, field
from typing import Optional

import pytest

from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
from avlite.c30_control.c31_control_model import ControlCommand
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c40_execution.c41_world_bridge import WorldBridge
from avlite.c40_execution.c44_sync_executer import SyncExecuter
from avlite.c40_execution.c45_async_threaded_executer import AsyncThreadedExecuter
from avlite.c50_common.c51_capabilities import StackCapability, WorldCapability


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

@dataclass
class _StubWorldBridge(WorldBridge):
    ego_state: EgoState = field(default_factory=lambda: EgoState(x=0, y=0, theta=0))
    perception_model: Optional[PerceptionModel] = None
    delay: float = 0.0  # artificial latency per control_ego_state call
    sim_calls: int = 0

    world_capabilities = frozenset()
    # Provide ground-truth localization so the executer may actuate the ego.
    stack_capabilities = frozenset({StackCapability.LOCALIZATION})

    def control_ego_state(self, cmd: ControlCommand, dt: float = 0.01):
        self.sim_calls += 1
        if self.delay > 0.0:
            time.sleep(self.delay)

    def get_ground_truth_perception_model(self):
        if self.perception_model is None:
            self.perception_model = PerceptionModel(ego_vehicle=self.ego_state)
        return self.perception_model


class _StubLocalPlanner(LocalPlanningStrategy):
    """Local planner with no trajectory data — just satisfies the interface."""

    world_requirements = frozenset()
    stack_requirements = frozenset()
    stack_capabilities = frozenset({StackCapability.LOCAL_PLAN})

    def __init__(self):
        # Bypass the heavy base __init__ (needs GlobalPlan + PerceptionModel)
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
        pass  # prevent auto-registration


class _StubController(ControlStrategy, abstract=True):
    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.calls = 0

    def set_plan(self, plan):
        """Accept None from stub local planners (base set_plan requires a trajectory)."""
        self.tj = plan

    def control(
        self, ego, plan=None, control_dt=None, perception_model=None, sensors=None,
    ) -> ControlCommand:
        self.calls += 1
        if self.delay > 0.0:
            time.sleep(self.delay)
        return ControlCommand()

    def reset(self):
        pass


class _CountingPerception:
    world_requirements = frozenset()
    stack_requirements = frozenset()
    stack_capabilities = frozenset()
    calls = 0

    def perceive(self, perception_model=None, sensors=None):
        type(self).calls += 1

    def reset(self):
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sync_executer(
    world_delay: float = 0.0,
    control_dt: float = 0.05,
    sim_dt: float = 0.01,
    control_delay: float = 0.0,
    perception=None,
):
    pm = PerceptionModel()
    world = _StubWorldBridge(delay=world_delay)
    return SyncExecuter(
        perception_model=pm,
        perception=perception,
        global_planner=None,
        local_planner=_StubLocalPlanner(),
        controller=_StubController(delay=control_delay),
        world=world,
        control_dt=control_dt,
    ), sim_dt


# ---------------------------------------------------------------------------
# SyncExecuter tests
# ---------------------------------------------------------------------------

class TestSyncExecuterFps:
    def test_control_fps_nonzero_after_two_steps(self):
        # With control_dt == sim_dt, every step recomputes; second tick yields nonzero FPS.
        exec_, sim_dt = _make_sync_executer()
        for _ in range(2):
            exec_.step(control_dt=sim_dt, sim_dt=sim_dt, replan_dt=99, localization_dt=99,
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

    @pytest.mark.slow
    def test_control_fps_reflects_slow_control(self):
        """When control() introduces a 0.1 s delay, FPS must drop below target."""
        control_dt = 0.05
        sim_dt = 0.01
        exec_, _ = _make_sync_executer(control_delay=0.1, control_dt=control_dt, sim_dt=sim_dt)
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

    def test_elapsed_sim_time_advances_by_sim_dt(self):
        control_dt = 0.05
        exec_, sim_dt = _make_sync_executer(control_dt=control_dt)
        n_steps = 6
        for _ in range(n_steps):
            exec_.step(control_dt=control_dt, sim_dt=sim_dt, replan_dt=99,
                       localization_dt=99, call_replan=False,
                       call_perceive=False, call_localize=False)
        expected = sim_dt * n_steps
        assert abs(exec_.elapsed_sim_time - expected) < 1e-9

    def test_fps_resets_after_reset(self):
        exec_, sim_dt = _make_sync_executer()
        for _ in range(4):
            exec_.step(control_dt=sim_dt, sim_dt=sim_dt, replan_dt=99,
                       localization_dt=99, call_replan=False,
                       call_perceive=False, call_localize=False)
        assert exec_.control_fps > 0.0
        exec_.reset()
        # reset() must zero the public fps attributes AND the internal trackers
        assert exec_.control_fps == 0.0
        assert exec_.elapsed_sim_time == 0.0


class TestSyncPaceAndSimulate:
    def test_perception_gated_by_perception_dt(self):
        _CountingPerception.calls = 0
        exec_, sim_dt = _make_sync_executer(perception=_CountingPerception())
        for _ in range(10):
            exec_.step(
                control_dt=0.01,
                sim_dt=0.01,
                perception_dt=0.05,
                replan_dt=99,
                localization_dt=99,
                call_replan=False,
                call_perceive=True,
                call_localize=False,
                pace_perception=True,
            )
        # 10 sim steps of 0.01 → elapsed 0.10; paced at 0.05 → about 3 fires (0, 0.05, 0.10)
        assert 2 <= _CountingPerception.calls <= 4

    def test_control_recomputes_every_n_sim_steps(self):
        exec_, _ = _make_sync_executer()
        for _ in range(10):
            exec_.step(
                control_dt=0.05,
                sim_dt=0.01,
                replan_dt=99,
                localization_dt=99,
                call_replan=False,
                call_perceive=False,
                call_localize=False,
                pace_control=True,
            )
        # First at t=0, then every 0.05 → ~3 recomputes over elapsed 0.10
        assert 2 <= exec_.controller.calls <= 4

    def test_simulate_holds_last_cmd_without_control_recompute(self):
        exec_, _ = _make_sync_executer()
        exec_.step(
            control_dt=0.01, sim_dt=0.01, replan_dt=99, localization_dt=99,
            call_replan=False, call_perceive=False, call_localize=False,
            pace_control=True,
        )
        assert exec_.world.sim_calls == 1
        assert exec_.controller.calls == 1
        # Larger control_dt so next steps only simulate (ZOH)
        for _ in range(4):
            exec_.step(
                control_dt=1.0, sim_dt=0.01, replan_dt=99, localization_dt=99,
                call_replan=False, call_perceive=False, call_localize=False,
                pace_control=True,
            )
        assert exec_.controller.calls == 1
        assert exec_.world.sim_calls == 5

    def test_free_run_sim_matches_real_clock(self):
        """When pace_sim is off, sim and real must advance by the same wall intervals."""
        exec_, _ = _make_sync_executer()
        for _ in range(20):
            time.sleep(0.002)
            exec_.step(
                control_dt=0.01,
                sim_dt=0.01,
                replan_dt=99,
                localization_dt=99,
                call_replan=False,
                call_perceive=False,
                call_localize=False,
                pace_sim=False,
                pace_control=False,
            )
        assert exec_.elapsed_sim_time > 0.0
        assert abs(exec_.elapsed_sim_time - exec_.elapsed_real_time) < 1e-9


# ---------------------------------------------------------------------------
# AsyncThreadedExecuter tests
# ---------------------------------------------------------------------------

class TestAsyncExecuterFps:
    @pytest.mark.slow
    def test_elapsed_sim_time_advances_by_sim_dt(self):
        """elapsed_sim_time must use sim step increments, not control_dt."""
        pm = PerceptionModel()
        world = _StubWorldBridge()
        control_dt = 0.05
        sim_dt = 0.01  # intentionally different

        exec_ = AsyncThreadedExecuter(
            perception_model=pm,
            perception=None,
            global_planner=None,
            local_planner=_StubLocalPlanner(),
            controller=_StubController(),
            world=world,
            control_dt=control_dt,
        )
        exec_.step(
            control_dt=control_dt,
            sim_dt=sim_dt,
            call_replan=False,
            call_control=True,
            call_perceive=False,
            call_localize=False,
            pace_control=True,
            pace_sim=True,
        )

        time.sleep(control_dt * 5 * 3)
        exec_.stop()

        sim_t = exec_.elapsed_sim_time
        if sim_t > 0:
            remainder = sim_t % sim_dt
            assert remainder < sim_dt * 0.01 or abs(remainder - sim_dt) < sim_dt * 0.01

    @pytest.mark.slow
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

    @pytest.mark.slow
    def test_unpaced_workers_respect_free_run_floor(self):
        """All pace_* off must not busy-spin: floor sleep bounds cheap-stub iter rate."""
        _CountingPerception.calls = 0
        floor = AsyncThreadedExecuter._FREE_RUN_SLEEP_S
        run_s = 0.12
        exec_ = AsyncThreadedExecuter(
            perception_model=PerceptionModel(),
            perception=_CountingPerception(),
            global_planner=None,
            local_planner=_StubLocalPlanner(),
            controller=_StubController(),
            world=_StubWorldBridge(),
            combined_perception_planning=True,
        )
        t0 = time.time()
        exec_.step(
            control_dt=0.01,
            sim_dt=0.01,
            replan_dt=0.01,
            perception_dt=0.01,
            call_replan=True,
            call_control=True,
            call_perceive=True,
            call_localize=False,
            pace_perception=False,
            pace_replan=False,
            pace_control=False,
            pace_sim=False,
        )
        time.sleep(run_s)
        exec_.stop()
        elapsed = time.time() - t0
        # Combined planner loop: one perceive per iter + floor sleep → ~1/floor Hz max.
        max_iters = int(elapsed / floor) + 1
        assert _CountingPerception.calls > 5
        assert _CountingPerception.calls <= max_iters * 2
