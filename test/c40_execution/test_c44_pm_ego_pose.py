"""PerceptionModel-only ego pose: GT sync vs localization ownership."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest

from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
from avlite.c10_perception.c13_localization_strategy import LocalizationStrategy
from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy
from avlite.c30_control.c31_control_model import ControlCommand
from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c40_execution.c41_world_bridge import WorldBridge
from avlite.c40_execution.c44_sync_executer import SyncExecuter
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c50_common.c51_capabilities import StackCapability
from avlite.c60_apps.c62_factory import executor_factory


@dataclass
class _PlantWorld(WorldBridge):
    """World ego advances on simulate; separate from stack PM ego."""

    ego_state: EgoState = field(default_factory=lambda: EgoState(x=0.0, y=0.0, theta=0.0))
    perception_model: Optional[PerceptionModel] = None
    advance_dx: float = 1.0
    world_capabilities = frozenset()
    stack_capabilities = frozenset({StackCapability.LOCALIZATION})

    def control_ego_state(self, cmd: ControlCommand, dt: float = 0.01):
        self.ego_state.x += self.advance_dx

    def get_ground_truth_perception_model(self):
        if self.perception_model is None:
            self.perception_model = PerceptionModel(ego_vehicle=self.ego_state)
        return self.perception_model


class _StubLocalPlanner(LocalPlanningStrategy):
    world_requirements = frozenset()
    stack_requirements = frozenset()
    stack_capabilities = frozenset({StackCapability.LOCAL_PLAN})

    def __init__(self):
        self.lap = 0
        self._tj = None
        self.last_step_ego = None

    def replan(self, perception_model=None, sensors=None):
        pass

    def step(self, ego_state):
        self.last_step_ego = (ego_state.x, ego_state.y)

    def get_local_plan(self):
        return None

    def reset(self):
        pass

    def __init_subclass__(cls, **kwargs):
        pass


class _RecordingController(ControlStrategy, abstract=True):
    # Default ControlStrategy requires LOCALIZATION; keep that for GT / loc tests.
    def __init__(self):
        self.seen_xy: list[tuple[float, float]] = []

    def control(
        self, ego, plan=None, control_dt=None, perception_model=None, sensors=None,
    ) -> ControlCommand:
        self.seen_xy.append((ego.x, ego.y))
        return ControlCommand()

    def reset(self):
        pass


class _ControllerNoLocReq(_RecordingController, abstract=True):
    """Assembleable when LOCALIZATION is unavailable (to assert _can_actuate)."""

    stack_requirements = frozenset()


class _StubLocalization(LocalizationStrategy, abstract=True):
    """Writes a fixed estimated pose into pm.ego_vehicle."""

    stack_capabilities = frozenset({StackCapability.LOCALIZATION})

    def __init__(self, perception_model: PerceptionModel, x: float = 50.0, y: float = 60.0):
        self.perception_model = perception_model
        self.x = x
        self.y = y
        self.calls = 0

    def localize(self, perception_model=None, sensors=None) -> None:
        self.calls += 1
        pm = perception_model if perception_model is not None else self.perception_model
        pm.ego_vehicle.x = self.x
        pm.ego_vehicle.y = self.y
        pm.ego_vehicle.theta = 0.0
        pm.ego_vehicle.velocity = 0.0

    def reset(self):
        pass


def _make_exec(
    *,
    world: _PlantWorld,
    pm: PerceptionModel,
    localization=None,
    controller=None,
    local_planner=None,
) -> SyncExecuter:
    return SyncExecuter(
        perception_model=pm,
        perception=None,
        global_planner=None,
        local_planner=local_planner or _StubLocalPlanner(),
        controller=controller or _RecordingController(),
        world=world,
        localization=localization,
        control_dt=0.01,
        localization_dt=0.0,
    )


@pytest.fixture(autouse=True)
def _restore_stack_cap_filter():
    prev = ExecutionSettings.c41_world_stack_capabilities
    yield
    ExecutionSettings.c41_world_stack_capabilities = prev


def test_gt_on_next_tick_control_sees_world_advanced_pose():
    """GT LOCALIZATION: after simulate, next tick's control sees world pose in PM."""
    ExecutionSettings.c41_world_stack_capabilities = None  # all GT caps on
    world_ego = EgoState(x=0.0, y=0.0, theta=0.0)
    stack_ego = EgoState(x=0.0, y=0.0, theta=0.0)
    assert world_ego is not stack_ego
    world = _PlantWorld(ego_state=world_ego, advance_dx=2.5)
    pm = PerceptionModel(ego_vehicle=stack_ego)
    ctrl = _RecordingController()
    exec_ = _make_exec(world=world, pm=pm, controller=ctrl)

    assert exec_.ego_state is pm.ego_vehicle

    exec_.step(
        sim_dt=0.01, control_dt=0.01, replan_dt=99, localization_dt=0,
        call_replan=False, call_perceive=False, call_localize=False,
    )
    assert world_ego.x == pytest.approx(2.5)
    # Pose update for this tick was at start (0); plant advanced after control.
    assert ctrl.seen_xy[0] == pytest.approx((0.0, 0.0))

    exec_.step(
        sim_dt=0.01, control_dt=0.01, replan_dt=99, localization_dt=0,
        call_replan=False, call_perceive=False, call_localize=False,
    )
    assert ctrl.seen_xy[1] == pytest.approx((2.5, 0.0))
    assert pm.ego_vehicle.x == pytest.approx(2.5)
    assert pm.ego_vehicle is not world_ego


def test_gt_off_localization_owns_pm_world_can_diverge():
    """GT off: localization writes PM; plant may diverge; stack does not use world ego as belief."""
    ExecutionSettings.c41_world_stack_capabilities = []  # disable GT LOCALIZATION
    world_ego = EgoState(x=0.0, y=0.0, theta=0.0)
    stack_ego = EgoState(x=0.0, y=0.0, theta=0.0)
    world = _PlantWorld(ego_state=world_ego, advance_dx=3.0)
    pm = PerceptionModel(ego_vehicle=stack_ego)
    loc = _StubLocalization(pm, x=50.0, y=60.0)
    ctrl = _RecordingController()
    planner = _StubLocalPlanner()
    exec_ = _make_exec(
        world=world, pm=pm, localization=loc, controller=ctrl, local_planner=planner,
    )

    exec_.step(
        sim_dt=0.01, control_dt=0.01, replan_dt=99, localization_dt=0,
        call_replan=False, call_perceive=False, call_localize=True,
    )
    assert loc.calls == 1
    assert ctrl.seen_xy[0] == pytest.approx((50.0, 60.0))
    assert planner.last_step_ego == pytest.approx((50.0, 60.0))
    assert world_ego.x == pytest.approx(3.0)
    assert pm.ego_vehicle.x == pytest.approx(50.0)
    assert pm.ego_vehicle is not world_ego

    # Second tick: localize again (same estimate); world keeps advancing.
    exec_.step(
        sim_dt=0.01, control_dt=0.01, replan_dt=99, localization_dt=0,
        call_replan=False, call_perceive=False, call_localize=True,
    )
    assert world_ego.x == pytest.approx(6.0)
    assert ctrl.seen_xy[1] == pytest.approx((50.0, 60.0))
    assert pm.ego_vehicle.x == pytest.approx(50.0)


def test_gt_off_no_localization_no_world_to_pm_sync_cannot_actuate():
    ExecutionSettings.c41_world_stack_capabilities = []
    world_ego = EgoState(x=1.0, y=2.0, theta=0.0)
    stack_ego = EgoState(x=0.0, y=0.0, theta=0.0)
    world = _PlantWorld(ego_state=world_ego, advance_dx=5.0)
    pm = PerceptionModel(ego_vehicle=stack_ego)
    ctrl = _ControllerNoLocReq()
    exec_ = _make_exec(world=world, pm=pm, localization=None, controller=ctrl)

    assert not exec_._can_actuate()
    exec_.step(
        sim_dt=0.01, control_dt=0.01, replan_dt=99, localization_dt=0,
        call_replan=False, call_perceive=False, call_localize=True,
    )
    assert ctrl.seen_xy == []
    # No GT sync: PM stays at stack initial; plant still integrates ZOH (may be no-op cmd).
    assert pm.ego_vehicle.x == pytest.approx(0.0)
    assert pm.ego_vehicle.y == pytest.approx(0.0)
    assert world_ego.x != pm.ego_vehicle.x


def test_factory_world_ego_is_not_pm_ego(minimal_corridor_map_path):
    ExecutionSettings.c40_map = str(minimal_corridor_map_path.resolve())
    ExecutionSettings.c40_mapping = ""
    exec_ = executor_factory(
        load_plugins=False,
        executer_type=SyncExecuter.__name__,
        bridge="BasicSim",
        perception_strategy_name="",
        localization_strategy_name="",
        mapping_strategy_name="",
        global_planner_strategy_name="",
        local_planner_strategy_name="",
        controller_strategy_name="",
    )
    assert exec_.ego_state is exec_.pm.ego_vehicle
    assert exec_.world.get_ego_state() is not exec_.pm.ego_vehicle
    assert exec_.world.get_ego_state().x == pytest.approx(exec_.pm.ego_vehicle.x)
    assert exec_.world.get_ego_state().y == pytest.approx(exec_.pm.ego_vehicle.y)


def test_control_align_must_teleport_world_or_gt_undoes_stack_only_write():
    """Control Align used to assign only ``exec.ego_state`` (stack PM).

    After the world/stack ego split, GT localization copies world → PM each tick,
    so a stack-only write is discarded. Align must teleport the plant and sync PM
    (same dual-write as ``VisualizerApp.teleport_ego``).
    """
    ExecutionSettings.c41_world_stack_capabilities = None  # GT LOCALIZATION on
    world_ego = EgoState(x=10.0, y=20.0, theta=0.0)
    stack_ego = EgoState(x=10.0, y=20.0, theta=0.0)
    world = _PlantWorld(ego_state=world_ego, advance_dx=0.0)
    pm = PerceptionModel(ego_vehicle=stack_ego)
    exec_ = _make_exec(world=world, pm=pm)

    # Broken Align pattern (stack only) — undone by the next GT tick.
    exec_.ego_state.x, exec_.ego_state.y = 100.0, 200.0
    exec_.step(
        sim_dt=0.01, control_dt=0.01, replan_dt=99, localization_dt=0,
        call_replan=False, call_perceive=False, call_localize=False, call_control=False,
    )
    assert pm.ego_vehicle.x == pytest.approx(10.0)
    assert world_ego.x == pytest.approx(10.0)

    # Correct Align pattern: move plant, then sync stack (teleport_ego).
    world_ego.x, world_ego.y = 100.0, 200.0
    pm.ego_vehicle.copy_from(world.get_ego_state())
    exec_.step(
        sim_dt=0.01, control_dt=0.01, replan_dt=99, localization_dt=0,
        call_replan=False, call_perceive=False, call_localize=False, call_control=False,
    )
    assert world_ego.x == pytest.approx(100.0)
    assert world_ego.y == pytest.approx(200.0)
    assert pm.ego_vehicle.x == pytest.approx(100.0)
    assert pm.ego_vehicle.y == pytest.approx(200.0)


def test_manual_world_control_must_sync_stack_pm():
    """Control Step / Steer used to call ``world.control_ego_state`` only.

    Stack PM ego then lagged until the next GT tick, so the UI and subsequent
    manual control steps read a stale pose. Dual-write like teleport.
    """
    world_ego = EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0)
    stack_ego = EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0)
    world = _PlantWorld(ego_state=world_ego, advance_dx=1.5)
    pm = PerceptionModel(ego_vehicle=stack_ego)

    # Broken pattern: plant moves, stack stays put.
    world.control_ego_state(ControlCommand(), dt=0.01)
    assert world_ego.x == pytest.approx(1.5)
    assert pm.ego_vehicle.x == pytest.approx(0.0)

    # Correct pattern (VisualizerApp.apply_world_control).
    world.control_ego_state(ControlCommand(), dt=0.01)
    pm.ego_vehicle.copy_from(world.get_ego_state())
    assert world_ego.x == pytest.approx(3.0)
    assert pm.ego_vehicle.x == pytest.approx(3.0)
    assert pm.ego_vehicle.y == pytest.approx(world_ego.y)
