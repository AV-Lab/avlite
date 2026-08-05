"""Tests for VisualizerApp.apply_global_plan and apply_world_control."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from avlite import TrajectoryTracker
from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c30_control.c31_control_model import ControlCommand
from avlite.plugins.p60_visualizer_tk.p61_visualizer_app import VisualizerApp


def _sample_plan() -> GlobalPlan:
    path = [(0.0, 0.0), (10.0, 0.0)]
    traj = TrajectoryTracker(path=path, velocity=[1.0, 1.0])
    return GlobalPlan(path=path, velocity=[1.0, 1.0], trajectory=traj)


def test_apply_global_plan_delegates_to_exec_when_defined():
    app = VisualizerApp.__new__(VisualizerApp)

    class _ExecWithApply:
        def apply_global_plan(self, global_plan, ego_xy=None):
            self.called_with = (global_plan, ego_xy)

    stub = _ExecWithApply()
    app.exec = stub
    plan = _sample_plan()
    VisualizerApp.apply_global_plan(app, plan, ego_xy=(1.0, 2.0))
    assert stub.called_with[0] is plan
    assert stub.called_with[1] == (1.0, 2.0)


def test_apply_global_plan_updates_local_proxy_when_exec_has_no_method():
    app = VisualizerApp.__new__(VisualizerApp)

    class _PlainExec:
        def __init__(self):
            self.ego_state = MagicMock(x=0.0, y=0.0)
            self.local_planner = MagicMock()
            self.controller = MagicMock()

    stub = _PlainExec()
    app.exec = stub
    plan = _sample_plan()
    VisualizerApp.apply_global_plan(app, plan, ego_xy=(3.0, 4.0))
    stub.local_planner.set_global_plan.assert_called_once_with(plan, ego_xy=(3.0, 4.0))
    stub.controller.set_trajectory_tracker.assert_called_once_with(plan.trajectory)
    stub.controller.reset.assert_called_once()


def test_apply_world_control_dual_writes_plant_then_stack_pm():
    """After world/stack ego split, Control Step/Steer must sync pm.ego_vehicle."""
    app = VisualizerApp.__new__(VisualizerApp)

    world_ego = EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0)
    stack_ego = EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0)

    class _World:
        def __init__(self, ego):
            self._ego = ego

        def control_ego_state(self, cmd, dt):
            del cmd
            self._ego.x += self._ego.velocity * dt
            self._calls = getattr(self, "_calls", 0) + 1

        def get_ego_state(self):
            return self._ego

    class _Exec:
        def __init__(self):
            self.world = _World(world_ego)
            self.pm = PerceptionModel(ego_vehicle=stack_ego)

    stub = _Exec()
    app.exec = stub
    VisualizerApp.apply_world_control(app, ControlCommand(), dt=0.2)

    assert stub.world._calls == 1
    assert world_ego.x == pytest.approx(1.0)
    assert stub.pm.ego_vehicle.x == pytest.approx(1.0)
    assert stub.pm.ego_vehicle.velocity == pytest.approx(5.0)
    # Identity preserved; fields copied in place.
    assert stub.pm.ego_vehicle is stack_ego
