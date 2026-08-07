"""Sensors→control must run when no local planner is assembled."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest

from avlite.c10_perception.c11_perception_model import EGO_AGENT_ID, EgoState, PerceptionModel
from avlite.c30_control.c31_control_model import AckermannControlCommand
from avlite.c30_control.c35_pure_pursuit import FollowTheGapController
from avlite.c30_control.c39_settings import ControlSettings
from avlite.c40_execution.c41_world_bridge import WorldBridge
from avlite.c40_execution.c44_sync_executer import SyncExecuter
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c50_common.c51_capabilities import StackCapability
from avlite.c50_common.c52_world_sensor_datatypes import SensorFrame
from avlite.c60_apps.c62_factory import executor_factory


@dataclass
class _PlantWorld(WorldBridge):
    """Minimal plant that records applied commands and serves LiDAR."""

    ego_state: EgoState = field(
        default_factory=lambda: EgoState(x=0.0, y=0.0, theta=0.0, velocity=0.0)
    )
    perception_model: Optional[PerceptionModel] = None
    world_capabilities = frozenset()
    stack_capabilities = frozenset({StackCapability.LOCALIZATION})
    applied: list = field(default_factory=list)

    def control_ego_state(self, cmd, dt: Optional[float] = 0.01):
        self.applied.append(cmd)
        # Crude integrate so factory smoke can observe motion.
        self.ego_state.velocity = max(0.0, self.ego_state.velocity + float(cmd.acceleration) * float(dt or 0.01))
        self.ego_state.x += self.ego_state.velocity * float(dt or 0.01)

    def get_sensor_frame(self, agent_id: int = EGO_AGENT_ID) -> SensorFrame:
        # Forward cone of returns so Follow-the-Gap can pick a gap.
        angles = np.linspace(-0.8, 0.8, 40)
        r = 12.0
        pts = np.column_stack(
            [
                self.ego_state.x + r * np.cos(self.ego_state.theta + angles),
                self.ego_state.y + r * np.sin(self.ego_state.theta + angles),
                np.zeros_like(angles),
                np.zeros_like(angles),
            ]
        ).astype(np.float32)
        return SensorFrame(lidar=pts)


def test_sync_control_runs_without_local_planner():
    """Hard gate on local_planner previously skipped sensors→control entirely."""
    seen: dict = {}

    def control(ego, plan=None, control_dt=None, perception_model=None, sensors=None):
        seen["plan"] = plan
        seen["sensors"] = sensors
        return AckermannControlCommand(steer=0.1, acceleration=1.0)

    world = _PlantWorld()
    exec_ = SyncExecuter(
        perception_model=PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0)),
        world=world,
        perception=None,
        localization=None,
        global_planner=None,
        local_planner=None,
        controller=SimpleNamespace(
            world_requirements=frozenset(),
            stack_requirements=frozenset({StackCapability.LOCALIZATION}),
            stack_capabilities=frozenset({StackCapability.CONTROL}),
            control=control,
            reset=lambda: None,
        ),
    )

    exec_.step(sim_dt=0.05, control_dt=0.0, pace_control=True, pace_sim=True)

    assert "sensors" in seen
    assert seen["plan"] is None
    assert exec_._last_cmd is not None
    assert exec_._last_cmd.acceleration == pytest.approx(1.0)
    assert len(world.applied) == 1


def test_follow_the_gap_factory_stack_actuates_without_local_planner():
    """Documented sensors→control composition must move the plant."""
    # Keep world GT localization so _can_actuate passes with no localization module.
    prev_caps = ExecutionSettings.c41_world_stack_capabilities
    prev_cruise = ControlSettings.c35_cruise_velocity
    try:
        ExecutionSettings.c41_world_stack_capabilities = None  # allow defaults / all
        ControlSettings.c35_cruise_velocity = 5.0
        ex = executor_factory(
            local_planner_strategy_name="",
            controller_strategy_name="FollowTheGapController",
            global_planner_strategy_name="",
            perception_strategy_name="",
            localization_strategy_name="",
            load_plugins=False,
        )
        assert ex.local_planner is None
        assert isinstance(ex.controller, FollowTheGapController)

        x0 = ex.world.get_ego_state().x
        for _ in range(25):
            ex.step(sim_dt=0.05, control_dt=0.0, replan_dt=1e9, perception_dt=1e9)

        assert ex._last_cmd is not None
        assert ex._last_cmd.acceleration != 0.0 or ex._last_cmd.steer != 0.0
        assert ex.world.get_ego_state().x != pytest.approx(x0, abs=1e-6)
    finally:
        ExecutionSettings.c41_world_stack_capabilities = prev_caps
        ControlSettings.c35_cruise_velocity = prev_cruise
