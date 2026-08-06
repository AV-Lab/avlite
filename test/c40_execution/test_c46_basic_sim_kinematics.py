"""Regression tests for BasicSim bicycle ego and NPC kinematics."""

import math

import numpy as np
import pytest

from avlite.c10_perception.c11_perception_model import AgentState, EgoState, PerceptionModel
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c30_control.c31_control_model import ControlCommand
from avlite.c40_execution.c46_basic_sim import BasicSim
from avlite.c40_execution.c49_settings import ExecutionSettingsSchema
from avlite.c50_common.c54_trajectory_tracker import TrajectoryTracker


def _straight_global_plan(v: float = 5.0) -> GlobalPlan:
    trajectory = TrajectoryTracker(
        path=[(0.0, 0.0), (20.0, 0.0), (40.0, 0.0)],
        velocity=[v, v, v],
    )
    return GlobalPlan(
        start_point=(0.0, 0.0),
        goal_point=(40.0, 0.0),
        path=list(trajectory.path),
        velocity=list(trajectory.velocity),
        trajectory=trajectory,
    )


def test_ego_straight_cruise_uses_pre_update_velocity_for_xy():
    """XY integrates with pre-accel v; yaw uses post-accel v (steer=0 → no yaw)."""
    ego = EgoState(x=0.0, y=0.0, theta=0.0, velocity=10.0)
    setting = ExecutionSettingsSchema(c46_npc_control=False)
    sim = BasicSim(ego_state=ego, pm=PerceptionModel(ego_vehicle=ego), setting=setting)
    dt = 0.01
    accel = 2.0

    sim.control_ego_state(ControlCommand(acceleration=accel, steer=0.0), dt=dt)

    assert ego.x == pytest.approx(10.0 * dt)
    assert ego.y == pytest.approx(0.0)
    assert ego.velocity == pytest.approx(10.0 + accel * dt)
    assert ego.theta == pytest.approx(0.0)


def test_ego_constant_steer_matches_bicycle_yaw_rate():
    ego = EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0)
    setting = ExecutionSettingsSchema(c46_npc_control=False)
    sim = BasicSim(ego_state=ego, pm=PerceptionModel(ego_vehicle=ego), setting=setting)
    dt = 0.01
    steer = 0.2
    L = 2.5  # default when no ego_controller is attached

    sim.control_ego_state(ControlCommand(acceleration=0.0, steer=steer), dt=dt)

    # accel=0 → post-update v equals pre-update v for yaw integration
    assert ego.theta == pytest.approx((5.0 / L) * steer * dt)
    assert ego.x == pytest.approx(5.0 * dt)
    assert ego.y == pytest.approx(0.0)


def test_npc_control_advances_agent_with_steer_slew_limit():
    ego = EgoState(x=0.0, y=0.0, theta=0.0, velocity=0.0)
    pm = PerceptionModel(ego_vehicle=ego)
    setting = ExecutionSettingsSchema(c46_npc_control=True)
    sim = BasicSim(ego_state=ego, pm=pm, setting=setting)
    plan = _straight_global_plan(v=5.0)

    agent = AgentState(x=0.0, y=0.0, theta=0.0, velocity=0.0)
    sim.spawn_agent(agent, global_plan=plan)
    assert agent.agent_id in sim.npc_controllers
    ctrl = sim.npc_controllers[agent.agent_id]
    # Force a large commanded steer so the slew clamp is the observable contract.
    ctrl._npc_steer = 0.0

    class _HugeSteer:
        acceleration = 0.0
        steer = 1.0

    ctrl.control = lambda *args, **kwargs: _HugeSteer()  # type: ignore[method-assign]

    x0, y0, theta0, v0 = agent.x, agent.y, agent.theta, agent.velocity
    dt = 0.01
    sim.control_ego_state(ControlCommand(acceleration=0.0, steer=0.0), dt=dt)

    max_dsteer = 3.0 * dt
    assert ctrl._npc_steer == pytest.approx(max_dsteer)
    assert agent.velocity == pytest.approx(v0)  # accel 0
    # NPC integrates velocity→yaw→xy (post-update v for position).
    expected_theta = theta0 + agent.velocity / ctrl.ego_distance_front_axle * max_dsteer * dt
    assert agent.theta == pytest.approx(expected_theta)
    assert agent.x == pytest.approx(x0 + agent.velocity * math.cos(agent.theta) * dt)
    assert agent.y == pytest.approx(y0 + agent.velocity * math.sin(agent.theta) * dt)
    assert abs(agent.x - x0) + abs(agent.y - y0) + abs(agent.theta - theta0) > 0.0


def test_npc_control_disabled_leaves_agents_frozen():
    ego = EgoState(x=0.0, y=0.0, theta=0.0, velocity=8.0)
    pm = PerceptionModel(ego_vehicle=ego)
    setting = ExecutionSettingsSchema(c46_npc_control=False)
    sim = BasicSim(ego_state=ego, pm=pm, setting=setting)
    plan = _straight_global_plan()

    agent = AgentState(x=1.0, y=0.0, theta=0.0, velocity=5.0)
    sim.spawn_agent(agent, global_plan=plan)
    assert sim.npc_controllers == {}

    before = (agent.x, agent.y, agent.theta, agent.velocity)
    sim.control_ego_state(ControlCommand(acceleration=0.0, steer=0.0), dt=0.01)

    assert (agent.x, agent.y, agent.theta, agent.velocity) == before
    assert ego.x == pytest.approx(8.0 * 0.01)
