"""Unit tests for BasicSim agent spawn with ego global plan."""

import math

from avlite.c10_perception.c11_perception_model import AgentState, EgoState, PerceptionModel
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c40_execution.c46_basic_sim import BasicSim
from avlite.c60_common.c63_trajectory_tracker import TrajectoryTracker


def _straight_global_plan() -> GlobalPlan:
    trajectory = TrajectoryTracker(
        path=[(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)],
        velocity=[5.0, 5.0, 5.0],
    )
    return GlobalPlan(
        start_point=(0.0, 0.0),
        goal_point=(20.0, 0.0),
        path=list(trajectory.path),
        velocity=list(trajectory.velocity),
        trajectory=trajectory,
    )


def test_spawn_agent_uses_provided_global_plan():
    ego = EgoState()
    pm = PerceptionModel(ego_vehicle=ego)
    sim = BasicSim(ego_state=ego, pm=pm)
    plan = _straight_global_plan()

    agent = AgentState(x=1.0, y=0.0, theta=0.0, velocity=0.0)
    sim.spawn_agent(agent, global_plan=plan)

    assert len(pm.agent_vehicles) == 1
    assert agent.agent_id in sim.npc_controllers
    assert agent.velocity == 5.0 * sim.speed_factor
    assert math.isclose(agent.theta, 0.0, abs_tol=1e-6)


def test_spawn_agent_without_global_plan_skips_controller():
    ego = EgoState()
    pm = PerceptionModel(ego_vehicle=ego)
    sim = BasicSim(ego_state=ego, pm=pm)

    agent = AgentState(x=1.0, y=0.0, theta=0.0, velocity=0.0)
    sim.spawn_agent(agent)

    assert len(pm.agent_vehicles) == 1
    assert sim.npc_controllers == {}
