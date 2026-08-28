"""Unit tests for BasicSim agent spawn with ego global plan."""

import math

import numpy as np

from avlite.c10_perception.c11_perception_model import AgentState, EgoState, PerceptionModel, RaceMap
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c40_execution.c46_basic_sim import BasicSim, boundary_segments_from_map, lidar_2d_to_4
from avlite.c50_common.c51_capabilities import StackCapability
from avlite.c50_common.c54_trajectory_tracker import TrajectoryTracker


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

    agent = AgentState(x=1.0, y=0.0, theta=math.pi / 2, velocity=0.0)
    sim.spawn_agent(agent, global_plan=plan)

    assert len(pm.agent_vehicles) == 1
    assert agent.agent_id in sim.npc_controllers
    assert agent.velocity == 5.0 * sim.speed_factor
    assert math.isclose(agent.theta, math.pi / 2, abs_tol=1e-6)


def test_spawn_agent_without_global_plan_skips_controller():
    ego = EgoState()
    pm = PerceptionModel(ego_vehicle=ego)
    sim = BasicSim(ego_state=ego, pm=pm)

    agent = AgentState(x=1.0, y=0.0, theta=0.0, velocity=0.0)
    sim.spawn_agent(agent)

    assert len(pm.agent_vehicles) == 1
    assert sim.npc_controllers == {}


def test_basic_sim_segments_and_no_map_capability_from_race_map():
    left = np.array([[0.0, 1.0], [10.0, 1.0], [20.0, 1.0]])
    right = np.array([[0.0, -1.0], [10.0, -1.0], [20.0, -1.0]])
    race_map = RaceMap(source_path="synthetic", left_bound=left, right_bound=right)
    ego = EgoState()
    sim = BasicSim(ego_state=ego, pm=PerceptionModel(ego_vehicle=ego), map=race_map)

    assert StackCapability.MAP_HD not in sim.stack_capabilities
    assert StackCapability.MAP_RACE_TRACK not in sim.stack_capabilities
    assert sim.boundary_segments.shape[0] == 4  # 2 segs per bound × 2 bounds
    np.testing.assert_array_equal(sim.boundary_segments, boundary_segments_from_map(race_map))


def test_basic_sim_no_map_has_empty_segments():
    ego = EgoState()
    sim = BasicSim(ego_state=ego, pm=PerceptionModel(ego_vehicle=ego), map=None)
    assert StackCapability.MAP_HD not in sim.stack_capabilities
    assert StackCapability.MAP_RACE_TRACK not in sim.stack_capabilities
    assert sim.boundary_segments.shape == (0, 2, 2)


def test_basic_sim_stack_requirements_control_readable_from_class():
    assert BasicSim.stack_requirements == frozenset({StackCapability.CONTROL})
    assert StackCapability.DETECTION in BasicSim.stack_capabilities
    assert StackCapability.TRACKING in BasicSim.stack_capabilities
    assert StackCapability.LOCALIZATION in BasicSim.stack_capabilities


def test_lidar_2d_to_4():
    pts = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    out = lidar_2d_to_4(pts)
    assert out.shape == (2, 4)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out[:, :2], pts.astype(np.float32))
    assert np.all(out[:, 2:] == 0)


def test_lidar_2d_to_4_empty():
    out = lidar_2d_to_4(np.zeros((0, 2)))
    assert out.shape == (0, 4)
