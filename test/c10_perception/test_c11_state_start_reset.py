"""Regression tests for State.set_start / reset (profile start pose)."""

import math

from avlite.c10_perception.c11_perception_model import AgentState, AgentType, EgoState, State


def test_state_reset_restores_construction_pose():
    state = State(x=1.0, y=2.0, theta=0.5)
    state.x, state.y, state.theta = 9.0, 8.0, -1.0
    state.reset()
    assert state.x == 1.0
    assert state.y == 2.0
    assert state.theta == 0.5


def test_set_start_updates_reset_snapshot():
    state = State(x=0.0, y=0.0, theta=0.0)
    state.x, state.y, state.theta = 3.0, 4.0, math.pi / 4
    state.set_start()
    state.x, state.y, state.theta = 0.0, 0.0, 0.0
    state.reset()
    assert state.x == 3.0
    assert state.y == 4.0
    assert state.theta == math.pi / 4


def test_agent_reset_restores_velocity_and_type():
    """Polymorphic copy_from must restore AgentState fields, not only x/y/theta."""
    agent = AgentState(x=1.0, y=2.0, theta=0.3, velocity=5.0, agent_type=AgentType.DIFF_DRIVE)
    agent.x, agent.y, agent.theta = 10.0, 20.0, 1.0
    agent.velocity = 0.0
    agent.agent_type = AgentType.PEDESTRIAN
    agent.reset()
    assert agent.x == 1.0
    assert agent.y == 2.0
    assert agent.theta == 0.3
    assert agent.velocity == 5.0
    assert agent.agent_type == AgentType.DIFF_DRIVE


def test_set_start_then_reset_preserves_agent_identity():
    ego = EgoState(x=0.0, y=0.0, velocity=1.0)
    original_id = id(ego)
    ego.x, ego.y, ego.velocity = 7.0, 8.0, 3.0
    ego.set_start()
    ego.x, ego.y, ego.velocity = 0.0, 0.0, 0.0
    ego.reset()
    assert id(ego) == original_id
    assert ego.x == 7.0
    assert ego.y == 8.0
    assert ego.velocity == 3.0
