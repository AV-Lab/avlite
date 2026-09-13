from dataclasses import dataclass, field
from typing import Optional

import pytest

from avlite.c10_perception.c11_perception_model import (
    AgentState,
    AgentType,
    EGO_AGENT_ID,
    EgoState,
    PerceptionModel,
)
from avlite.c30_control.c31_control_model import AckermannControlCommand, ControlCommand
from avlite.c40_execution.c41_world_bridge import WorldBridge


def test_ego_agent_id_default():
    assert EgoState().agent_id == EGO_AGENT_ID == 0


def test_add_agent_starts_at_one():
    pm = PerceptionModel()
    first = AgentState(x=1.0, y=2.0)
    second = AgentState(x=3.0, y=4.0)
    assert pm.add_agent_vehicle(first) == 1
    assert pm.add_agent_vehicle(second) == 2


def test_add_agent_skips_zero():
    pm = PerceptionModel()
    pm.add_agent_vehicle(AgentState())
    assert all(a.agent_id >= 1 for a in pm.agent_vehicles)


def test_add_agent_evicts_oldest_when_cap_reached():
    """Over-cap adds must not wipe the whole detection set (FastBEV / spawn)."""
    pm = PerceptionModel()
    pm.max_agent_vehicles = 3
    kept_xy = []
    for i in range(5):
        pm.add_agent_vehicle(AgentState(x=float(i), y=0.0))
        kept_xy.append((float(i), 0.0))
        assert 1 <= len(pm.agent_vehicles) <= 3

    assert len(pm.agent_vehicles) == 3
    xs = [a.x for a in pm.agent_vehicles]
    assert xs == [2.0, 3.0, 4.0]
    assert all(a.agent_id >= 1 for a in pm.agent_vehicles)


def test_add_agent_refuses_non_positive_cap():
    pm = PerceptionModel()
    pm.max_agent_vehicles = 0
    assert pm.add_agent_vehicle(AgentState(x=1.0)) == -1
    assert pm.agent_vehicles == []


@dataclass
class _StubBridge(WorldBridge, abstract=True):
    ego_state: EgoState = field(default_factory=EgoState)
    last_cmd: Optional[ControlCommand] = None
    last_teleport: Optional[tuple[float, float, Optional[float]]] = None

    world_capabilities = frozenset()
    stack_capabilities = frozenset()

    def control_ego_state(self, cmd, dt=0.01):
        self.last_cmd = cmd

    def teleport_ego(self, x, y, theta=None):
        self.last_teleport = (x, y, theta)


def test_ego_agent_type_default():
    assert EgoState().agent_type == AgentType.ACKERMANN


def test_control_agent_delegates_to_control_ego():
    bridge = _StubBridge()
    cmd = ControlCommand(steer=0.2, acceleration=1.0)
    bridge.control_agent(EGO_AGENT_ID, cmd, dt=0.05)
    assert bridge.last_cmd is cmd


def test_control_agent_raises_for_npc():
    bridge = _StubBridge()
    with pytest.raises(NotImplementedError):
        bridge.control_agent(1, ControlCommand(), dt=0.01)


def test_teleport_agent_delegates_to_teleport_ego():
    bridge = _StubBridge()
    bridge.teleport_agent(AgentState(agent_id=EGO_AGENT_ID, x=3.0, y=4.0, theta=0.5))
    assert bridge.last_teleport == (3.0, 4.0, 0.5)


def test_teleport_agent_raises_for_npc():
    bridge = _StubBridge()
    with pytest.raises(NotImplementedError):
        bridge.teleport_agent(AgentState(agent_id=1, x=1.0, y=2.0))


def test_get_lidar_data_raises_for_npc():
    bridge = _StubBridge()
    with pytest.raises(NotImplementedError, match="lidar for agent 1"):
        bridge.get_lidar_data(agent_id=1)


