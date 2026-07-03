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
from avlite.c60_common.c61_capabilities import WorldCapability


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


@dataclass
class _StubBridge(WorldBridge, abstract=True):
    ego_state: EgoState = field(default_factory=EgoState)
    last_cmd: Optional[ControlCommand] = None
    last_teleport: Optional[tuple[float, float, Optional[float]]] = None

    @property
    def capabilities(self) -> set[WorldCapability]:
        return set()

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
    bridge.teleport_agent(EGO_AGENT_ID, 3.0, 4.0, theta=0.5)
    assert bridge.last_teleport == (3.0, 4.0, 0.5)


def test_teleport_agent_raises_for_npc():
    bridge = _StubBridge()
    with pytest.raises(NotImplementedError):
        bridge.teleport_agent(1, 1.0, 2.0)


def test_get_lidar_data_raises_for_npc():
    bridge = _StubBridge()
    with pytest.raises(NotImplementedError, match="lidar for agent 1"):
        bridge.get_lidar_data(agent_id=1)


def test_control_type_default_ackermann():
    from avlite.c40_execution.c46_basic_sim import BasicSim

    pm = PerceptionModel()
    sim = BasicSim(ego_state=EgoState(), pm=pm)
    assert sim.control_type(EgoState()) is AckermannControlCommand
