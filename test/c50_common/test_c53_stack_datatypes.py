"""Tests for stack capability / agent datatype registries in c53_stack_datatypes."""

from __future__ import annotations

from avlite.c10_perception.c11_perception_model import (
    AgentState,
    AgentType,
    EgoState,
    HDMap,
    Map,
    PerceptionModel,
    PredictionModelBase,
    RaceMap,
)
from avlite.c20_planning.c21_planning_model import GlobalPlan, LocalPlan
from avlite.c30_control.c31_control_model import (
    AckermannControlCommand,
    BodyVelocityControlCommand,
    ControlCommandBase,
    DiffDriveControlCommand,
)
from avlite.c50_common.c51_capabilities import StackCapability
from avlite.c50_common.c53_stack_datatypes import (
    DEFAULT_CONTROL_TYPE_BY_AGENT,
    STACK_CAPABILITY_DATATYPES,
    capabilities_for,
    control_type_for_agent,
    datatype_for,
)


def test_every_stack_capability_has_datatype():
    assert set(STACK_CAPABILITY_DATATYPES) == set(StackCapability)


def test_datatype_for_known_mappings():
    assert datatype_for(StackCapability.DETECTION) is PerceptionModel
    assert datatype_for(StackCapability.TRACKING) is PerceptionModel
    assert datatype_for(StackCapability.PREDICTION) is PredictionModelBase
    assert datatype_for(StackCapability.LOCALIZATION) is EgoState
    assert datatype_for(StackCapability.MAP_HD) is HDMap
    assert datatype_for(StackCapability.MAP_RACE_TRACK) is RaceMap
    assert datatype_for(StackCapability.SLAM) == (EgoState, Map)
    assert datatype_for(StackCapability.GLOBAL_PLAN) is GlobalPlan
    assert datatype_for(StackCapability.LOCAL_PLAN) is LocalPlan
    assert datatype_for(StackCapability.CONTROL) is ControlCommandBase


def test_capabilities_for_reverse_lookup():
    assert capabilities_for(PerceptionModel) == frozenset(
        {StackCapability.DETECTION, StackCapability.TRACKING}
    )
    assert capabilities_for(EgoState) == frozenset(
        {StackCapability.LOCALIZATION, StackCapability.SLAM}
    )
    assert capabilities_for(Map) == frozenset({StackCapability.SLAM})
    assert capabilities_for(HDMap) == frozenset({StackCapability.MAP_HD})
    assert capabilities_for(RaceMap) == frozenset({StackCapability.MAP_RACE_TRACK})
    assert capabilities_for(GlobalPlan) == frozenset({StackCapability.GLOBAL_PLAN})
    assert capabilities_for(ControlCommandBase) == frozenset({StackCapability.CONTROL})


def test_control_type_for_agent_ackermann():
    assert control_type_for_agent(EgoState()) is AckermannControlCommand


def test_control_type_for_agent_diff_drive():
    agent = AgentState(agent_type=AgentType.DIFF_DRIVE)
    assert control_type_for_agent(agent) is DiffDriveControlCommand


def test_control_type_for_agent_aerial():
    agent = AgentState(agent_type=AgentType.AERIAL)
    assert control_type_for_agent(agent) is BodyVelocityControlCommand


def test_control_type_for_agent_surface_vessel():
    agent = AgentState(agent_type=AgentType.SURFACE_VESSEL)
    assert control_type_for_agent(agent) is BodyVelocityControlCommand


def test_control_type_for_agent_underwater():
    agent = AgentState(agent_type=AgentType.UNDERWATER)
    assert control_type_for_agent(agent) is BodyVelocityControlCommand


def test_control_type_for_agent_cyclist():
    agent = AgentState(agent_type=AgentType.CYCLIST)
    assert control_type_for_agent(agent) is DiffDriveControlCommand


def test_control_type_for_agent_pedestrian():
    agent = AgentState(agent_type=AgentType.PEDESTRIAN)
    assert control_type_for_agent(agent) is BodyVelocityControlCommand


def test_control_type_for_agent_dynamic_object():
    agent = AgentState(agent_type=AgentType.DYNAMIC_OBJECT)
    assert control_type_for_agent(agent) is BodyVelocityControlCommand


def test_default_control_type_covers_all_agent_types():
    assert set(DEFAULT_CONTROL_TYPE_BY_AGENT) == set(AgentType)
