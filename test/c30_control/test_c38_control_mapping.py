from avlite.c10_perception.c11_perception_model import AgentState, AgentType, EgoState
from avlite.c30_control.c31_control_model import (
    AckermannControlCommand,
    BodyVelocityControlCommand,
    DiffDriveControlCommand,
)
from avlite.c30_control.c38_control_mapping import (
    CONTROL_COMMAND_REGISTRY,
    control_type_for_agent,
)


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


def test_control_command_registry():
    assert set(CONTROL_COMMAND_REGISTRY) == {
        "AckermannControlCommand",
        "DiffDriveControlCommand",
        "BodyVelocityControlCommand",
    }
