from __future__ import annotations

from avlite.c10_perception.c11_perception_model import AgentState, AgentType
from avlite.c30_control.c31_control_model import (
    AckermannControlCommand,
    BodyVelocityControlCommand,
    ControlCommandBase,
    DiffDriveControlCommand,
)

CONTROL_COMMAND_REGISTRY: dict[str, type[ControlCommandBase]] = {
    "AckermannControlCommand": AckermannControlCommand,
    "DiffDriveControlCommand": DiffDriveControlCommand,
    "BodyVelocityControlCommand": BodyVelocityControlCommand,
}

DEFAULT_CONTROL_TYPE_BY_AGENT: dict[AgentType, type[ControlCommandBase]] = {
    AgentType.ACKERMANN: AckermannControlCommand,
    AgentType.DIFF_DRIVE: DiffDriveControlCommand,
    AgentType.AERIAL: BodyVelocityControlCommand,
    AgentType.SURFACE_VESSEL: BodyVelocityControlCommand,
    AgentType.UNDERWATER: BodyVelocityControlCommand,
    AgentType.CYCLIST: DiffDriveControlCommand,
    AgentType.PEDESTRIAN: BodyVelocityControlCommand,
    AgentType.DYNAMIC_OBJECT: BodyVelocityControlCommand,
}


def control_type_for_agent(agent: AgentState) -> type[ControlCommandBase]:
    """Map agent platform type to the expected control command class."""
    return DEFAULT_CONTROL_TYPE_BY_AGENT.get(agent.agent_type, AckermannControlCommand)
