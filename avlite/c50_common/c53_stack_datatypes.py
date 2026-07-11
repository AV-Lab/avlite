"""Stack capability and agent → datatype registries.

Type *definitions* live in their layer modules (c11 / c21 / c31). This module
only indexes them for lookup by :class:`StackCapability` or :class:`AgentType`.
"""

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

STACK_CAPABILITY_DATATYPES: dict[StackCapability, type | tuple[type, ...]] = {
    StackCapability.DETECTION: PerceptionModel,
    StackCapability.TRACKING: PerceptionModel,
    StackCapability.PREDICTION: PredictionModelBase,
    StackCapability.LOCALIZATION: EgoState,
    StackCapability.MAP_HD: HDMap,
    StackCapability.MAP_RACE_TRACK: RaceMap,
    StackCapability.SLAM: (EgoState, Map),
    StackCapability.GLOBAL_PLAN: GlobalPlan,
    StackCapability.LOCAL_PLAN: LocalPlan,
    StackCapability.CONTROL: ControlCommandBase,
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


def datatype_for(cap: StackCapability) -> type | tuple[type, ...]:
    """Return the canonical payload type(s) for *cap*."""
    return STACK_CAPABILITY_DATATYPES[cap]


def capabilities_for(datatype: type) -> frozenset[StackCapability]:
    """Return stack capabilities whose payload type is *datatype* (or includes it)."""
    found: set[StackCapability] = set()
    for cap, mapped in STACK_CAPABILITY_DATATYPES.items():
        if isinstance(mapped, tuple):
            if datatype in mapped:
                found.add(cap)
        elif mapped is datatype:
            found.add(cap)
    return frozenset(found)


def control_type_for_agent(agent: AgentState) -> type[ControlCommandBase]:
    """Map agent platform type to the expected control command class."""
    return DEFAULT_CONTROL_TYPE_BY_AGENT.get(agent.agent_type, AckermannControlCommand)
