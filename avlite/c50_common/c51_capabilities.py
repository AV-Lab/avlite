from __future__ import annotations
from enum import Enum, auto



class WorldCapability(Enum):
    CAMERA_RGB = auto() # Whether the world supports RGB image
    CAMERA_DEPTH = auto() # Whether the world supports depth image
    LIDAR_3D = auto() # Whether the world supports lidar data
    LIDAR_2D = auto()             # 2D LiDAR scanner
    AGENT_SPAWN = auto()          # World supports spawning agent vehicles
    AGENT_CONTROL = auto()        # Bridge can actuate spawned NPC agents via control_agent
    RADAR = auto()                # Radar sensor
    WHEEL_ENCODER = auto()        # Wheel encoder for odometry
    IMU = auto()                  # Inertial measurement unit
    GNSS = auto()                 # GNSS / GPS receiver


class StackCapability(Enum):
    """What a stack module produces for downstream modules.

    Used both as a module's advertised ``capabilities`` and as inter-module
    ``stack_requirements``. A world bridge may also advertise a subset of these
    via ``stack_capabilities`` to provide ground truth (e.g. GT detection).
    """
    DETECTION = auto() # Whether the strategy supports detection
    TRACKING = auto() # Whether the strategy supports tracking
    PREDICTION = auto() # Whether the strategy supports prediction
    LOCAL_PLAN = auto() # Whether the strategy produces a local plan
    GLOBAL_PLAN = auto() # Whether the strategy produces a global plan
    CONTROL = auto() # Whether the strategy produces control commands

    LOCALIZATION = auto() # Whether the strategy provides ego localization
    MAP = auto() # Whether the strategy provides a map
    SLAM = auto() # Whether the strategy provides simultaneous localization and mapping





class AnyOf:
    """Requirement satisfied when the world provides *at least one* of the listed capabilities.

    Usage::

        @property
        def world_requirements(self):
            return {AnyOf(WorldCapability.LIDAR_2D, WorldCapability.LIDAR_3D)}
    """

    def __init__(self, *caps):
        self.capabilities = frozenset(caps)

    def __hash__(self):
        return hash(self.capabilities)

    def __eq__(self, other):
        return isinstance(other, AnyOf) and self.capabilities == other.capabilities

    def __repr__(self):
        names = ", ".join(c.name for c in self.capabilities)
        return f"AnyOf({names})"


def satisfies_requirements(requirements: set, capabilities: set) -> bool:
    """Return True when every requirement in *requirements* is met by *capabilities*.

    Plain :class:`WorldCapability` entries require an exact match (AND semantics).
    :class:`AnyOf` entries require at least one of their members to be present (OR semantics).
    """
    for req in requirements:
        if isinstance(req, AnyOf):
            if not (req.capabilities & capabilities):
                return False
        elif req not in capabilities:
            return False
    return True
