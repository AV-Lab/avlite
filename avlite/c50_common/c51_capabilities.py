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
    MAP_HD = auto() # Whether the strategy provides an HD / OpenDRIVE map
    MAP_RACE_TRACK = auto() # Whether the strategy provides a race-track corridor map
    SLAM = auto() # Whether the strategy provides simultaneous localization and mapping


def is_any_of(req) -> bool:
    """True when *req* is an :class:`AnyOf` (reload-safe; not ``isinstance``)."""
    return type(req).__name__ == "AnyOf" and hasattr(req, "capabilities")


def is_may_use(req) -> bool:
    """True when *req* is a :class:`MayUse` (reload-safe; not ``isinstance``)."""
    return type(req).__name__ == "MayUse" and hasattr(req, "capabilities")


def is_requirement_wrapper(req) -> bool:
    """True when *req* is :class:`AnyOf` or :class:`MayUse` (reload-safe)."""
    return is_any_of(req) or is_may_use(req)


class AnyOf:
    """Requirement satisfied when *at least one* of the listed capabilities is present.

    Usage::

        @property
        def world_requirements(self):
            return {AnyOf(WorldCapability.LIDAR_2D, WorldCapability.LIDAR_3D)}
    """

    def __init__(self, *caps):
        self.capabilities = frozenset(caps)

    def __hash__(self):
        return hash(("AnyOf", self.capabilities))

    def __eq__(self, other):
        return is_any_of(other) and self.capabilities == other.capabilities

    def __repr__(self):
        names = ", ".join(c.name for c in self.capabilities)
        return f"AnyOf({names})"


class MayUse:
    """Soft requirement: never blocks assembly; the module uses these if present.

    Usage::

        @property
        def stack_requirements(self):
            return {StackCapability.LOCALIZATION, MayUse(StackCapability.DETECTION)}
    """

    def __init__(self, *caps):
        self.capabilities = frozenset(caps)

    def __hash__(self):
        return hash(("MayUse", self.capabilities))

    def __eq__(self, other):
        return is_may_use(other) and self.capabilities == other.capabilities

    def __repr__(self):
        names = ", ".join(c.name for c in self.capabilities)
        return f"MayUse({names})"


def required_stack_capabilities(modules) -> set:
    """Flatten hard ``stack_requirements`` from *modules* (``AnyOf`` → members).

    ``MayUse`` entries are ignored — soft deps do not count as consumed for assembly.
    """
    needed: set = set()
    for module in modules:
        if module is None:
            continue
        for req in module.stack_requirements:
            if is_may_use(req):
                continue
            if is_any_of(req):
                needed |= set(req.capabilities)
            else:
                needed.add(req)
    return needed


def used_stack_capabilities(modules) -> set:
    """Flatten hard and soft ``stack_requirements`` from *modules* for UI coloring.

    Includes ``MayUse`` members (soft use) and ``AnyOf`` members. Assembly
    validation still uses :func:`required_stack_capabilities` (hard only).
    """
    used: set = set()
    for module in modules:
        if module is None:
            continue
        for req in module.stack_requirements:
            if is_requirement_wrapper(req):
                used |= set(req.capabilities)
            else:
                used.add(req)
    return used


def satisfies_requirements(requirements: set, capabilities: set) -> bool:
    """Return True when every hard requirement in *requirements* is met.

    - Plain capability entries require an exact match (AND).
    - :class:`AnyOf` requires at least one member present (OR).
    - :class:`MayUse` is always satisfied (soft / optional).
    """
    for req in requirements:
        if is_may_use(req):
            continue
        if is_any_of(req):
            if not (req.capabilities & capabilities):
                return False
        elif req not in capabilities:
            return False
    return True
