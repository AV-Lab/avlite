from __future__ import annotations

from enum import Enum, auto


class WorldCapability(Enum):
    CAMERA_RGB = auto()  # Whether the world supports RGB image
    CAMERA_DEPTH = auto()  # Whether the world supports depth image
    LIDAR_3D = auto()  # Whether the world supports lidar data
    LIDAR_2D = auto()  # 2D LiDAR scanner
    AGENT_SPAWN = auto()  # World supports spawning agent vehicles
    AGENT_CONTROL = auto()  # World can actuate spawned NPC agents via control_agent
    RADAR = auto()  # Radar sensor
    WHEEL_ENCODER = auto()  # Wheel encoder for odometry
    IMU = auto()  # Inertial measurement unit
    GNSS = auto()  # GNSS / GPS receiver


class StackCapability(Enum):
    """What a stack module produces for downstream modules.

    Used both as a module's advertised ``capabilities`` and as inter-module
    ``stack_requirements``. A world bridge may also advertise a subset of these
    via ``stack_capabilities`` to provide ground truth (e.g. GT detection).
    """

    DETECTION = auto()  # Whether the strategy supports detection
    TRACKING = auto()  # Whether the strategy supports tracking
    PREDICTION = auto()  # Whether the strategy supports prediction
    LOCAL_PLAN = auto()  # Whether the strategy produces a local plan
    GLOBAL_PLAN = auto()  # Whether the strategy produces a global plan
    CONTROL = auto()  # Whether the strategy produces control commands

    LOCALIZATION = auto()  # Whether the strategy provides ego localization
    MAP_HD = auto()  # Whether the strategy provides an HD / OpenDRIVE map
    MAP_RACE_TRACK = auto()  # Whether the strategy provides a race-track corridor map
    SLAM = auto()  # Whether the strategy provides simultaneous localization and mapping


class CapabilityGroup:
    """Shared AnyOf / MayUse; identity by class name (importlib-reload safe)."""

    def __init__(self, *caps):
        self.capabilities = frozenset(caps)

    @classmethod
    def matches(cls, obj) -> bool:
        name = type(obj).__name__
        if cls is CapabilityGroup:
            return name in ("AnyOf", "MayUse") and hasattr(obj, "capabilities")
        return name == cls.__name__ and hasattr(obj, "capabilities")

    def __hash__(self):
        return hash((type(self).__name__, self.capabilities))

    def __eq__(self, other):
        return type(self).matches(other) and self.capabilities == other.capabilities

    def __repr__(self):
        names = ", ".join(c.name for c in self.capabilities)
        return f"{type(self).__name__}({names})"


class AnyOf(CapabilityGroup):
    """Requirement satisfied when *at least one* of the listed capabilities is present.

    Usage::

        @property
        def world_requirements(self):
            return {AnyOf(WorldCapability.LIDAR_2D, WorldCapability.LIDAR_3D)}
    """


class MayUse(CapabilityGroup):
    """Soft requirement: never blocks assembly; the module uses these if present.

    Usage::

        @property
        def stack_requirements(self):
            return {StackCapability.LOCALIZATION, MayUse(StackCapability.DETECTION)}
    """


def combine_stack_requirements(modules, *, soft: bool = False) -> set:
    """Union ``stack_requirements`` across *modules*, preserving structure.

    Returns a set that may contain:

    - plain caps (AND) — union of all non-wrapper requirements
    - :class:`AnyOf` — one entry per input ``AnyOf``, members minus AND caps
    - :class:`MayUse` — at most one, union of all ``MayUse`` members minus AND
      caps (omitted when *soft* is False, or when empty after prune)

    Empty wrappers after prune are dropped. Does not merge separate ``AnyOf``
    groups into one (that would weaken OR semantics across modules).
    """
    hard: set = set()
    any_ofs: list[set] = []
    soft_caps: set = set()
    for module in modules:
        if module is None:
            continue
        for req in module.stack_requirements:
            if MayUse.matches(req):
                if soft:
                    soft_caps |= set(req.capabilities)
            elif AnyOf.matches(req):
                any_ofs.append(set(req.capabilities))
            else:
                hard.add(req)

    out: set = set(hard)
    for members in any_ofs:
        pruned = members - hard
        if pruned:
            out.add(AnyOf(*pruned))
    if soft:
        may = soft_caps - hard
        if may:
            out.add(MayUse(*may))
    return out


def satisfies_requirements(requirements: set, capabilities: set) -> bool:
    """Return True when every hard requirement in *requirements* is met.

    - Plain capability entries require an exact match (AND).
    - :class:`AnyOf` requires at least one member present (OR).
    - :class:`MayUse` is always satisfied (soft / optional).
    """
    for req in requirements:
        if MayUse.matches(req):
            continue
        if AnyOf.matches(req):
            if not (req.capabilities & capabilities):
                return False
        elif req not in capabilities:
            return False
    return True
