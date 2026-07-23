from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import ClassVar

from avlite.c10_perception.c11_perception_model import PerceptionModel
from avlite.c10_perception.c19_settings import PerceptionSettings, PerceptionSettingsSchema
from avlite.c50_common.c51_capabilities import StackCapability, StackRequirement, WorldRequirement
from avlite.c50_common.c52_world_sensor_datatypes import SensorFrame

log = logging.getLogger(__name__)


class LocalizationStrategy(ABC):
    """Abstract base for localization strategies.

    Estimates the ego pose (and optionally velocity) from sensors. The tick
    entrypoint :meth:`localize` takes optional ``perception_model`` and
    ``sensors``, supplied by the executer. Implementations update
    ``perception_model.ego_vehicle`` in-place so downstream modules see the
    latest pose.

    Capability attrs default to empty world/stack requirements and
    ``LOCALIZATION`` stack capability; override as class attributes when needed.
    """

    registry = {}

    world_requirements: ClassVar[frozenset[WorldRequirement]] = frozenset()
    stack_requirements: ClassVar[frozenset[StackRequirement]] = frozenset()
    stack_capabilities: ClassVar[frozenset[StackCapability]] = frozenset({StackCapability.LOCALIZATION})

    def __init__(self, perception_model: PerceptionModel, setting: PerceptionSettingsSchema = PerceptionSettings):
        self.perception_model = perception_model

    @abstractmethod
    def localize(
        self,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
    ) -> None:
        """Run one localization step; update ego pose in-place.

        Args:
            perception_model: Stack world-state snapshot. When provided, becomes
                the authoritative model for this step (also stored on ``self``).
                When omitted, use constructor-held ``self.perception_model``.
            sensors: World sensor snapshot for this tick (``None`` if unused).
                Read fields as needed (e.g. ``sensors.lidar``, ``sensors.imu``).
        """
        pass

    def reset(self):
        """Reset any internal state.  Override in subclasses if needed."""
        pass

    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:
            LocalizationStrategy.registry[cls.__name__] = cls
