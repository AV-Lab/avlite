from __future__ import annotations

import logging
from abc import ABC
from typing import ClassVar

from avlite.c10_perception.c11_perception_model import PerceptionModel
from avlite.c10_perception.c19_settings import PerceptionSettings, PerceptionSettingsSchema
from avlite.c50_common.c51_capabilities import StackCapability, StackRequirement, WorldRequirement
from avlite.c50_common.c52_world_sensor_datatypes import SensorFrame

log = logging.getLogger(__name__)


class MappingStrategy(ABC):
    registry = {}

    world_requirements: ClassVar[frozenset[WorldRequirement]] = frozenset()
    stack_requirements: ClassVar[frozenset[StackRequirement]] = frozenset()
    stack_capabilities: ClassVar[frozenset[StackCapability]] = frozenset()

    def __init__(self, setting: PerceptionSettingsSchema = PerceptionSettings):
        self.setting = setting

    def update(
        self,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
        ego=None,
    ) -> None:
        """Run one mapping step. Default is a no-op (static map providers).

        Named ``update`` so it does not collide with the ``map`` attribute
        on ``OccupancyMapper``.
        """

    def reset(self) -> None:
        """Reset any internal mapping state. Default no-op."""

    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:
            MappingStrategy.registry[cls.__name__] = cls
