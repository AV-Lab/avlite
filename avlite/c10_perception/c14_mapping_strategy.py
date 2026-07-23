from __future__ import annotations

import logging
from abc import ABC
from typing import ClassVar

from avlite.c10_perception.c11_perception_model import HDMap, Map, RaceMap
from avlite.c10_perception.c19_settings import PerceptionSettings, PerceptionSettingsSchema
from avlite.c50_common.c51_capabilities import StackCapability, StackRequirement, WorldRequirement

log = logging.getLogger(__name__)


class MappingStrategy(ABC):
    registry = {}

    world_requirements: ClassVar[frozenset[WorldRequirement]] = frozenset()
    stack_requirements: ClassVar[frozenset[StackRequirement]] = frozenset()
    stack_capabilities: ClassVar[frozenset[StackCapability]] = frozenset()

    def __init__(self, setting: PerceptionSettingsSchema = PerceptionSettings):
        self.setting = setting

    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:
            MappingStrategy.registry[cls.__name__] = cls


class MapReader(MappingStrategy):
    """Static map provider: holds a pre-loaded Map (no online mapping)."""

    def __init__(self, map: Map, setting: PerceptionSettingsSchema = PerceptionSettings):
        super().__init__(setting=setting)
        self.map = map
        if isinstance(map, HDMap):
            self.stack_capabilities = frozenset({StackCapability.MAP_HD})
        elif isinstance(map, RaceMap):
            self.stack_capabilities = frozenset({StackCapability.MAP_RACE_TRACK})
        else:
            self.stack_capabilities = frozenset()
