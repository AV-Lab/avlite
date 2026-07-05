import logging
from abc import ABC, abstractmethod
from typing import Mapping

from avlite.c10_perception.c11_perception_model import PerceptionModel
from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.c50_common.c51_capabilities import WorldCapability, StackCapability

log = logging.getLogger(__name__)


class MappingStrategy(ABC):
    registry = {}
    
    def __init__(self,  setting:PerceptionSettings = PerceptionSettings):
        self.setting = setting

    @property
    @abstractmethod
    def world_requirements(self) -> set[WorldCapability]:
        pass

    @property
    def stack_requirements(self) -> set[StackCapability]:
        """Upstream stack capabilities this strategy depends on (default: none)."""
        return set()

    @property
    def stack_capabilities(self) -> set[StackCapability]:
        return {StackCapability.MAP}
    

    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:  
            MappingStrategy.registry[cls.__name__] = cls
