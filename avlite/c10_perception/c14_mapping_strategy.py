import logging
from abc import ABC, abstractmethod
from typing import Mapping

from avlite.c10_perception.c11_perception_model import PerceptionModel
from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.c60_common.c62_capabilities import WorldCapability, MappingCapability

log = logging.getLogger(__name__)


class MappingStrategy(ABC):
    registry = {}
    
    def __init__(self,  setting:PerceptionSettings = PerceptionSettings):
        self.setting = setting

    @property
    @abstractmethod
    def requirements(self) -> frozenset[WorldCapability]:
        pass

    @property
    @abstractmethod
    def capabilities(self) -> frozenset[MappingCapability]:
        pass
    

    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:  
            MappingStrategy.registry[cls.__name__] = cls
