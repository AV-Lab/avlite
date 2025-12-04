import logging
from abc import ABC, abstractmethod

from avlite.c10_perception.c11_perception_model import perceptionmodel
from avlite.c10_perception.c19_settings import perceptionsettings

from avlite.c60_common.c62_capabilities import WorldCapability

log = logging.getLogger(__name__)

class LocalizationStrategy(ABC):

    registry = {}
    def __init__(self, perception_model: PerceptionModel, setting:PerceptionSettings = PerceptionSettings):
        self.perception_model = perception_model
    
    @property
    @abstractmethod
    def requirements(self) -> frozenset[WorldCapability]:
        pass

    
    @abstractmethod
    def localize(self, imu = None, lidar = None, rgb_img = None):
        pass
    

    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:  
            LocalizationStrategy.registry[cls.__name__] = cls
