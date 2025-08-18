import logging
from abc import ABC, abstractmethod

from avlite.c10_perception.c11_perception_model import PerceptionModel
from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.c60_common.c62_capabilities import WorldCapability, PerceptionCapability

log = logging.getLogger(__name__)

class PerceptionStrategy(ABC):
    """
    Abstract base class for perception strategies.
    This class defines the interface for perception strategies, including methods for detection, tracking, and prediction
    """
    registry = {}
    def __init__(self, perception_model: PerceptionModel, setting:PerceptionSettings = PerceptionSettings):
        self.perception_model = perception_model
    
    @property
    @abstractmethod
    def requirements(self) -> frozenset[WorldCapability]:
        pass

    @property
    @abstractmethod
    def capabilities(self) -> frozenset[PerceptionCapability]:
        pass

    def detect(self, rgb_img=None, depth_img=None, lidar_data=None) -> PerceptionModel:
        """
        Detect objects in the environment using the specified detection method.
        """
        raise NotImplementedError("Detection method not implemented.")

    def track(self) -> PerceptionModel | None:
        """
        Track detected objects over time.
        """
        raise NotImplementedError("Tracking method not implemented.")

    def predict(self)-> PerceptionModel | None:
        """
        Predict future states of tracked objects.
        """
        raise NotImplementedError("Prediction method not implemented.")

    @abstractmethod
    def perceive(self, rgb_img=None, depth_img=None, lidar_data=None, perception_model=None)-> PerceptionModel | None:
        """
        Main perception method that combines detection, tracking, and prediction.
        """
        raise NotImplementedError("Perception method not implemented.")
    
    def reset(self):
        """
        Reset the perception strategy to its initial state.
        """
        pass
    
    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:  
            PerceptionStrategy.registry[cls.__name__] = cls

