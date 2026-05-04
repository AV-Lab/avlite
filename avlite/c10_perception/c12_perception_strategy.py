import logging
from abc import ABC, abstractmethod
from typing import Type 
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
    def __init__(self, perception_model: PerceptionModel, setting:Type[PerceptionSettings] = PerceptionSettings):
        self.perception_model = perception_model
    
    @property
    @abstractmethod
    def requirements(self) -> set[WorldCapability]:
        pass

    @property
    @abstractmethod
    def capabilities(self) -> set[PerceptionCapability]:
        pass

    # def detect(self, rgb_img=None, depth_img=None, lidar_data=None) -> PerceptionModel:
    #     """
    #     Detect objects in the environment using the specified detection method.
    #     """
    #     raise NotImplementedError("Detection method not implemented.")
    #
    # def track(self) -> PerceptionModel | None:
    #     """
    #     Track detected objects over time.
    #     """
    #     raise NotImplementedError("Tracking method not implemented.")
    #
    # def predict(self)-> PerceptionModel | None:
    #     """
    #     Predict future states of tracked objects.
    #     """
    #     raise NotImplementedError("Prediction method not implemented.")

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


class DetectionStrategy(ABC):
    """
    A simple perception strategy that only performs detection.
    """
    registry = {}
    
    @property
    @abstractmethod
    def requirements(self) -> set[WorldCapability]:
        pass

    @abstractmethod
    def detect(self, rgb_img=None, depth_img=None, lidar_data=None) -> PerceptionModel:
        """
        Detect objects in the environment using the specified detection method.
        """
        pass
    
    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:  
            DetectionStrategy.registry[cls.__name__] = cls

    
class TrackingStrategy(ABC):
    """
    A simple perception strategy that only performs tracking.
    """
    registry = {}
    def __init__(self, perception_model: PerceptionModel, setting:Type[PerceptionSettings] = PerceptionSettings):
        self.perception_model = perception_model
    
    @property
    @abstractmethod
    def requirements(self) -> set[WorldCapability]:
        pass


class PerceptionPipeline(PerceptionStrategy):
    """
    A simple pipe-lined perception strategy that performs detection, tracking, and prediction in sequence.
    """
    def __init__(self, perception_model: PerceptionModel, setting:Type[PerceptionSettings] = PerceptionSettings):
        super().__init__(perception_model, setting)
    
    @property
    def requirements(self) -> set[WorldCapability]:
        return set()
    
    @property
    def capabilities(self) -> set[PerceptionCapability]:
        return {PerceptionCapability.DETECTION, PerceptionCapability.TRACKING, PerceptionCapability.PREDICTION}
    
    def perceive(self, rgb_img=None, depth_img=None, lidar_data=None, perception_model=None)-> PerceptionModel | None:
        if perception_model is not None:
            self.perception_model = perception_model
        
        self.detect(rgb_img=rgb_img, depth_img=depth_img, lidar_data=lidar_data)
        self.track()
        self.predict()
        return self.perception_model

