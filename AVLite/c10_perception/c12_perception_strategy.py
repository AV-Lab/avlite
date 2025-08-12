from c10_perception.c11_perception_model import PerceptionModel
from c10_perception.c19_settings import PerceptionSettings
import logging
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)

class PerceptionStrategy(ABC):
    """
    Abstract base class for perception strategies.
    This class defines the interface for perception strategies, including methods for detection, tracking, prediction, and perception.
    """
    

    registry = {}
    def __init__(self, perception_model: PerceptionModel, setting:PerceptionSettings = PerceptionSettings):
        self.supports_detection = False
        self.supports_tracking = False    
        self.supports_prediction = False
        self.perception_model = perception_model

        
    
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
    
    
    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:  
            PerceptionStrategy.registry[cls.__name__] = cls

    def reset(self):
        """
        Reset the perception strategy to its initial state.
        """
        pass


