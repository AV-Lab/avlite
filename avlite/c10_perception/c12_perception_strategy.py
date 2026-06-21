import logging
from abc import ABC, abstractmethod
from typing import Type
from avlite.c10_perception.c11_perception_model import PerceptionModel
from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.c60_common.c62_capabilities import WorldCapability, PerceptionCapability
from avlite.c60_common.c67_sensor_data import SensorFrame

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

    @abstractmethod
    def perceive(
        self,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
    ) -> PerceptionModel | None:
        """Main perception method that combines detection, tracking, and prediction."""
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
    def detect(self, perception_model: PerceptionModel, rgb_img=None, depth_img=None, lidar_data=None) -> PerceptionModel:
        """
        Detect objects in the environment using the specified detection method.
        """
        pass
    
    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:  
            DetectionStrategy.registry[cls.__name__] = cls

    
class TrackingStrategy(ABC):
    registry = {}

    @property
    @abstractmethod
    def requirements(self) -> set[WorldCapability]:
        pass

    @abstractmethod
    def track(self, perception_model: PerceptionModel) -> PerceptionModel:
        pass

    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:
            TrackingStrategy.registry[cls.__name__] = cls


class PredictionStrategy(ABC):
    registry = {}

    @property
    @abstractmethod
    def requirements(self) -> set[WorldCapability]:
        pass

    @abstractmethod
    def predict(self, perception_model: PerceptionModel) -> PerceptionModel | None:
        pass

    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:
            PredictionStrategy.registry[cls.__name__] = cls


class PerceptionPipeline(PerceptionStrategy):
    """
    Pipelined perception strategy: detect → track → predict.
    Each stage is resolved by name from its registry at construction time.
    Empty name or unknown name means that stage is skipped.
    """
    def __init__(self, perception_model: PerceptionModel, setting: Type[PerceptionSettings] = PerceptionSettings):
        super().__init__(perception_model, setting)
        self._detector = self._resolve(DetectionStrategy.registry, setting.c12_detection_strategy)
        self._tracker = self._resolve(TrackingStrategy.registry, setting.c12_tracking_strategy)
        self._predictor = self._resolve(PredictionStrategy.registry, setting.c12_prediction_strategy)

    @staticmethod
    def _resolve(registry: dict, name: str):
        if name and name in registry:
            return registry[name]()
        return None

    @property
    def requirements(self) -> set[WorldCapability]:
        reqs = set()
        for child in (self._detector, self._tracker, self._predictor):
            if child is not None:
                reqs |= child.requirements
        # Stages with no strategy fall back to ground truth from the world bridge
        if self._detector is None:
            reqs.add(WorldCapability.GT_DETECTION)
        if self._tracker is None:
            reqs.add(WorldCapability.GT_TRACKING)
        return reqs

    @property
    def capabilities(self) -> set[PerceptionCapability]:
        return {PerceptionCapability.DETECTION, PerceptionCapability.TRACKING, PerceptionCapability.PREDICTION}

    def perceive(
        self,
        perception_model=None,
        sensors: SensorFrame | None = None,
    ) -> PerceptionModel | None:
        if perception_model is not None:
            self.perception_model = perception_model
        if self._detector is not None:
            self.perception_model = self._detector.detect(
                self.perception_model,
                rgb_img=sensors.rgb if sensors else None,
                depth_img=sensors.depth if sensors else None,
                lidar_data=sensors.lidar if sensors else None,
            )
        if self._tracker is not None:
            self.perception_model = self._tracker.track(self.perception_model)
        if self._predictor is not None:
            self.perception_model = self._predictor.predict(self.perception_model)
        return self.perception_model
