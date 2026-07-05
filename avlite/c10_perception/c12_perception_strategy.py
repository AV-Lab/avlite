import logging
from abc import ABC, abstractmethod
from avlite.c10_perception.c11_perception_model import PerceptionModel
from avlite.c10_perception.c19_settings import PerceptionSettings, PerceptionSettingsSchema
from avlite.c50_common.c51_capabilities import WorldCapability, StackCapability
from avlite.c50_common.c52_sensor_datatypes import SensorFrame

log = logging.getLogger(__name__)


class PerceptionStrategy(ABC):
    """
    Abstract base class for perception strategies.
    This class defines the interface for perception strategies, including methods for detection, tracking, and prediction
    """
    registry = {}
    def __init__(self, perception_model: PerceptionModel, setting: PerceptionSettingsSchema = PerceptionSettings):
        self.perception_model = perception_model
    
    @property
    @abstractmethod
    def world_requirements(self) -> set[WorldCapability]:
        """World (sensor) capabilities this strategy requires from the bridge."""
        pass

    @property
    def stack_requirements(self) -> set[StackCapability]:
        """Upstream stack capabilities this strategy depends on (default: none)."""
        return set()

    @property
    @abstractmethod
    def stack_capabilities(self) -> set[StackCapability]:
        """Stack capabilities this strategy provides to downstream modules."""
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
    def world_requirements(self) -> set[WorldCapability]:
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
    def world_requirements(self) -> set[WorldCapability]:
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
    def world_requirements(self) -> set[WorldCapability]:
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
    Empty name means that stage is skipped (ground truth for detect/track; no prediction).
    """
    def __init__(self, perception_model: PerceptionModel, setting: PerceptionSettingsSchema = PerceptionSettings):
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
    def world_requirements(self) -> set[WorldCapability]:
        reqs = set()
        for child in (self._detector, self._tracker, self._predictor):
            if child is not None:
                reqs |= child.world_requirements
        return reqs

    @property
    def stack_requirements(self) -> set[StackCapability]:
        # Stages with no strategy fall back to ground truth from the world bridge
        reqs = set()
        if self._detector is None:
            reqs.add(StackCapability.DETECTION)
        if self._tracker is None:
            reqs.add(StackCapability.TRACKING)
        return reqs

    @property
    def stack_capabilities(self) -> set[StackCapability]:
        return {StackCapability.DETECTION, StackCapability.TRACKING, StackCapability.PREDICTION}

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
