from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import ClassVar

from avlite.c10_perception.c11_perception_model import PerceptionModel
from avlite.c10_perception.c19_settings import PerceptionSettings, PerceptionSettingsSchema
from avlite.c50_common.c51_capabilities import (
    AnyOf,
    MayUse,
    StackCapability,
    StackRequirement,
    WorldCapability,
    WorldRequirement,
)
from avlite.c50_common.c52_world_sensor_datatypes import SensorFrame

log = logging.getLogger(__name__)


class PerceptionStrategy(ABC):
    """Abstract base for perception strategies (monolithic or pipelined).

    The tick entrypoint :meth:`perceive` takes optional ``perception_model`` and
    ``sensors``, supplied by the executer (or UI). Capability attrs declare
    world/stack needs; payloads arrive as these kwargs.
    """
    registry = {}

    world_requirements: ClassVar[frozenset[WorldRequirement]] = frozenset()
    stack_requirements: ClassVar[frozenset[StackRequirement]] = frozenset()
    stack_capabilities: ClassVar[frozenset[StackCapability]] = frozenset()

    def __init__(self, perception_model: PerceptionModel, setting: PerceptionSettingsSchema = PerceptionSettings):
        self.perception_model = perception_model

    @abstractmethod
    def perceive(
        self,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
    ) -> PerceptionModel | None:
        """Run one perception step (detect / track / predict as applicable).

        Args:
            perception_model: Stack world-state snapshot. When provided, becomes
                the authoritative model for this step (also stored on ``self``).
                When omitted, use constructor-held ``self.perception_model``.
            sensors: World sensor snapshot for this tick (``None`` if unused).
                Read fields as needed (e.g. ``sensors.lidar``, ``sensors.rgb``).

        Returns:
            Updated perception model, or ``None`` when not applicable.
        """
        raise NotImplementedError("Perception method not implemented.")
    
    def reset(self):
        """Reset the perception strategy to its initial state."""
        pass
    
    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:  
            PerceptionStrategy.registry[cls.__name__] = cls


class DetectionStrategy(ABC):
    """Detection stage: produce agents/obstacles from sensors into a perception model.

    Entrypoint :meth:`detect` takes optional ``perception_model`` and ``sensors``,
    forwarded by :class:`PerceptionPipeline` (or called directly in tests).
    """
    registry = {}

    world_requirements: ClassVar[frozenset[WorldRequirement]] = frozenset()
    stack_requirements: ClassVar[frozenset[StackRequirement]] = frozenset()
    stack_capabilities: ClassVar[frozenset[StackCapability]] = frozenset({StackCapability.DETECTION})

    @abstractmethod
    def detect(
        self,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
        rgb_img=None,
        depth_img=None,
        lidar_data=None,
    ) -> PerceptionModel:
        """Run one detection step.

        Args:
            perception_model: Stack world-state snapshot to update. Required for
                a useful result; callers (pipeline) always pass it.
            sensors: World sensor snapshot for this tick (``None`` if unused).
            rgb_img: Optional RGB image (convenience; prefer ``sensors.rgb``).
            depth_img: Optional depth image (convenience; prefer ``sensors.depth``).
            lidar_data: Optional LiDAR cloud (convenience; prefer ``sensors.lidar``).

        Returns:
            Updated perception model.
        """
        pass
    
    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:  
            DetectionStrategy.registry[cls.__name__] = cls

    
class TrackingStrategy(ABC):
    """Tracking stage: associate detections over time on a perception model.

    Entrypoint :meth:`track` takes optional ``perception_model`` and ``sensors``.
    """
    registry = {}

    world_requirements: ClassVar[frozenset[WorldRequirement]] = frozenset()
    stack_requirements: ClassVar[frozenset[StackRequirement]] = frozenset()
    stack_capabilities: ClassVar[frozenset[StackCapability]] = frozenset({StackCapability.TRACKING})

    @abstractmethod
    def track(
        self,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
    ) -> PerceptionModel:
        """Run one tracking step.

        Args:
            perception_model: Stack world-state snapshot to update.
            sensors: World sensor snapshot for this tick (``None`` if unused).

        Returns:
            Updated perception model.
        """
        pass

    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:
            TrackingStrategy.registry[cls.__name__] = cls


class PredictionStrategy(ABC):
    """Prediction stage: forecast agent motion on a perception model.

    Entrypoint :meth:`predict` takes optional ``perception_model`` and ``sensors``.
    """
    registry = {}

    world_requirements: ClassVar[frozenset[WorldRequirement]] = frozenset()
    stack_requirements: ClassVar[frozenset[StackRequirement]] = frozenset()
    stack_capabilities: ClassVar[frozenset[StackCapability]] = frozenset({StackCapability.PREDICTION})

    @abstractmethod
    def predict(
        self,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
    ) -> PerceptionModel | None:
        """Run one prediction step.

        Args:
            perception_model: Stack world-state snapshot to update.
            sensors: World sensor snapshot for this tick (``None`` if unused).

        Returns:
            Updated perception model, or ``None`` when not applicable.
        """
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
    def world_requirements(self) -> frozenset[WorldRequirement]:
        reqs: set[WorldRequirement] = set()
        for child in (self._detector, self._tracker, self._predictor):
            if child is not None:
                reqs |= child.world_requirements
        return frozenset(reqs)

    @property
    def stack_requirements(self) -> frozenset[StackRequirement]:
        # Union active children's contracts only (empty stage = no requirement).
        reqs: set[StackRequirement] = set()
        for child in (self._detector, self._tracker, self._predictor):
            if child is not None:
                reqs |= child.stack_requirements
        # Drop MayUse members already covered by hard requirements (e.g. tracker
        # hard DETECTION + predictor MayUse(DETECTION, TRACKING) → soft TRACKING).
        hard = {r for r in reqs if not (AnyOf.matches(r) or MayUse.matches(r))}
        pruned: set[StackRequirement] = set(hard)
        for r in reqs:
            if MayUse.matches(r):
                soft = r.capabilities - hard
                if soft:
                    pruned.add(MayUse(*soft))
            elif AnyOf.matches(r):
                pruned.add(r)
        return frozenset(pruned)

    @property
    def stack_capabilities(self) -> frozenset[StackCapability]:
        # Only active stages advertise caps.
        caps: set[StackCapability] = set()
        if self._detector is not None:
            caps |= self._detector.stack_capabilities
        if self._tracker is not None:
            caps |= self._tracker.stack_capabilities
        if self._predictor is not None:
            caps |= self._predictor.stack_capabilities
        return frozenset(caps)

    def perceive(
        self,
        perception_model: PerceptionModel | None = None,
        sensors: SensorFrame | None = None,
    ) -> PerceptionModel | None:
        """Run detect → track → predict, forwarding ``perception_model`` and ``sensors``."""
        if perception_model is not None:
            self.perception_model = perception_model
        if self._detector is not None:
            self.perception_model = self._detector.detect(
                perception_model=self.perception_model, sensors=sensors,
            )
        if self._tracker is not None:
            self.perception_model = self._tracker.track(
                perception_model=self.perception_model, sensors=sensors,
            )
        if self._predictor is not None:
            self.perception_model = self._predictor.predict(
                perception_model=self.perception_model, sensors=sensors,
            )
        return self.perception_model
