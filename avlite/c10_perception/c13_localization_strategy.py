import logging
from abc import ABC, abstractmethod
from typing import Optional

from avlite.c10_perception.c11_perception_model import PerceptionModel
from avlite.c10_perception.c19_settings import PerceptionSettings, PerceptionSettingsSchema
from avlite.c60_common.c61_capabilities import WorldCapability, LocalizationCapability
from avlite.c60_common.c62_sensor_data import SensorFrame

log = logging.getLogger(__name__)


class LocalizationStrategy(ABC):
    """
    Abstract base class for localization strategies.

    A localization strategy estimates the ego vehicle's pose (and optionally
    velocity) using sensor data such as IMU, LiDAR, or camera images.
    Implementations update ``self.perception_model.ego_vehicle`` in-place so
    that downstream planning and control modules always see the latest pose.

    Subclasses must implement:
        - ``requirements``  – the :class:`WorldCapability` set the bridge must
          provide for this strategy to work.
        - ``capabilities``  – the :class:`LocalizationCapability` set this
          strategy advertises (e.g. GNSS, LIDAR_LOCALIZATION).
        - ``localize(...)`` – the main estimation step.
    """

    registry = {}

    def __init__(self, perception_model: PerceptionModel, setting: PerceptionSettingsSchema = PerceptionSettings):
        self.perception_model = perception_model

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def requirements(self) -> set[WorldCapability]:
        """World capabilities required by this localization strategy."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> set[LocalizationCapability]:
        """Localization capabilities provided by this strategy."""
        pass

    @abstractmethod
    def localize(self, sensors: SensorFrame | None = None) -> None:
        """Run one localization step; update ``self.perception_model.ego_vehicle`` in-place."""
        pass

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def reset(self):
        """Reset any internal state.  Override in subclasses if needed."""
        pass

    # ------------------------------------------------------------------
    # Auto-registration
    # ------------------------------------------------------------------

    def __init_subclass__(cls, abstract=False, **kwargs):
        super().__init_subclass__(**kwargs)
        if not abstract:
            LocalizationStrategy.registry[cls.__name__] = cls
