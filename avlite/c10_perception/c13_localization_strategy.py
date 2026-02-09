import logging
from abc import ABC, abstractmethod
from typing import Type, Optional

import numpy as np

from avlite.c10_perception.c11_perception_model import PerceptionModel
from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.c60_common.c62_capabilities import WorldCapability, LocalizationCapability

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

    def __init__(self, perception_model: PerceptionModel, setting: Type[PerceptionSettings] = PerceptionSettings):
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
    def localize(
        self,
        imu: Optional[np.ndarray] = None,
        lidar: Optional[np.ndarray] = None,
        rgb_img: Optional[np.ndarray] = None,
    ) -> None:
        """
        Run one localization step.

        The method must update ``self.perception_model.ego_vehicle`` in-place
        (x, y, theta, velocity, …).  It does **not** return a value.

        Parameters
        ----------
        imu : np.ndarray, optional
            IMU measurement array.
        lidar : np.ndarray, optional
            LiDAR point cloud (N×3 or N×4).
        rgb_img : np.ndarray, optional
            RGB camera image (H×W×3).
        """
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
