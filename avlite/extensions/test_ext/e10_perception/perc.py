from avlite.c10_perception.c12_perception_strategy import PerceptionStrategy
from avlite.extensions.test_ext.settings import ExtensionSettings
from avlite.c60_common.c62_capabilities import PerceptionCapability, WorldCapability
import logging

log = logging.getLogger(__name__)

class testClass(PerceptionStrategy):
    def __init__(self, perception_model, setting=None):
        super().__init__(perception_model, setting)
        self.supports_detection = True
        self.supports_tracking = True
        self.supports_prediction = True

    def perceive(self, rgb_img=None, depth_img=None, lidar_data=None, perception_model=None):
        log.warning(f"Perceiving environment...loaded var: {ExtensionSettings.test}")
    
    @property
    def requirements(self) -> frozenset[WorldCapability]:
        return frozenset({
            WorldCapability.GT_DETECTION,
            WorldCapability.GT_LOCALIZATION,
        })
    @property
    def capabilities(self) -> frozenset[PerceptionCapability]:
        return frozenset({
            PerceptionCapability.PREDICTION,
        })
