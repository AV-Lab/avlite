from enum import Enum, auto

class WorldCapability(Enum):
    GT_DETECTION = auto() # Whether the world supports ground truth detection
    GT_TRACKING = auto() # Whether the world supports ground truth tracking ids
    GT_LOCALIZATION = auto() # Whether the world supports ground truth localization
    RGB_IMAGE = auto() # Whether the world supports RGB image
    DEPTH_IMAGE = auto() # Whether the world supports depth image
    LIDAR = auto() # Whether the world supports lidar data

class PerceptionCapability(Enum):
    DETECTION = auto() # Whether the perception strategy supports detection
    TRACKING = auto() # Whether the perception strategy supports tracking
    PREDICTION = auto() # Whether the perception strategy supports prediction
