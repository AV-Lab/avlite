from enum import Enum, auto

class WorldCapability(Enum):
    GT_DETECTION = auto() # Whether the world supports ground truth detection
    GT_TRACKING = auto() # Whether the world supports ground truth tracking ids
    GT_LOCALIZATION = auto() # Whether the world supports ground truth localization
    CAMERA_RGB = auto() # Whether the world supports RGB image
    CAMERA_DEPTH = auto() # Whether the world supports depth image
    LIDAR_3D = auto() # Whether the world supports lidar data
    LIDAR_2D = auto()             # 2D LiDAR scanner
    RADAR = auto()                # Radar sensor
    WHEEL_ENCODER = auto()        # Wheel encoder for odometry
    IMU = auto()                  # Inertial measurement unit
    GNSS = auto()                 # GNSS / GPS receiver

class PerceptionCapability(Enum):
    DETECTION = auto() # Whether the perception strategy supports detection
    TRACKING = auto() # Whether the perception strategy supports tracking
    PREDICTION = auto() # Whether the perception strategy supports prediction

class LocalizationCapability(Enum):
    LOCALIZATION_2D = auto() # Whether the localization strategy provides 2D pose (x, y)
    LOCALIZATION_3D = auto() # Whether the localization strategy provides 3D pose (x, y, z)
    LOCALIZATION_HEADING = auto() # Whether the localization strategy provides heading estimation
    LOCALIZATION_HEADING_3D = auto() # Whether the localization strategy provides 3D heading estimation (e.g. roll, pitch, yaw)
    VELOCITY = auto() # Whether the localization strategy provides velocity estimation

class MappingCapability(Enum):
    OCCUPANCY_GRID = auto()
    PATH_BOUNDARY = auto()
    OPENDRIVE_HDMAP = auto()




