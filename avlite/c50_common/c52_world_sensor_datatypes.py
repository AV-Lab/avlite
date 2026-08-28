"""AVLite canonical sensor formats.

All WorldBridge implementations must populate SensorFrame using these exact
layouts. Convert simulator/ROS messages in the bridge; do not pass raw
message layouts to perception or localization.

Canonical formats
-----------------
rgb            (H, W, 3) uint8, row-major RGB
depth          (H, W) float32, metres
camera_params  CameraParams — intrinsic K + world-to-camera extrinsic for rgb/depth
lidar          (N, 4) float32, [x, y, z, intensity] world frame
imu            ImuReading — linear accel + angular velocity, sensor frame
gnss           GnssReading — WGS84 lat/lon/alt + optional map x/y/z
wheel_odometry WheelOdometry — linear_velocity m/s + yaw_rate rad/s
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from avlite.c50_common.c51_capabilities import WorldCapability

# Semantic ndarray aliases — layout defined in module docstring above.
RgbImage = np.ndarray  # (H, W, 3) uint8 RGB
DepthImage = np.ndarray  # (H, W) float32 metres
LidarCloud = np.ndarray  # (N, 4) float32 [x, y, z, intensity]



@dataclass
class ImuReading:
    """Inertial measurement at a single timestep."""

    linear_accel: tuple[float, float, float]  # (ax, ay, az) m/s², sensor frame
    angular_velocity: tuple[float, float, float]  # (gx, gy, gz) rad/s, sensor frame


class GnssDatum(Enum):
    """Geodetic datum for GNSS latitude/longitude/altitude."""

    WGS84 = "WGS84"


@dataclass
class GnssReading:
    """GNSS fix: raw geodetic measurement plus optional map-frame position.

    Geodetic fields record what the receiver reports. Map fields record the
    same fix expressed in the AVLite map frame (same coordinates as EgoState.x/y/z).

    Population rules:
      - ROS NavSatFix bridge: always set latitude/longitude/altitude/datum.
        Set map_x/y/z when HDMap geoReference is available; else leave map_* None
        and let localization convert via HDMap.geoReference.
      - Sim bridges without GNSS: leave SensorFrame.gnss as None.
    """

    # Geodetic fix from the GNSS receiver (WGS84).
    latitude: float  # degrees, north-positive
    longitude: float  # degrees, east-positive
    altitude: float  # metres above the WGS84 ellipsoid
    datum: GnssDatum = GnssDatum.WGS84

    # Position in the AVLite map frame (OpenDRIVE local coordinates).
    # Same frame as EgoState.x, EgoState.y, EgoState.z.
    # None when the bridge has not converted yet — localization fills these
    # using HDMap.geoReference (proj string, datum=WGS84 in OpenDRIVE files).
    map_x: float | None = None
    map_y: float | None = None
    map_z: float | None = None


@dataclass
class WheelOdometry:
    """Ego motion derived from wheel encoders."""

    linear_velocity: float  # forward speed along ego x-axis, m/s (+ = forward)
    yaw_rate: float  # heading change rate, rad/s (+ = counter-clockwise)


@dataclass
class CameraParams:
    """Pinhole geometry for one camera image.

    Frame convention: ``world_to_camera`` maps homogeneous world (map) points
    into the OpenCV optical camera frame — x right, y down, z forward along the
    optical axis, z > 0 in front of the camera::

        p_cam = world_to_camera @ [x_world, y_world, z_world, 1]
        u = fx * X / Z + cx,  v = fy * Y / Z + cy

    World coordinates are the same frame as EgoState.x/y/z and
    SensorFrame.lidar, so projecting a LiDAR cloud into the image needs no
    other transform. ``world_to_camera`` is the pose at capture time and
    therefore changes every frame as the ego moves.

    Self-contained per camera: an instance carries everything needed to project
    into its own image, so extra cameras in ``SensorFrame.additional_frames``
    each carry their own ``CameraParams``.
    """

    intrinsic: np.ndarray  # (3, 3) float64 K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    world_to_camera: np.ndarray  # (4, 4) float64 world → camera optical frame
    width: int  # pixels; resolution the intrinsic is valid for
    height: int  # pixels; resolution the intrinsic is valid for

    def __post_init__(self) -> None:
        self.intrinsic = np.asarray(self.intrinsic, dtype=np.float64)
        self.world_to_camera = np.asarray(self.world_to_camera, dtype=np.float64)
        if self.intrinsic.shape != (3, 3):
            raise ValueError(f"expected (3, 3) intrinsic, got shape {self.intrinsic.shape}")
        if self.world_to_camera.shape != (4, 4):
            raise ValueError(
                f"expected (4, 4) world_to_camera, got shape {self.world_to_camera.shape}"
            )


@dataclass
class SensorFrame:
    """Snapshot of all sensor readings for one execution tick.

    Any field may be None when the bridge does not provide that sensor or when
    gated off by the ExecutionSettings.c41_world_capabilities filter.
    """

    # Camera: colour image from the primary camera.
    # Shape (H, W, 3), dtype uint8, channels in RGB order (not BGR).
    # H and W vary by camera; algorithms must not assume fixed resolution.
    rgb: RgbImage | None = None

    # Camera: per-pixel distance from the primary camera's image plane.
    # Shape (H, W), dtype float32, values in metres.
    # Must match rgb height/width when both are present.
    depth: DepthImage | None = None

    # Geometry of the primary camera, i.e. the one that produced rgb/depth:
    # intrinsic K + world-to-camera extrinsic at capture time. None when the
    # bridge exposes no camera. Required to project world-frame lidar points
    # into the image.
    camera_params: CameraParams | None = None

    # LiDAR: point cloud in the world (map) frame.
    # Shape (N, 4), dtype float32, columns [x, y, z, intensity].
    # x, y, z in metres; intensity is sensor-specific reflectance (0+).
    # N varies per scan. 2D scanners: set z=0 and intensity=0 in the bridge.
    lidar: LidarCloud | None = None

    imu: ImuReading | None = None
    gnss: GnssReading | None = None
    wheel_odometry: WheelOdometry | None = None

    stamp: float | None = None  # acquisition time, seconds (sim or wall clock)
    frame_id: str | None = None  # coordinate frame name, e.g. "map" or "base_link"

    # Extra named units on this tick (lidars, IMUs, cameras, mixed rigs).
    # Keys are stable sensor names ("lidar_top", "front", …). Each value is a
    # leaf SensorFrame with only that unit's channels set and additional_frames
    # left None (no nesting). Default None: old bridges and the default
    # WorldBridge.get_sensor_frame() compose path do not populate this.
    additional_frames: dict[str, SensorFrame] | None = None

# WorldCapability → SensorFrame attribute name (None = no sensor field yet).
WORLD_CAPABILITY_SENSOR_FIELDS: dict[WorldCapability, str | None] = {
    WorldCapability.CAMERA_RGB: "rgb",
    WorldCapability.CAMERA_DEPTH: "depth",
    WorldCapability.LIDAR_3D: "lidar",
    WorldCapability.LIDAR_2D: "lidar",
    WorldCapability.IMU: "imu",
    WorldCapability.GNSS: "gnss",
    WorldCapability.WHEEL_ENCODER: "wheel_odometry",
    WorldCapability.RADAR: None,
    WorldCapability.AGENT_SPAWN: None,
    WorldCapability.AGENT_CONTROL: None,
}
