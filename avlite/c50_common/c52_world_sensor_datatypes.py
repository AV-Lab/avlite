"""AVLite canonical sensor formats.

All WorldBridge implementations must populate SensorFrame using these exact
layouts. Convert simulator/ROS messages in the bridge; do not pass raw
message layouts to perception or localization.

Two kinds of object live here. A **sensor** is the static description of a
device mounted on the ego body (where it sits, how it is calibrated); it never
changes per tick. A **measurement** is the per-tick data that device produced.
``SensorFrame`` holds both, one rule: ``<x>`` is the measurement, ``<x>_sensor``
is the device that produced it.

Measurements (per tick)
-----------------------
rgb            (H, W, 3) uint8, row-major RGB
depth          (H, W) float32, metres
lidar          (N, 4) float32, [x, y, z, intensity] in the lidar's own coordinate frame
imu            ImuReading — linear accel + angular velocity in the IMU's coordinate frame
gnss           GnssReading — WGS84 lat/lon/alt + optional map x/y/z
wheel_odometry WheelOdometry — linear_velocity m/s + yaw_rate rad/s (body frame)

Sensors (static)
----------------
camera_sensor  Camera — intrinsic K, resolution, mount of the optical frame (None = no camera)
lidar_sensor   Lidar  — mount (identity by default)
imu_sensor     Sensor — mount (identity by default)
gnss_sensor    Sensor — antenna mount (identity by default)

Coordinate frames
-----------------
Every ``Sensor`` carries one static mount, ``base_to_sensor``: the pose of the
device in the ego body frame, so ``p_body = base_to_sensor @ p_sensor``. The
body frame has its origin at ``State.x/y/z``, +x along the heading, z up.
Bridges never bake the ego pose into measurements; the stack composes
sensor → body → map from its own pose estimate with
``Sensor.to_map(points, state)``, which uses ``State.pose_matrix()``. That is
what keeps localization independent of the bridge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from avlite.c50_common.c51_capabilities import WorldCapability

if TYPE_CHECKING:
    from avlite.c10_perception.c11_perception_model import State

# Semantic ndarray aliases — layout defined in module docstring above.
RgbImage = np.ndarray  # (H, W, 3) uint8 RGB
DepthImage = np.ndarray  # (H, W) float32 metres
LidarCloud = np.ndarray  # (N, 4) float32 [x, y, z, intensity]


@dataclass(kw_only=True)
class Sensor:
    """Static description of a device mounted on the ego body.

    ``base_to_sensor`` is the (4, 4) homogeneous pose of the device in the ego
    body frame: ``p_body = base_to_sensor @ p_sensor``. Identity by default
    (device at the body origin, axes aligned with the body). Used directly for
    devices with no calibration of their own (IMU, GNSS); ``Camera`` and
    ``Lidar`` subclass it. Keyword-only so subclasses keep positional fields.
    """

    base_to_sensor: np.ndarray = field(default_factory=lambda: np.eye(4))

    def __post_init__(self) -> None:
        self.base_to_sensor = np.asarray(self.base_to_sensor, dtype=np.float64)
        if self.base_to_sensor.shape != (4, 4):
            raise ValueError(
                f"expected (4, 4) base_to_sensor, got shape {self.base_to_sensor.shape}"
            )

    def to_base(self, points: np.ndarray | None) -> np.ndarray | None:
        """Express ``points`` (N, 3+) measured in this device's coordinate frame in the ego body frame.

        Columns beyond xyz (e.g. intensity) are preserved. 2D input (N, 2) is
        treated as z = 0 and returned as (N, 2). ``None`` passes through.
        """
        return self._transform(self.base_to_sensor, points)

    def to_map(self, points: np.ndarray | None, state: State) -> np.ndarray | None:
        """Express ``points`` (N, 3+) measured in this device's coordinate frame in the map frame.

        Composes the static mount with the body pose of ``state`` (any
        ``State``; ``perception_model.ego_vehicle`` for the ego). Full 4×4
        composition, so it follows whatever ``state.pose_matrix()`` encodes.
        """
        return self._transform(state.pose_matrix() @ self.base_to_sensor, points)

    @staticmethod
    def _transform(transform: np.ndarray, points: np.ndarray | None) -> np.ndarray | None:
        """Apply a (4, 4) homogeneous transform to the xyz columns of ``points``."""
        if points is None:
            return None
        pts = np.asarray(points)
        if pts.ndim != 2 or pts.shape[0] == 0:
            return pts
        n_xyz = min(3, pts.shape[1])
        xyz = np.zeros((pts.shape[0], 3), dtype=np.float64)
        xyz[:, :n_xyz] = pts[:, :n_xyz]
        xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
        out = pts.astype(np.float64, copy=True) if pts.dtype.kind != "f" else pts.copy()
        out[:, :n_xyz] = xyz[:, :n_xyz]
        return out


@dataclass
class ImuReading:
    """Inertial measurement at a single timestep, in the IMU's coordinate frame.

    The IMU mount is ``SensorFrame.imu_sensor``.
    """

    linear_accel: tuple[float, float, float]  # (ax, ay, az) m/s²
    angular_velocity: tuple[float, float, float]  # (gx, gy, gz) rad/s


class GnssDatum(Enum):
    """Geodetic datum for GNSS latitude/longitude/altitude."""

    WGS84 = "WGS84"


@dataclass
class GnssReading:
    """GNSS fix: raw geodetic measurement plus optional map-frame position.

    The antenna mount is ``SensorFrame.gnss_sensor``.

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
    """Ego motion derived from wheel encoders, in the ego body frame."""

    linear_velocity: float  # forward speed along ego x-axis, m/s (+ = forward)
    yaw_rate: float  # heading change rate, rad/s (+ = counter-clockwise)


@dataclass
class Camera(Sensor):
    """A pinhole camera: intrinsics, resolution, and mount of its optical frame.

    The camera's coordinate frame is the OpenCV optical frame — x right, y down,
    z forward along the optical axis, z > 0 in front of the camera — so the
    inherited ``base_to_sensor`` is the static pose of that optical frame in the
    ego body frame (it includes the body → optical axis rotation). To project a
    map-frame point, compose with the ego pose estimate
    (``perception_model.ego_vehicle``)::

        p_cam = inv(ego.pose_matrix() @ base_to_sensor) @ [x_map, y_map, z_map, 1]
        u = fx * X / Z + cx,  v = fy * Y / Z + cy

    Self-contained per camera: an instance carries everything needed to project
    into its own image, so extra cameras in ``SensorFrame.additional_frames``
    each carry their own ``Camera``.
    """

    intrinsic: np.ndarray  # (3, 3) float64 K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    width: int  # pixels; resolution the intrinsic is valid for
    height: int  # pixels; resolution the intrinsic is valid for

    def __post_init__(self) -> None:
        super().__post_init__()
        self.intrinsic = np.asarray(self.intrinsic, dtype=np.float64)
        if self.intrinsic.shape != (3, 3):
            raise ValueError(f"expected (3, 3) intrinsic, got shape {self.intrinsic.shape}")


@dataclass
class Lidar(Sensor):
    """A lidar: only its mount. Identity means the cloud is already in the ego body frame."""


@dataclass
class SensorFrame:
    """Snapshot of all measurements for one execution tick, plus the sensors that produced them.

    Field rule: ``<x>`` is the measurement, ``<x>_sensor`` is the static device
    description. Measurements may be None when the bridge does not provide that
    device or when gated off by the ExecutionSettings.c41_world_capabilities
    filter; ``*_sensor`` fields default to an identity mount (``camera_sensor``
    is None without a camera, since it needs intrinsics).
    """

    # Camera: colour image from the primary camera.
    # Shape (H, W, 3), dtype uint8, channels in RGB order (not BGR).
    # H and W vary by camera; algorithms must not assume fixed resolution.
    rgb: RgbImage | None = None

    # Camera: per-pixel distance from the primary camera's image plane.
    # Shape (H, W), dtype float32, values in metres.
    # Must match rgb height/width when both are present.
    depth: DepthImage | None = None

    # The primary camera, i.e. the device that produced rgb/depth. None when
    # the bridge exposes no camera. Required to project lidar into the image.
    camera_sensor: Camera | None = None

    # LiDAR: point cloud in the lidar's own coordinate frame (the ego body frame
    # when lidar_sensor is identity). Shape (N, 4), dtype float32, columns
    # [x, y, z, intensity]. x, y, z in metres; intensity is device-specific
    # reflectance (0+). N varies per scan. 2D scanners: set z=0 and intensity=0
    # in the bridge. Map frame: ``lidar_sensor.to_map(lidar, ego)``.
    lidar: LidarCloud | None = None
    lidar_sensor: Lidar = field(default_factory=Lidar)

    imu: ImuReading | None = None
    imu_sensor: Sensor = field(default_factory=Sensor)

    gnss: GnssReading | None = None
    gnss_sensor: Sensor = field(default_factory=Sensor)  # antenna mount

    wheel_odometry: WheelOdometry | None = None  # body-frame quantity; no mount

    stamp: float | None = None  # acquisition time, seconds (sim or wall clock)
    frame_id: str | None = None  # optional label for the bridge's body frame

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
