"""AVLite canonical sensor formats.

All WorldBridge implementations must populate SensorFrame using these exact
layouts. Convert simulator/ROS messages in the bridge; do not pass raw
message layouts to perception or localization.

``Camera`` and ``Lidar`` are sensor-state snapshots: each contains its device
identity, mount, calibration (where applicable), acquisition time, and reading.
``SensorFrame.cameras`` and ``SensorFrame.lidars`` hold all devices by name;
``frame.camera`` and ``frame.lidar`` refer to the explicitly selected primary
objects in those collections. There is no separate storage for extra devices.

Reading layouts
---------------
Camera.rgb     (H, W, 3) uint8, row-major RGB
Camera.depth   (H, W) float32, metres
Lidar.points   (N, 4) float32, [x, y, z, intensity] in the lidar's coordinate frame
imu            Imu — linear accel + angular velocity in the IMU's coordinate frame
gnss           Gnss — WGS84 lat/lon/alt + optional map x/y/z
wheel_odometry WheelOdometry — linear_velocity m/s + yaw_rate rad/s (body frame)

IMU, GNSS, and wheel odometry are single Sensor-derived snapshots on the frame.
Each owns its identity, mount, timestamp, and reading fields; there are no
separate IMU/GNSS mount objects or collections. Their frame entries are None
when unavailable or disabled.

Bridges must create a fresh sensor-state object for each acquisition, rather
than mutate an object retained by an earlier frame. ``dataclasses.replace``
can reuse device metadata while attaching a new reading and ``stamp``. It is
a shallow copy: do not mutate shared calibration or reading buffers; copy
buffers if the driver reuses them. A missing camera/lidar reading is ``None``
on the sensor, not a reason to remove it or select a different primary.

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
from typing import TYPE_CHECKING, Mapping, TypeVar

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
    """Shared device identity, mount, and acquisition time.

    ``base_to_sensor`` is the (4, 4) homogeneous pose of the device in the ego
    body frame: ``p_body = base_to_sensor @ p_sensor``. Identity by default
    (device at the body origin, axes aligned with the body). All concrete
    sensor snapshots subclass it. Keyword-only so subclasses keep positional
    reading/calibration fields.
    ``stamp`` belongs to the reading in this snapshot, not the device's lifetime.
    """

    sensor_name: str | None = None  # stable name; if set, must match the collection key
    sensor_id: str | None = None  # stable ID, unique within the sensor modality
    stamp: float | None = None  # acquisition time in seconds; None when unknown
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
class Imu(Sensor):
    """IMU snapshot: device metadata and readings in the IMU's coordinate frame.

    The inherited ``base_to_sensor`` describes this IMU's mount. Acceleration
    and angular velocity are vectors, not positions: do not pass them through
    the inherited point-transform helpers, which include translation.
    """

    linear_accel: tuple[float, float, float]  # (ax, ay, az) m/s²
    angular_velocity: tuple[float, float, float]  # (gx, gy, gz) rad/s


class GnssDatum(Enum):
    """Geodetic datum for GNSS latitude/longitude/altitude."""

    WGS84 = "WGS84"


@dataclass
class Gnss(Sensor):
    """GNSS snapshot: receiver metadata, geodetic fix, and optional map position.

    The inherited ``base_to_sensor`` is the antenna mount. Point-transform
    helpers do not convert latitude/longitude/altitude into Cartesian points.

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
class WheelOdometry(Sensor):
    """Wheel-odometry source metadata and derived ego-body motion.

    Velocity and yaw rate remain body-relative, regardless of the source
    mount metadata. Do not apply the inherited point transforms to them.
    """

    linear_velocity: float  # forward speed along ego x-axis, m/s (+ = forward)
    yaw_rate: float  # heading change rate, rad/s (+ = counter-clockwise)


@dataclass
class Camera(Sensor):
    """A pinhole camera snapshot: calibration, optical mount, and RGB/depth data.

    The camera's coordinate frame is the OpenCV optical frame — x right, y down,
    z forward along the optical axis, z > 0 in front of the camera — so the
    inherited ``base_to_sensor`` is the static pose of that optical frame in the
    ego body frame (it includes the optical → body axis rotation). To project a
    map-frame point, compose with the ego pose estimate
    (``perception_model.ego_vehicle``)::

        p_cam = inv(ego.pose_matrix() @ base_to_sensor) @ [x_map, y_map, z_map, 1]
        u = fx * X / Z + cx,  v = fy * Y / Z + cy

    Every camera in ``SensorFrame.cameras`` carries its own calibration and
    readings. RGB and depth, when both present, must be aligned to this optical
    frame and match its resolution. Use separate camera entries when they have
    different calibration, optical frames, or acquisition times.
    """

    intrinsic: np.ndarray  # (3, 3) float64 K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    width: int  # pixels; resolution the intrinsic is valid for
    height: int  # pixels; resolution the intrinsic is valid for

    rgb: RgbImage | None = None  # (H, W, 3) uint8 RGB, not BGR
    depth: DepthImage | None = None  # (H, W) float32, metres from the image plane

    def __post_init__(self) -> None:
        super().__post_init__()
        self.intrinsic = np.asarray(self.intrinsic, dtype=np.float64)
        if self.intrinsic.shape != (3, 3):
            raise ValueError(f"expected (3, 3) intrinsic, got shape {self.intrinsic.shape}")


@dataclass
class Lidar(Sensor):
    """A lidar snapshot: device mount and point cloud in the device's frame.

    ``points`` is (N, 4) float32 [x, y, z, intensity], with xyz in metres.
    Bridges pad 2D scans with z=0 and intensity=0. An identity mount describes
    a scanner at the body origin with body-aligned axes. Keep the physical
    mount for offset scanners; do not pre-transform their points to the body.
    Map coordinates: ``lidar.to_map(lidar.points, ego)``.
    """

    points: LidarCloud | None = None


_SensorT = TypeVar("_SensorT", bound=Sensor)


def _find_sensor(sensors: Mapping[str, _SensorT], key: str) -> _SensorT:
    """Find a sensor by collection name or device ID, rejecting ambiguity."""
    matches = [
        sensor
        for name, sensor in sensors.items()
        if name == key or sensor.sensor_id == key
    ]
    if not matches:
        raise KeyError(f"Unknown sensor: {key}")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous sensor name/ID: {key}")
    return matches[0]


def _validate_sensors(
    sensors: Mapping[str, _SensorT],
    primary_name: str | None,
    sensor_type: type[_SensorT],
) -> None:
    """Validate one modality's names, IDs, and explicit primary selection."""
    ids: set[str] = set()
    for name, sensor in sensors.items():
        if not isinstance(name, str) or not name or name == "primary":
            raise ValueError("Sensor names must be nonempty strings other than 'primary'")
        # Hot reload replaces class objects while bridges can retain snapshots.
        # Accept the same qualified type (or a subclass) from an earlier reload.
        if not isinstance(sensor, sensor_type) and not any(
            cls.__module__ == sensor_type.__module__
            and cls.__qualname__ == sensor_type.__qualname__
            for cls in type(sensor).__mro__
        ):
            raise TypeError(f"Expected {sensor_type.__name__} for sensor {name!r}")
        if sensor.sensor_name is not None and sensor.sensor_name != name:
            raise ValueError(f"Sensor name {sensor.sensor_name!r} does not match key {name!r}")
        sensor_id = sensor.sensor_id
        if sensor_id is None:
            continue
        if not isinstance(sensor_id, str) or not sensor_id or sensor_id == "primary":
            raise ValueError("Sensor IDs must be nonempty strings other than 'primary'")
        if sensor_id in ids:
            raise ValueError(f"Duplicate sensor ID: {sensor_id}")
        if sensor_id in sensors and sensor_id != name:
            raise ValueError(f"Ambiguous sensor name/ID: {sensor_id}")
        ids.add(sensor_id)
    if primary_name is not None and primary_name not in sensors:
        raise ValueError(f"Unknown primary {sensor_type.__name__} name: {primary_name}")


@dataclass
class SensorFrame:
    """Named sensor-state snapshots assembled for one execution tick.

    All cameras/lidars, including the primaries, live in their respective
    dictionaries. Keys are stable names; ``get_camera`` and ``get_lidar`` also
    accept device IDs. ``"primary"`` is reserved as a lookup alias. Primary
    selection uses a collection name, never an ID or dictionary order.

    ``camera``/``lidar`` return None when no primary is selected, even when the
    collection is nonempty. A selected sensor remains available when its
    reading is None. Unknown explicit lookups raise KeyError; ambiguous ones
    raise ValueError. Names, IDs, and primary references are validated on
    construction. Build a new frame when changing the sensor configuration.

    Per-sensor stamps are acquisition times; ``stamp`` is assembly time, not
    a claim that the readings are synchronized. Bridges must use the same
    clock domain for these timestamps.
    """

    cameras: dict[str, Camera] = field(default_factory=dict)
    lidars: dict[str, Lidar] = field(default_factory=dict)

    primary_camera_name: str | None = None
    primary_lidar_name: str | None = None

    # Single-source sensors: each object owns its metadata and readings.
    imu: Imu | None = None
    gnss: Gnss | None = None
    wheel_odometry: WheelOdometry | None = None  # readings remain body-relative

    stamp: float | None = None  # snapshot assembly time, seconds (sim or wall clock)
    frame_id: str | None = None  # optional label for the bridge's body frame

    def __post_init__(self) -> None:
        _validate_sensors(self.cameras, self.primary_camera_name, Camera)
        _validate_sensors(self.lidars, self.primary_lidar_name, Lidar)

    @property
    def camera(self) -> Camera | None:
        """The selected primary camera, or None when none is selected."""
        if self.primary_camera_name is None:
            return None
        return self.cameras[self.primary_camera_name]

    @property
    def lidar(self) -> Lidar | None:
        """The selected primary lidar, or None when none is selected."""
        if self.primary_lidar_name is None:
            return None
        return self.lidars[self.primary_lidar_name]

    def get_camera(self, key: str = "primary") -> Camera | None:
        """Look up a camera by name, device ID, or the primary alias."""
        return self.camera if key == "primary" else _find_sensor(self.cameras, key)

    def get_lidar(self, key: str = "primary") -> Lidar | None:
        """Look up a lidar by name, device ID, or the primary alias."""
        return self.lidar if key == "primary" else _find_sensor(self.lidars, key)


# WorldCapability → reading path (None = no sensor field yet). A dotted path
# such as "cameras.rgb" targets rgb on EVERY value in frame.cameras, not just
# the primary camera. Collection filtering clears payloads, preserving metadata;
# a single-source entry (imu, gnss, wheel_odometry) is set to None when disabled.
# The bridge filter must traverse these paths; setattr(frame, path, None) is
# not sufficient. LiDAR 2D/3D share a path: keep points when either is enabled.
WORLD_CAPABILITY_SENSOR_FIELDS: dict[WorldCapability, str | None] = {
    WorldCapability.CAMERA_RGB: "cameras.rgb",
    WorldCapability.CAMERA_DEPTH: "cameras.depth",
    WorldCapability.LIDAR_3D: "lidars.points",
    WorldCapability.LIDAR_2D: "lidars.points",
    WorldCapability.IMU: "imu",
    WorldCapability.GNSS: "gnss",
    WorldCapability.WHEEL_ENCODER: "wheel_odometry",
    WorldCapability.RADAR: None,
    WorldCapability.AGENT_SPAWN: None,
    WorldCapability.AGENT_CONTROL: None,
    WorldCapability.AGENT_SENSING: None,
}
