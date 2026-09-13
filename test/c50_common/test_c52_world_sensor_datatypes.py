import numpy as np
import pytest

from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c50_common.c52_world_sensor_datatypes import (
    Camera,
    GnssDatum,
    GnssReading,
    ImuReading,
    Lidar,
    Sensor,
    SensorFrame,
    WheelOdometry,
)


def _mount(x=0.0, y=0.0, z=0.0, yaw=0.0) -> np.ndarray:
    t = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    t[:2, :2] = [[c, -s], [s, c]]
    t[:3, 3] = [x, y, z]
    return t


def test_sensor_frame_defaults():
    frame = SensorFrame()
    assert frame.rgb is None
    assert frame.lidar is None
    assert frame.camera_sensor is None
    # Every *_sensor field is present with an identity mount.
    assert isinstance(frame.lidar_sensor, Lidar)
    assert isinstance(frame.imu_sensor, Sensor)
    assert isinstance(frame.gnss_sensor, Sensor)
    for sensor in (frame.lidar_sensor, frame.imu_sensor, frame.gnss_sensor):
        np.testing.assert_array_equal(sensor.base_to_sensor, np.eye(4))


def test_camera_coerces_to_float64():
    params = Camera(
        intrinsic=[[400, 0, 320], [0, 400, 240], [0, 0, 1]],
        width=640,
        height=480,
        base_to_sensor=np.eye(4, dtype=np.float32),
    )
    assert params.intrinsic.shape == (3, 3)
    assert params.intrinsic.dtype == np.float64
    assert params.base_to_sensor.dtype == np.float64
    assert params.intrinsic[0, 2] == pytest.approx(320.0)

    frame = SensorFrame(camera_sensor=params)
    assert frame.camera_sensor.width == 640


def test_camera_rejects_bad_intrinsic_shape():
    with pytest.raises(ValueError, match=r"\(3, 3\) intrinsic"):
        Camera(intrinsic=np.zeros((3, 4)), width=640, height=480)


def test_sensor_mount_defaults_to_identity_and_validates_shape():
    for sensor in (
        Camera(intrinsic=np.eye(3), width=4, height=4),
        Lidar(),
        Sensor(),
    ):
        assert isinstance(sensor, Sensor)
        np.testing.assert_array_equal(sensor.base_to_sensor, np.eye(4))

    with pytest.raises(ValueError, match=r"\(4, 4\) base_to_sensor"):
        Lidar(base_to_sensor=np.eye(3))


def test_measurements_carry_no_mount():
    for measurement in (
        ImuReading(linear_accel=(0, 0, 9.8), angular_velocity=(0, 0, 0.0)),
        GnssReading(latitude=1.0, longitude=2.0, altitude=3.0),
        WheelOdometry(linear_velocity=1.0, yaw_rate=0.0),
    ):
        assert not isinstance(measurement, Sensor)
        assert not hasattr(measurement, "base_to_sensor")


def test_to_base_applies_mount_and_keeps_intensity():
    mount = Lidar(base_to_sensor=_mount(x=1.0, z=0.5, yaw=np.pi / 2))
    cloud = np.array([[2.0, 0.0, 0.0, 7.0]], dtype=np.float32)
    out = mount.to_base(cloud)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [[1.0, 2.0, 0.5, 7.0]], atol=1e-6)
    # 2D input stays 2D; None passes through.
    np.testing.assert_allclose(mount.to_base(np.array([[2.0, 0.0]])), [[1.0, 2.0]], atol=1e-9)
    assert mount.to_base(None) is None


def test_to_map_composes_mount_and_state_pose():
    mount = Lidar(base_to_sensor=_mount(x=1.0))
    cloud = np.array([[1.0, 0.0, 0.3, 9.0]], dtype=np.float32)
    # Lidar 1 m ahead of the body origin, ego at (10, 5) facing +y.
    ego = EgoState(x=10.0, y=5.0, theta=np.pi / 2)
    np.testing.assert_allclose(mount.to_map(cloud, ego), [[10.0, 7.0, 0.3, 9.0]], atol=1e-6)
    # Identity mount = already in the body frame; ego z is added to the cloud z.
    ego = EgoState(x=10.0, y=5.0, z=0.2, theta=0.0)
    np.testing.assert_allclose(Lidar().to_map(cloud, ego), [[11.0, 5.0, 0.5, 9.0]], atol=1e-6)
    assert Lidar().to_map(None, ego) is None
    assert Lidar().to_map(np.zeros((0, 4)), ego).shape == (0, 4)


def test_world_capability_sensor_fields_cover_all_caps():
    from avlite.c50_common.c51_capabilities import WorldCapability
    from avlite.c50_common.c52_world_sensor_datatypes import WORLD_CAPABILITY_SENSOR_FIELDS

    assert set(WORLD_CAPABILITY_SENSOR_FIELDS) == set(WorldCapability)
    assert WORLD_CAPABILITY_SENSOR_FIELDS[WorldCapability.CAMERA_RGB] == "rgb"
    assert WORLD_CAPABILITY_SENSOR_FIELDS[WorldCapability.LIDAR_2D] == "lidar"
    assert WORLD_CAPABILITY_SENSOR_FIELDS[WorldCapability.LIDAR_3D] == "lidar"
    assert WORLD_CAPABILITY_SENSOR_FIELDS[WorldCapability.RADAR] is None


def test_gnss_reading_datum():
    fix = GnssReading(latitude=24.0, longitude=54.0, altitude=10.0)
    assert fix.datum == GnssDatum.WGS84


def test_imu_and_wheel_odometry():
    imu = ImuReading(linear_accel=(0, 0, 9.8), angular_velocity=(0, 0, 0.1))
    odom = WheelOdometry(linear_velocity=5.0, yaw_rate=0.05)
    frame = SensorFrame(imu=imu, wheel_odometry=odom)
    assert frame.imu.linear_accel[2] == pytest.approx(9.8)
    assert frame.wheel_odometry.yaw_rate == pytest.approx(0.05)
