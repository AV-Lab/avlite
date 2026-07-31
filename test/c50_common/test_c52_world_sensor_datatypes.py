import numpy as np
import pytest

from avlite.c50_common.c52_world_sensor_datatypes import (
    CameraParams,
    GnssDatum,
    GnssReading,
    ImuReading,
    SensorFrame,
    WheelOdometry,
    lidar_2d_to_4,
)


def test_lidar_2d_to_4():
    pts = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    out = lidar_2d_to_4(pts)
    assert out.shape == (2, 4)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out[:, :2], pts.astype(np.float32))
    assert np.all(out[:, 2:] == 0)


def test_lidar_2d_to_4_empty():
    out = lidar_2d_to_4(np.zeros((0, 2)))
    assert out.shape == (0, 4)


def test_sensor_frame_defaults():
    frame = SensorFrame()
    assert frame.rgb is None
    assert frame.lidar is None
    assert frame.camera_params is None


def test_camera_params_coerces_to_float64():
    params = CameraParams(
        intrinsic=[[400, 0, 320], [0, 400, 240], [0, 0, 1]],
        world_to_camera=np.eye(4, dtype=np.float32),
        width=640,
        height=480,
    )
    assert params.intrinsic.shape == (3, 3)
    assert params.intrinsic.dtype == np.float64
    assert params.world_to_camera.dtype == np.float64
    assert params.intrinsic[0, 2] == pytest.approx(320.0)

    frame = SensorFrame(camera_params=params)
    assert frame.camera_params.width == 640


def test_camera_params_rejects_bad_intrinsic_shape():
    with pytest.raises(ValueError, match=r"\(3, 3\) intrinsic"):
        CameraParams(
            intrinsic=np.zeros((3, 4)),
            world_to_camera=np.eye(4),
            width=640,
            height=480,
        )


def test_camera_params_rejects_bad_extrinsic_shape():
    with pytest.raises(ValueError, match=r"\(4, 4\) world_to_camera"):
        CameraParams(
            intrinsic=np.eye(3),
            world_to_camera=np.eye(3),
            width=640,
            height=480,
        )


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
