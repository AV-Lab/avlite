import numpy as np
import pytest

from avlite.c50_common.c52_sensor_datatypes import (
    DepthImage,
    GnssDatum,
    GnssReading,
    ImuReading,
    LidarCloud,
    RgbImage,
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


def test_sensor_type_aliases_are_ndarray():
    assert RgbImage is np.ndarray
    assert DepthImage is np.ndarray
    assert LidarCloud is np.ndarray


def test_sensor_frame_defaults():
    frame = SensorFrame()
    assert frame.rgb is None
    assert frame.lidar is None


def test_gnss_reading_datum():
    fix = GnssReading(latitude=24.0, longitude=54.0, altitude=10.0)
    assert fix.datum == GnssDatum.WGS84


def test_imu_and_wheel_odometry():
    imu = ImuReading(linear_accel=(0, 0, 9.8), angular_velocity=(0, 0, 0.1))
    odom = WheelOdometry(linear_velocity=5.0, yaw_rate=0.05)
    frame = SensorFrame(imu=imu, wheel_odometry=odom)
    assert frame.imu.linear_accel[2] == pytest.approx(9.8)
    assert frame.wheel_odometry.yaw_rate == pytest.approx(0.05)
