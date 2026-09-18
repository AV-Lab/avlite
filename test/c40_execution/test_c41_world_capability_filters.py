"""Tests for Bridge Setting world / stack capability enablement filters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c40_execution.c41_world_bridge import (
    WorldBridge,
    is_world_capability_enabled,
    is_world_stack_capability_enabled,
)
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c50_common.c51_capabilities import StackCapability, WorldCapability
from avlite.c50_common.c52_world_sensor_datatypes import (
    Camera,
    Gnss,
    Imu,
    Lidar,
    LidarCloud,
    SensorFrame,
    WheelOdometry,
)


@dataclass
class _StubSensorBridge(WorldBridge):
    ego_state: EgoState = None  # type: ignore[assignment]
    world_capabilities = frozenset(
        {
            WorldCapability.LIDAR_2D,
            WorldCapability.GNSS,
            WorldCapability.CAMERA_RGB,
            WorldCapability.AGENT_SENSING,
        }
    )
    stack_capabilities = frozenset()

    def __post_init__(self):
        if self.ego_state is None:
            self.ego_state = EgoState(x=0.0, y=0.0, theta=0.0)

    def control_ego_state(self, cmd, dt=0.01):
        pass

    def get_lidar_data(self, agent_id=0) -> LidarCloud:
        return np.zeros((1, 4), dtype=np.float32)

    def get_lidar_sensor(self, agent_id=0) -> Lidar:
        return Lidar()

    def get_gnss(self, agent_id=0) -> Gnss:
        return Gnss(latitude=1.0, longitude=2.0, altitude=0.0)

    def get_rgb_image(self, agent_id=0):
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def get_camera_sensor(self, agent_id=0) -> Camera:
        return Camera(intrinsic=np.eye(3), width=4, height=4)


def test_world_capability_none_means_all_enabled():
    ExecutionSettings.c41_world_capabilities = None
    assert is_world_capability_enabled(WorldCapability.LIDAR_2D)
    assert is_world_capability_enabled(WorldCapability.GNSS)


def test_world_capability_explicit_list():
    ExecutionSettings.c41_world_capabilities = ["LIDAR_2D"]
    assert is_world_capability_enabled(WorldCapability.LIDAR_2D)
    assert not is_world_capability_enabled(WorldCapability.GNSS)
    ExecutionSettings.c41_world_capabilities = None


def test_get_sensor_frame_nulls_disabled_capabilities():
    bridge = _StubSensorBridge()
    ExecutionSettings.c41_world_capabilities = ["LIDAR_2D"]
    try:
        frame = bridge.get_sensor_frame()
        assert frame.lidar is not None
        assert frame.lidar.points is not None
        assert frame.gnss is None
    finally:
        ExecutionSettings.c41_world_capabilities = None


def test_get_sensor_frame_keeps_lidar_if_either_2d_or_3d_enabled():
    bridge = _StubSensorBridge()
    ExecutionSettings.c41_world_capabilities = ["LIDAR_3D"]
    try:
        frame = bridge.get_sensor_frame()
        assert frame.lidar is not None
        assert frame.lidar.points is not None
        assert frame.gnss is None
    finally:
        ExecutionSettings.c41_world_capabilities = None


def test_get_sensor_frame_keeps_camera_sensor_when_camera_enabled():
    bridge = _StubSensorBridge()
    ExecutionSettings.c41_world_capabilities = ["CAMERA_RGB"]
    try:
        frame = bridge.get_sensor_frame()
        assert frame.camera is not None
        assert frame.camera.rgb is not None
        assert frame.camera.width == 4
    finally:
        ExecutionSettings.c41_world_capabilities = None


def test_get_sensor_frame_preserves_camera_metadata_when_camera_disabled():
    bridge = _StubSensorBridge()
    ExecutionSettings.c41_world_capabilities = ["LIDAR_2D"]
    try:
        frame = bridge.get_sensor_frame()
        assert frame.camera is not None
        assert frame.camera.rgb is None
        assert frame.camera.depth is None
        assert frame.camera.width == 4
    finally:
        ExecutionSettings.c41_world_capabilities = None


def test_get_sensor_frame_keeps_camera_sensor_for_depth_only_camera():
    bridge = _StubSensorBridge()
    ExecutionSettings.c41_world_capabilities = ["CAMERA_DEPTH"]
    try:
        frame = bridge.get_sensor_frame()
        assert frame.camera is not None
        assert frame.camera.rgb is None
    finally:
        ExecutionSettings.c41_world_capabilities = None


def test_world_stack_capability_none_means_all_enabled():
    ExecutionSettings.c41_world_stack_capabilities = None
    assert is_world_stack_capability_enabled(StackCapability.DETECTION)
    assert is_world_stack_capability_enabled(StackCapability.LOCALIZATION)


def test_world_stack_capability_explicit_list():
    ExecutionSettings.c41_world_stack_capabilities = ["LOCALIZATION"]
    assert is_world_stack_capability_enabled(StackCapability.LOCALIZATION)
    assert not is_world_stack_capability_enabled(StackCapability.DETECTION)
    ExecutionSettings.c41_world_stack_capabilities = None


def test_settings_schema_has_split_fields_not_c41_provided():
    assert hasattr(ExecutionSettings, "c41_world_capabilities")
    assert hasattr(ExecutionSettings, "c41_world_stack_capabilities")
    assert not hasattr(ExecutionSettings, "c41_provided")


@pytest.mark.parametrize("enabled", [None, [], ["CAMERA_RGB"], ["CAMERA_DEPTH"], ["LIDAR_2D"], ["LIDAR_3D"]])
def test_filter_applies_to_every_device_preserving_metadata(enabled):
    ExecutionSettings.c41_world_capabilities = enabled
    mount = np.eye(4)
    mount[0, 3] = 2.0
    frame = SensorFrame(
        cameras={name: Camera(
            np.eye(3), 4, 4, sensor_name=name, sensor_id=f"cam-{name}", stamp=1.0,
            base_to_sensor=mount, rgb=np.ones((4, 4, 3), dtype=np.uint8),
            depth=np.ones((4, 4), dtype=np.float32),
        ) for name in ("front", "rear")},
        lidars={name: Lidar(
            sensor_name=name, sensor_id=f"lidar-{name}", stamp=0.9,
            base_to_sensor=mount, points=np.ones((2, 4), dtype=np.float32),
        ) for name in ("top", "bumper")},
        primary_camera_name="rear", primary_lidar_name="bumper", stamp=1.1,
    )
    cameras, lidars = dict(frame.cameras), dict(frame.lidars)
    assert WorldBridge._apply_world_capability_filter(frame) is frame
    for name, camera in frame.cameras.items():
        assert camera is cameras[name]
        assert (camera.rgb is not None) == (enabled is None or "CAMERA_RGB" in enabled)
        assert (camera.depth is not None) == (enabled is None or "CAMERA_DEPTH" in enabled)
        assert camera.sensor_id == f"cam-{name}"
        assert camera.stamp == 1.0
        np.testing.assert_array_equal(camera.base_to_sensor, mount)
        np.testing.assert_array_equal(camera.intrinsic, np.eye(3))
    for name, lidar in frame.lidars.items():
        assert lidar is lidars[name]
        assert (lidar.points is not None) == (
            enabled is None or bool({"LIDAR_2D", "LIDAR_3D"}.intersection(enabled))
        )
        assert lidar.sensor_id == f"lidar-{name}"
        assert lidar.stamp == 0.9
        np.testing.assert_array_equal(lidar.base_to_sensor, mount)
    assert frame.camera is cameras["rear"]
    assert frame.lidar is lidars["bumper"]
    assert frame.stamp == 1.1
    assert not any("." in name for name in vars(frame))


def test_compose_preserves_names_ids_stamps_and_does_not_mutate_templates(monkeypatch):
    bridge = _StubSensorBridge()
    camera = Camera(np.eye(3), 4, 4, sensor_name="front", sensor_id="cam-01", stamp=1.0)
    lidar = Lidar(sensor_name="top", sensor_id="lidar-01", stamp=0.9)
    monkeypatch.setattr(bridge, "get_camera_sensor", lambda: camera)
    monkeypatch.setattr(bridge, "get_lidar_sensor", lambda: lidar)
    ExecutionSettings.c41_world_capabilities = None
    first = bridge.get_sensor_frame()
    ExecutionSettings.c41_world_capabilities = []
    second = bridge.get_sensor_frame()
    assert first.camera is first.cameras["front"]
    assert first.lidar is first.lidars["top"]
    assert first.get_camera("cam-01") is first.camera
    assert first.get_lidar("lidar-01") is first.lidar
    assert first.camera is not camera and first.camera is not second.camera
    assert first.lidar is not lidar and first.lidar is not second.lidar
    assert first.camera.rgb is not None and first.lidar.points is not None
    assert second.camera.rgb is None and second.lidar.points is None
    assert camera.rgb is None and lidar.points is None
    assert first.camera.stamp == 1.0 and first.lidar.stamp == 0.9
    assert first.stamp is None  # No reliable clock is available in the generic bridge.


def test_compose_keeps_declared_lidar_without_reading(monkeypatch):
    bridge = _StubSensorBridge()
    monkeypatch.setattr(bridge, "get_lidar_data", lambda: None)
    frame = bridge.get_sensor_frame()
    assert frame.lidar is not None
    assert frame.lidar.points is None


def test_compose_rejects_image_without_calibration(monkeypatch):
    bridge = _StubSensorBridge()
    monkeypatch.setattr(bridge, "get_camera_sensor", lambda: None)
    with pytest.raises(ValueError, match="Camera readings require calibration"):
        bridge.get_sensor_frame()


def test_non_ego_compose_forwards_agent_id_to_every_getter(monkeypatch):
    bridge = _StubSensorBridge()
    calls = []
    values = {
        "get_rgb_image": None, "get_depth_image": None, "get_camera_sensor": None,
        "get_lidar_data": None, "get_lidar_sensor": Lidar(),
        "get_imu": None, "get_gnss": None,
        "get_wheel_odometry": None,
    }
    for name, value in values.items():
        def getter(*, agent_id, name=name, value=value):
            calls.append((name, agent_id))
            return value
        monkeypatch.setattr(bridge, name, getter)
    frame = bridge.get_sensor_frame(agent_id=7)
    assert calls == [(name, 7) for name in values]
    assert frame.cameras == {}
    assert frame.lidar is not None


@pytest.mark.parametrize("field, getter_name, reading", [
    ("imu", "get_imu", Imu((0, 0, 9.8), (0, 0, 0.1))),
    ("gnss", "get_gnss", Gnss(24.0, 54.0, 10.0)),
    ("wheel_odometry", "get_wheel_odometry", WheelOdometry(5.0, 0.05)),
])
def test_compose_copies_complete_single_source_sensors(monkeypatch, field, getter_name, reading):
    bridge = _StubSensorBridge()
    reading.sensor_name = "body"
    reading.sensor_id = "device-01"
    reading.stamp = 2.0
    reading.base_to_sensor[0, 3] = 1.5
    monkeypatch.setattr(bridge, getter_name, lambda: reading)
    ExecutionSettings.c41_world_capabilities = None
    first = bridge.get_sensor_frame()
    second = bridge.get_sensor_frame()
    sensor = getattr(first, field)
    assert sensor is not reading and sensor is not getattr(second, field)
    assert sensor.sensor_id == "device-01" and sensor.sensor_name == "body"
    assert sensor.stamp == 2.0
    np.testing.assert_array_equal(sensor.base_to_sensor, reading.base_to_sensor)
    if field == "imu":
        assert sensor.linear_accel == (0, 0, 9.8)
    elif field == "gnss":
        assert sensor.latitude == 24.0
    else:
        assert sensor.linear_velocity == 5.0
    ExecutionSettings.c41_world_capabilities = []
    WorldBridge._apply_world_capability_filter(second)
    assert getattr(second, field) is None
    assert getattr(first, field) is sensor
