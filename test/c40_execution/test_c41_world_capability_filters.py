"""Tests for Bridge Setting world / stack capability enablement filters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c40_execution.c41_world_bridge import (
    WorldBridge,
    is_world_capability_enabled,
    is_world_stack_capability_enabled,
)
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c50_common.c51_capabilities import StackCapability, WorldCapability
from avlite.c50_common.c52_world_sensor_datatypes import CameraParams, GnssReading, LidarCloud


@dataclass
class _StubSensorBridge(WorldBridge):
    ego_state: EgoState = None  # type: ignore[assignment]
    world_capabilities = frozenset(
        {WorldCapability.LIDAR_2D, WorldCapability.GNSS, WorldCapability.CAMERA_RGB}
    )
    stack_capabilities = frozenset()

    def __post_init__(self):
        if self.ego_state is None:
            self.ego_state = EgoState(x=0.0, y=0.0, theta=0.0)

    def control_ego_state(self, cmd, dt=0.01):
        pass

    def get_lidar_data(self, agent_id=0) -> LidarCloud:
        return np.zeros((1, 4), dtype=np.float32)

    def get_gnss(self, agent_id=0) -> GnssReading:
        return GnssReading(latitude=1.0, longitude=2.0, altitude=0.0)

    def get_rgb_image(self, agent_id=0):
        return np.zeros((4, 4, 3), dtype=np.uint8)

    def get_camera_params(self, agent_id=0) -> CameraParams:
        return CameraParams(
            intrinsic=np.eye(3), world_to_camera=np.eye(4), width=4, height=4
        )


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
        assert frame.gnss is None
    finally:
        ExecutionSettings.c41_world_capabilities = None


def test_get_sensor_frame_keeps_lidar_if_either_2d_or_3d_enabled():
    bridge = _StubSensorBridge()
    ExecutionSettings.c41_world_capabilities = ["LIDAR_3D"]
    try:
        frame = bridge.get_sensor_frame()
        assert frame.lidar is not None
        assert frame.gnss is None
    finally:
        ExecutionSettings.c41_world_capabilities = None


def test_get_sensor_frame_keeps_camera_params_when_camera_enabled():
    bridge = _StubSensorBridge()
    ExecutionSettings.c41_world_capabilities = ["CAMERA_RGB"]
    try:
        frame = bridge.get_sensor_frame()
        assert frame.rgb is not None
        assert frame.camera_params is not None
        assert frame.camera_params.width == 4
    finally:
        ExecutionSettings.c41_world_capabilities = None


def test_get_sensor_frame_nulls_camera_params_when_camera_disabled():
    bridge = _StubSensorBridge()
    ExecutionSettings.c41_world_capabilities = ["LIDAR_2D"]
    try:
        frame = bridge.get_sensor_frame()
        assert frame.rgb is None
        assert frame.camera_params is None
    finally:
        ExecutionSettings.c41_world_capabilities = None


def test_get_sensor_frame_keeps_camera_params_for_depth_only_camera():
    bridge = _StubSensorBridge()
    ExecutionSettings.c41_world_capabilities = ["CAMERA_DEPTH"]
    try:
        frame = bridge.get_sensor_frame()
        assert frame.rgb is None
        assert frame.camera_params is not None
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
