"""Unit tests for Pure Pursuit and Follow the Gap (avlite.c30_control.c35_pure_pursuit)."""

import numpy as np
import pytest

from avlite.c10_perception.c11_perception_model import EgoState
from avlite.c30_control.c35_pure_pursuit import FollowTheGapController, PurePursuitController
from avlite.c30_control.c39_settings import ControlSettingsSchema
from avlite.c50_common.c52_world_sensor_datatypes import SensorFrame
from avlite.c50_common.c54_trajectory_tracker import TrajectoryTracker


def _straight_path(x_end: float = 100.0, n: int = 21, velocity: float = 5.0) -> TrajectoryTracker:
    xs = [x_end * i / (n - 1) for i in range(n)]
    path = [(x, 0.0) for x in xs]
    return TrajectoryTracker(path=path, velocity=[velocity] * n)


def _pp_settings(**overrides) -> ControlSettingsSchema:
    defaults = {
        "c35_lookahead_distance": 8.0,
        "c35_min_lookahead": 3.0,
        "c35_max_lookahead": 20.0,
        "c35_lookahead_speed_gain": 0.0,
        "c35_valpha": 0.0,
        "c35_vbeta": 0.0,
        "c35_vgamma": 0.0,
        "c35_cruise_velocity": 5.0,
        "c35_lidar_z_min": -1.5,
        "c35_lidar_z_max": 2.0,
        "c32_ego_distance_front_axle": 2.5,
        "c32_ego_max_steering": 0.7,
        "c32_ego_min_steering": -0.7,
    }
    defaults.update(overrides)
    return ControlSettingsSchema(**defaults)


class TestPurePursuitController:
    def test_on_path_yields_near_zero_steer(self):
        trajectory = _straight_path()
        controller = PurePursuitController(tj=trajectory, setting=_pp_settings())
        ego = EgoState(x=20.0, y=0.0, theta=0.0, velocity=5.0)
        cmd = controller.control(ego)
        assert cmd.steer == pytest.approx(0.0, abs=0.05)

    def test_lateral_offset_produces_corrective_steer(self):
        trajectory = _straight_path()
        controller = PurePursuitController(tj=trajectory, setting=_pp_settings())
        # Ego above the path (positive world y) while heading along +x → need right steer (negative).
        ego = EgoState(x=20.0, y=2.0, theta=0.0, velocity=5.0)
        cmd = controller.control(ego)
        assert cmd.steer < 0.0

    def test_missing_trajectory_returns_zero(self):
        controller = PurePursuitController(tj=None, setting=_pp_settings())
        ego = EgoState(x=0.0, y=0.0, theta=0.0, velocity=0.0)
        cmd = controller.control(ego)
        assert cmd.steer == 0.0
        assert cmd.acceleration == 0.0


class TestFollowTheGapController:
    def test_largest_gap_aims_into_opening(self):
        """Dense returns on the right, open left → positive (left) steer."""
        setting = _pp_settings(c35_lookahead_distance=8.0)
        controller = FollowTheGapController(setting=setting)
        ego = EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0)
        # Forward half-plane: cluster of hits on the right (negative y), open on the left.
        angles = np.linspace(-np.pi / 2 + 0.05, -0.2, 20)
        ranges = np.full_like(angles, 5.0)
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        lidar = np.column_stack([xs, ys, np.zeros_like(xs), np.ones_like(xs)]).astype(np.float32)
        cmd = controller.control(ego, sensors=SensorFrame(lidar=lidar))
        assert cmd.steer > 0.0

    def test_missing_lidar_returns_zero(self):
        controller = FollowTheGapController(setting=_pp_settings())
        ego = EgoState(x=0.0, y=0.0, theta=0.0, velocity=0.0)
        cmd = controller.control(ego, sensors=None)
        assert cmd.steer == 0.0
        assert cmd.acceleration == 0.0

    def test_empty_lidar_returns_zero(self):
        controller = FollowTheGapController(setting=_pp_settings())
        ego = EgoState(x=0.0, y=0.0, theta=0.0, velocity=0.0)
        cmd = controller.control(ego, sensors=SensorFrame(lidar=np.empty((0, 4))))
        assert cmd.steer == 0.0
        assert cmd.acceleration == 0.0

    def test_cruise_velocity_without_plan(self):
        setting = _pp_settings(
            c35_lookahead_distance=8.0,
            c35_cruise_velocity=4.0,
            c35_valpha=1.0,
        )
        controller = FollowTheGapController(setting=setting)
        ego = EgoState(x=0.0, y=0.0, theta=0.0, velocity=0.0)
        # Symmetric forward returns so a gap mid-bearing exists near straight ahead.
        angles = np.array([-0.4, -0.2, 0.2, 0.4])
        ranges = np.full_like(angles, 5.0, dtype=float)
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        lidar = np.column_stack([xs, ys, np.zeros_like(xs), np.ones_like(xs)]).astype(np.float32)
        cmd = controller.control(ego, sensors=SensorFrame(lidar=lidar))
        # Below cruise → positive acceleration from P-term.
        assert cmd.acceleration > 0.0
