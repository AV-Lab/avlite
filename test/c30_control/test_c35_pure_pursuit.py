"""Unit tests for Pure Pursuit and Follow the Gap (avlite.c30_control.c35_pure_pursuit)."""

import math

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
        "c35_bubble_radius": 1.0,
        "c35_min_gap_width": 0.2,
        "c32_ego_distance_front_axle": 2.5,
        "c32_ego_max_steering": 0.7,
        "c32_ego_min_steering": -0.7,
    }
    defaults.update(overrides)
    return ControlSettingsSchema(**defaults)


def _lidar_at_angles(
    angles: np.ndarray,
    ranges: float | np.ndarray = 5.0,
    ego: EgoState | None = None,
) -> np.ndarray:
    """(N, 4) LiDAR hits: ego-frame bearings, expressed in world frame for ``ego``."""
    angles = np.asarray(angles, dtype=float)
    ranges = np.full_like(angles, ranges, dtype=float) if np.isscalar(ranges) else np.asarray(ranges, dtype=float)
    ex = ranges * np.cos(angles)
    ey = ranges * np.sin(angles)
    if ego is None:
        xs, ys = ex, ey
    else:
        c, s_th = np.cos(ego.theta), np.sin(ego.theta)
        xs = ego.x + c * ex - s_th * ey
        ys = ego.y + s_th * ex + c * ey
    return np.column_stack([xs, ys, np.zeros_like(xs), np.ones_like(xs)]).astype(np.float32)


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

    def test_closed_loop_midlap_lookahead_stays_local(self):
        """Lookahead must not jump to the start/finish when path_s is a closed lap."""
        path = [(0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0), (0.0, 0.0)]
        tj = TrajectoryTracker(path, velocity=[5.0] * len(path))
        controller = PurePursuitController(
            tj=tj, setting=_pp_settings(c35_lookahead_distance=10.0)
        )
        # Mid-segment on the far side of the rectangle, heading +y.
        ego = EgoState(x=50.0, y=25.0, theta=np.pi / 2, velocity=5.0)
        target = controller.find_path_lookahead(ego, ld=10.0)
        assert target is not None
        # Ego-frame target should be roughly ahead (~Ld forward), not yanked to start/finish.
        ex, ey = target
        assert ex == pytest.approx(10.0, abs=1.5)
        assert abs(ey) < 2.0
        # World lookahead must stay near the ego, not at the duplicated origin.
        s, _ = tj.convert_xy_to_sd(ego.x, ego.y)
        gx, gy = tj.convert_sd_to_xy(min(s + 10.0, float(np.max(tj.path_s))), 0.0)
        assert math.hypot(gx - path[0][0], gy - path[0][1]) > 20.0
        assert math.hypot(gx - ego.x, gy - ego.y) == pytest.approx(10.0, abs=1.5)


class TestFollowTheGapController:
    def test_largest_gap_aims_into_opening(self):
        """Dense right wall, sparse left wall → widest interior gap steers left."""
        setting = _pp_settings(c35_lookahead_distance=8.0)
        controller = FollowTheGapController(setting=setting)
        ego = EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0)
        # Interior gap between right cluster and a left wall return is the opening.
        right = np.linspace(-np.pi / 2 + 0.05, -0.2, 20)
        left = np.array([0.9])
        lidar = _lidar_at_angles(np.concatenate([right, left]))
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
        lidar = _lidar_at_angles(np.array([-0.4, -0.2, 0.2, 0.4]))
        cmd = controller.control(ego, sensors=SensorFrame(lidar=lidar))
        # Below cruise → positive acceleration from P-term.
        assert cmd.acceleration > 0.0

    def test_corridor_with_path_stays_near_center(self):
        """Symmetric corridor walls + straight path → near-zero steer."""
        trajectory = _straight_path()
        controller = FollowTheGapController(tj=trajectory, setting=_pp_settings())
        ego = EgoState(x=20.0, y=0.0, theta=0.0, velocity=5.0)
        # Parallel walls at ±0.5 rad (corridor opening centered ahead).
        right = np.linspace(-1.0, -0.5, 8)
        left = np.linspace(0.5, 1.0, 8)
        lidar = _lidar_at_angles(np.concatenate([right, left]), ego=ego)
        cmd = controller.control(ego, sensors=SensorFrame(lidar=lidar))
        assert cmd.steer == pytest.approx(0.0, abs=0.1)

    def test_path_bias_prefers_path_aligned_gap(self):
        """Wider side opening ignored when path aims straight ahead."""
        trajectory = _straight_path()
        controller = FollowTheGapController(tj=trajectory, setting=_pp_settings())
        ego = EgoState(x=20.0, y=0.0, theta=0.0, velocity=5.0)
        # Narrow gap ahead (~0 bearing) and a much wider gap on the left.
        angles = np.array([-0.8, -0.15, 0.15, 0.35, 1.2])
        lidar = _lidar_at_angles(angles, ego=ego)
        cmd = controller.control(ego, sensors=SensorFrame(lidar=lidar))
        # Path-biased pick should stay near center, not yank hard left into the wide gap.
        assert abs(cmd.steer) < 0.25
        # Without a path, widest interior gap is the left opening → positive steer.
        no_path = FollowTheGapController(tj=None, setting=_pp_settings())
        cmd_wide = no_path.control(ego, sensors=SensorFrame(lidar=lidar))
        assert cmd_wide.steer > cmd.steer
        assert cmd_wide.steer > 0.0

    def test_z_band_filter_drops_out_of_range_hits(self):
        """3D returns outside c35_lidar_z_* must not contribute to gap selection."""
        setting = _pp_settings(c35_lidar_z_min=-1.5, c35_lidar_z_max=2.0)
        controller = FollowTheGapController(setting=setting)
        ego = EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0)
        # In-band interior gap slightly left of center; out-of-band wall on the right.
        in_band = _lidar_at_angles(np.array([-0.15, 0.35]))
        in_band[:, 2] = 0.0
        out_band = _lidar_at_angles(np.linspace(-1.0, -0.2, 12))
        out_band[:, 2] = 8.0
        cmd = controller.control(ego, sensors=SensorFrame(lidar=np.vstack([in_band, out_band])))
        assert cmd.steer > 0.0

    def test_all_points_outside_z_band_returns_zero(self):
        controller = FollowTheGapController(setting=_pp_settings())
        ego = EgoState(x=0.0, y=0.0, theta=0.0, velocity=0.0)
        lidar = _lidar_at_angles(np.array([-0.3, 0.0, 0.3]))
        lidar[:, 2] = 10.0
        cmd = controller.control(ego, sensors=SensorFrame(lidar=lidar))
        assert cmd.steer == 0.0
        assert cmd.acceleration == 0.0

    def test_safety_bubble_ignores_near_returns(self):
        setting = _pp_settings(c35_bubble_radius=1.5, c35_lookahead_distance=8.0)
        controller = FollowTheGapController(setting=setting)
        ego = EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0)
        # Close-in clutter at 0.2 m would otherwise dominate bearings.
        near = _lidar_at_angles(np.array([-0.8, 0.8]), ranges=0.2)
        far = _lidar_at_angles(np.array([-0.2, 0.2]), ranges=6.0)
        cmd = controller.control(ego, sensors=SensorFrame(lidar=np.vstack([near, far])))
        assert cmd.steer == pytest.approx(0.0, abs=0.1)

    def test_only_bubble_interior_hits_returns_zero(self):
        setting = _pp_settings(c35_bubble_radius=2.0)
        controller = FollowTheGapController(setting=setting)
        ego = EgoState(x=0.0, y=0.0, theta=0.0, velocity=0.0)
        lidar = _lidar_at_angles(np.array([-0.4, 0.0, 0.4]), ranges=0.5)
        cmd = controller.control(ego, sensors=SensorFrame(lidar=lidar))
        assert cmd.steer == 0.0
        assert cmd.acceleration == 0.0

    def test_prefers_interior_gap_over_wider_edge(self):
        """±90° edge openings are wider than a narrow corridor; interior must win."""
        controller = FollowTheGapController(setting=_pp_settings())
        ego = EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0)
        # Two returns straddle center: interior gap ~0.3 rad, each edge ~1.4 rad.
        lidar = _lidar_at_angles(np.array([-0.15, 0.15]))
        cmd = controller.control(ego, sensors=SensorFrame(lidar=lidar))
        assert cmd.steer == pytest.approx(0.0, abs=0.08)

    def test_no_forward_returns_command_zero(self):
        controller = FollowTheGapController(setting=_pp_settings())
        ego = EgoState(x=0.0, y=0.0, theta=0.0, velocity=0.0)
        lidar = _lidar_at_angles(np.array([np.pi * 0.8, np.pi, -np.pi * 0.8]))
        cmd = controller.control(ego, sensors=SensorFrame(lidar=lidar))
        assert cmd.steer == 0.0
        assert cmd.acceleration == 0.0