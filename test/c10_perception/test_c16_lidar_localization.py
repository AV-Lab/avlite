"""LidarLocalization consumes sensor-frame scans and recovers ego motion via ICP."""

import numpy as np
import pytest

from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
from avlite.c10_perception.c16_localization_algs import LidarLocalization
from avlite.c50_common.c52_world_sensor_datatypes import Lidar, SensorFrame


def _corridor_map() -> np.ndarray:
    """Closed room: walls at x = ±6 and y = ±4 constrain x, y and heading.

    Dense (5 cm) sampling keeps point-to-point ICP from snapping to the lattice.
    """
    xs = np.linspace(-6.0, 6.0, 241)
    ys = np.linspace(-4.0, 4.0, 161)
    walls = [
        np.column_stack([xs, np.full_like(xs, 4.0)]),
        np.column_stack([xs, np.full_like(xs, -4.0)]),
        np.column_stack([np.full_like(ys, 6.0), ys]),
        np.column_stack([np.full_like(ys, -6.0), ys]),
    ]
    return np.vstack(walls)


def _scan_from(pose: tuple[float, float, float], world_pts: np.ndarray) -> np.ndarray:
    """Express world points in the ego body frame at ``pose`` as an (N, 4) cloud."""
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)
    dx, dy = world_pts[:, 0] - x, world_pts[:, 1] - y
    ex = c * dx + s * dy
    ey = -s * dx + c * dy
    return np.column_stack([ex, ey, np.zeros_like(ex), np.zeros_like(ex)]).astype(np.float32)


def test_state_pose_matrix_is_body_pose_in_map():
    state = EgoState(x=10.0, y=5.0, z=0.7, theta=np.pi / 2)
    T = state.pose_matrix()
    assert T.shape == (4, 4)
    # Body +x (heading) maps to map +y; translation is (x, y, z).
    np.testing.assert_allclose(T @ [1.0, 0.0, 0.0, 1.0], [10.0, 6.0, 0.7, 1.0], atol=1e-9)
    np.testing.assert_allclose(T @ [0.0, 0.0, 1.0, 1.0], [10.0, 5.0, 1.7, 1.0], atol=1e-9)
    np.testing.assert_allclose(T[3], [0.0, 0.0, 0.0, 1.0])


def test_icp_recovers_ego_motion_from_body_frame_scans():
    world = _corridor_map()
    ego = EgoState(x=0.0, y=0.0, theta=0.0)
    loc = LidarLocalization(PerceptionModel(ego_vehicle=ego))

    # Seed: first scan taken at the true start pose builds the reference map.
    loc.localize(sensors=SensorFrame(lidar=_scan_from((0.0, 0.0, 0.0), world)))
    np.testing.assert_allclose(loc._map[:, :2], world, atol=1e-4)

    # Ego moves; the stack pose is stale but ICP must recover the true pose.
    true_pose = (0.6, 0.3, 0.05)
    loc.localize(sensors=SensorFrame(lidar=_scan_from(true_pose, world)))
    assert ego.x == pytest.approx(true_pose[0], abs=0.05)
    assert ego.y == pytest.approx(true_pose[1], abs=0.05)
    assert ego.theta == pytest.approx(true_pose[2], abs=0.01)


def test_lidar_mount_is_applied_before_alignment():
    world = _corridor_map()
    ego = EgoState(x=0.0, y=0.0, theta=0.0)
    loc = LidarLocalization(PerceptionModel(ego_vehicle=ego))

    # Lidar mounted 1 m ahead of the body origin: sensor-frame scans are shifted by -1 in x.
    mount = np.eye(4)
    mount[0, 3] = 1.0
    params = Lidar(base_to_sensor=mount)

    def sensor_scan(pose):
        body = _scan_from(pose, world)
        body[:, 0] -= 1.0
        return body

    loc.localize(sensors=SensorFrame(lidar=sensor_scan((0.0, 0.0, 0.0)), lidar_sensor=params))
    np.testing.assert_allclose(loc._map[:, :2], world, atol=1e-4)

    loc.localize(sensors=SensorFrame(lidar=sensor_scan((0.5, -0.2, 0.0)), lidar_sensor=params))
    assert ego.x == pytest.approx(0.5, abs=0.05)
    assert ego.y == pytest.approx(-0.2, abs=0.05)
