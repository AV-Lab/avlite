"""Regression tests for LidarLocalization ICP scan-to-map pose updates."""

import math

import numpy as np
import pytest

from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
from avlite.c10_perception.c16_localization_algs import LidarLocalization
from avlite.c10_perception.c19_settings import PerceptionSettingsSchema
from avlite.c50_common.c52_world_sensor_datatypes import SensorFrame


def _loc(ego: EgoState | None = None) -> tuple[LidarLocalization, PerceptionModel]:
    ego = ego or EgoState(x=0.0, y=0.0, theta=0.0, velocity=0.0)
    pm = PerceptionModel(ego_vehicle=ego)
    setting = PerceptionSettingsSchema()
    return LidarLocalization(pm, setting=setting), pm


def _asymmetric_map(n: int = 40, seed: int = 0) -> np.ndarray:
    """Non-collinear cloud so translation and yaw are uniquely recoverable."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 2)) * np.array([3.0, 2.0]) + np.array([8.0, 1.0])


def test_first_scan_seeds_map_without_mutating_ego():
    ego = EgoState(x=1.5, y=-0.25, theta=0.1, velocity=0.0)
    loc, _ = _loc(ego)
    scan = _asymmetric_map()

    loc.localize(sensors=SensorFrame(lidar=scan))

    assert loc._map is not None
    np.testing.assert_allclose(loc._map, scan)
    assert loc._x == pytest.approx(1.5)
    assert loc._y == pytest.approx(-0.25)
    assert loc._theta == pytest.approx(0.1)
    assert ego.x == pytest.approx(1.5)
    assert ego.y == pytest.approx(-0.25)
    assert ego.theta == pytest.approx(0.1)


def test_reset_clears_map_and_pose_estimate():
    loc, _ = _loc()
    loc.localize(sensors=SensorFrame(lidar=_asymmetric_map()))
    assert loc._map is not None

    loc.reset()

    assert loc._map is None
    assert loc._x is None
    assert loc._y is None
    assert loc._theta is None


def test_pure_translation_recovers_ego_xy():
    """Body-frame scan = map − (dx, dy) recovers ego translation via ICP."""
    dx, dy = 0.4, -0.25
    ego = EgoState(x=0.0, y=0.0, theta=0.0, velocity=0.0)
    loc, _ = _loc(ego)
    map_scan = _asymmetric_map()
    loc.localize(sensors=SensorFrame(lidar=map_scan))

    # Seed estimate near truth so correspondences stay within max distance.
    ego.x, ego.y, ego.theta = 0.1, -0.05, 0.0
    loc._x, loc._y, loc._theta = ego.x, ego.y, ego.theta

    moved = map_scan - np.array([dx, dy])
    loc.localize(sensors=SensorFrame(lidar=moved))

    assert ego.x == pytest.approx(dx, abs=5e-3)
    assert ego.y == pytest.approx(dy, abs=5e-3)
    assert ego.theta == pytest.approx(0.0, abs=1e-2)


def test_pure_yaw_recovers_ego_theta():
    """Body-frame = R(-yaw) @ map recovers heading near +yaw."""
    yaw = math.radians(8.0)
    ego = EgoState(x=0.0, y=0.0, theta=0.0, velocity=0.0)
    loc, _ = _loc(ego)
    map_scan = _asymmetric_map()
    loc.localize(sensors=SensorFrame(lidar=map_scan))

    c, s = math.cos(-yaw), math.sin(-yaw)
    rot = np.array([[c, -s], [s, c]])
    rotated = (rot @ map_scan.T).T

    ego.theta = math.radians(6.0)
    loc._theta = ego.theta
    loc.localize(sensors=SensorFrame(lidar=rotated))

    assert ego.theta == pytest.approx(yaw, abs=math.radians(0.5))
    assert ego.x == pytest.approx(0.0, abs=5e-2)
    assert ego.y == pytest.approx(0.0, abs=5e-2)


@pytest.mark.parametrize(
    "lidar",
    [
        None,
        np.zeros((0, 2)),
        np.array([[1.0, 0.0], [2.0, 0.0]]),  # fewer than 3 points
        np.array([[1.0, 0.0, 10.0], [2.0, 0.0, 11.0], [3.0, 0.0, 12.0]]),  # all outside z-band
    ],
    ids=["none", "empty", "two_points", "z_band_empty"],
)
def test_invalid_or_filtered_scans_are_noops(lidar):
    ego = EgoState(x=3.0, y=4.0, theta=0.5, velocity=1.0)
    loc, _ = _loc(ego)
    before = (ego.x, ego.y, ego.theta, ego.velocity)

    loc.localize(sensors=SensorFrame(lidar=lidar))

    assert (ego.x, ego.y, ego.theta, ego.velocity) == before
    assert loc._map is None
