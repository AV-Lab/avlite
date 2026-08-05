"""Regression tests for FastBEVLidarDetection segmentation and MBR fitting."""

from __future__ import annotations

import numpy as np
import pytest

from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
from avlite.c10_perception.c15_perception_algs import FastBEVLidarDetection


def _rectangle_cluster(cx: float, cy: float, length: float = 2.0, width: float = 2.0, n: int = 8):
    """Axis-aligned rectangle outline in contiguous scan order (CCW from near face)."""
    x0, x1 = cx - length / 2, cx + length / 2
    y0, y1 = cy - width / 2, cy + width / 2
    pts = []
    for y in np.linspace(y0, y1, n):
        pts.append((x0, y))
    for x in np.linspace(x0, x1, n)[1:]:
        pts.append((x, y1))
    for y in np.linspace(y1, y0, n)[1:]:
        pts.append((x1, y))
    for x in np.linspace(x1, x0, n)[1:-1]:
        pts.append((x, y0))
    return np.asarray(pts, dtype=float)


def test_fast_bev_detect_requires_perception_model():
    with pytest.raises(ValueError, match="perception_model is required"):
        FastBEVLidarDetection().detect(perception_model=None, lidar_data=np.zeros((2, 2)))


def test_fast_bev_empty_or_none_lidar_clears_clusters():
    det = FastBEVLidarDetection()
    pm = PerceptionModel(ego_vehicle=EgoState())
    pm.detection_clusters = np.zeros((3, 2))
    pm.agent_vehicles = []

    out = det.detect(pm, lidar_data=None)
    assert out.detection_clusters is None

    pm.detection_clusters = np.zeros((3, 2))
    out = det.detect(pm, lidar_data=np.empty((0, 2)))
    assert out.detection_clusters is None


def test_fast_bev_gap_splits_clusters_and_fits_boxes():
    det = FastBEVLidarDetection(mu=0.5, delta_min=1.0, delta_max=6.0)
    c1 = _rectangle_cluster(9.0, 0.0, length=2.0, width=2.0)
    c2 = _rectangle_cluster(21.0, 0.0, length=2.0, width=2.0)
    assert np.linalg.norm(c2[0] - c1[-1]) > det._mu

    pm = PerceptionModel(ego_vehicle=EgoState())
    det.detect(pm, lidar_data=np.vstack([c1, c2]))

    assert len(pm.agent_vehicles) == 2
    xs = sorted(a.x for a in pm.agent_vehicles)
    np.testing.assert_allclose(xs, [9.0, 21.0], atol=0.15)
    for agent in pm.agent_vehicles:
        assert agent.length == pytest.approx(2.0, abs=0.2)
        assert agent.width == pytest.approx(2.0, abs=0.2)


def test_fast_bev_drops_clusters_outside_diagonal_range():
    det = FastBEVLidarDetection(mu=0.5, delta_min=1.0, delta_max=6.0)
    # Tiny cluster: axis-aligned span diagonal << delta_min.
    tiny = np.array([[5.0, 0.0], [5.1, 0.0], [5.2, 0.05]], dtype=float)
    diag = float(np.linalg.norm(tiny.max(axis=0) - tiny.min(axis=0)))
    assert diag < det._delta_min

    pm = PerceptionModel(ego_vehicle=EgoState())
    det.detect(pm, lidar_data=tiny)
    assert pm.agent_vehicles == []
    assert pm.detection_clusters is None


def test_fast_bev_edge_on_cluster_uses_default_box_pushed_from_ego():
    """Collinear face collapses width → default L/W box pushed along ego heading."""
    det = FastBEVLidarDetection(
        mu=0.5,
        delta_min=0.5,
        delta_max=6.0,
        min_length=0.5,
        min_width=0.5,
        default_length=4.5,
        default_width=2.0,
    )
    # Vertical line at x=10 (only the near face visible).
    line = np.array([[10.0, y] for y in np.linspace(-1.0, 1.0, 12)], dtype=float)
    pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0))
    det.detect(pm, lidar_data=line)

    assert len(pm.agent_vehicles) == 1
    agent = pm.agent_vehicles[0]
    assert agent.length == pytest.approx(4.5)
    assert agent.width == pytest.approx(2.0)
    # Centre pushed away from ego by length/2 along +x.
    assert agent.x == pytest.approx(10.0 + 4.5 / 2.0, abs=1e-6)
    assert agent.y == pytest.approx(0.0, abs=1e-6)


def test_fast_bev_3d_z_band_filters_points():
    det = FastBEVLidarDetection(z_min=-1.5, z_max=0.5, mu=0.5, delta_min=1.0, delta_max=6.0)
    cluster = _rectangle_cluster(8.0, 0.0, length=2.0, width=2.0)
    # All points outside the z-band → cleared.
    lidar_high = np.column_stack([cluster, np.full(len(cluster), 2.0)])
    pm = PerceptionModel(ego_vehicle=EgoState())
    det.detect(pm, lidar_data=lidar_high)
    assert pm.agent_vehicles == []
    assert pm.detection_clusters is None

    lidar_ok = np.column_stack([cluster, np.full(len(cluster), 0.0)])
    det.detect(pm, lidar_data=lidar_ok)
    assert len(pm.agent_vehicles) == 1
    assert pm.agent_vehicles[0].x == pytest.approx(8.0, abs=0.15)
