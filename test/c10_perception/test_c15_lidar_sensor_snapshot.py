"""Detection reads the selected lidar's points and matching mount."""

import numpy as np
import pytest

from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
from avlite.c10_perception.c15_perception_algs import FastBEVLidarDetection
from avlite.c50_common.c52_world_sensor_datatypes import Lidar, SensorFrame


def _cloud():
    return np.array([[3, -1, 0, 7], [3, 1, 0, 7], [5, 1, 0, 7], [5, -1, 0, 7]], dtype=np.float32)


def test_detection_uses_selected_lidar_and_its_mount():
    mount = np.eye(4)
    mount[0, 3] = 2.0
    lidar = Lidar(points=_cloud(), base_to_sensor=mount)
    frame = SensorFrame(
        lidars={"unused": Lidar(points=np.full((4, 4), 999, dtype=np.float32)), "top": lidar},
        primary_lidar_name="top",
    )
    ego = EgoState(x=10.0, y=20.0, theta=np.pi / 2)
    pm = PerceptionModel(ego_vehicle=ego)
    result = FastBEVLidarDetection(mu=3.0).detect(pm, frame)
    assert result is pm
    assert len(pm.agent_vehicles) == 1
    np.testing.assert_allclose(pm.detection_clusters, lidar.to_map(lidar.points, ego)[:, :2])
    np.testing.assert_array_equal(lidar.points, _cloud())


@pytest.mark.parametrize("frame", [
    None,
    SensorFrame(),
    SensorFrame(lidars={"top": Lidar()}, primary_lidar_name="top"),
    SensorFrame(lidars={"top": Lidar(points=np.empty((0, 4)))}, primary_lidar_name="top"),
    SensorFrame(lidars={"unselected": Lidar(points=_cloud())}),
])
def test_detection_handles_missing_primary_or_reading(frame):
    pm = PerceptionModel(ego_vehicle=EgoState())
    pm.detection_clusters = np.ones((1, 2))
    FastBEVLidarDetection().detect(pm, frame)
    assert pm.detection_clusters is None


def test_direct_cloud_argument_remains_supported():
    pm = PerceptionModel(ego_vehicle=EgoState(x=10.0))
    FastBEVLidarDetection(mu=3.0).detect(pm, lidar_data=_cloud())
    np.testing.assert_allclose(pm.detection_clusters, _cloud()[:, :2] + [10.0, 0.0])
