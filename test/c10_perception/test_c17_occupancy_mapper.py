"""OccupancyMapper rasterization and sensor-frame contract."""

import json
from dataclasses import replace

import numpy as np
import pytest

from avlite.c10_perception.c11_perception_model import (
    EgoState,
    Map,
    OccupancyMap,
    PerceptionModel,
    RaceMap,
)
from avlite.c10_perception.c17_mapping_algs import OccupancyMapper
from avlite.c10_perception.c19_settings import PerceptionSettingsSchema
from avlite.c50_common.c51_capabilities import StackCapability
from avlite.c50_common.c52_world_sensor_datatypes import Lidar, SensorFrame

_FINE = PerceptionSettingsSchema(c17_resolution=1.0, c17_size=20.0)


def _prob_at(om, x: float, y: float) -> float:
    col = int(np.floor((x - om.origin_x) / om.resolution))
    row = int(np.floor((y - om.origin_y) / om.resolution))
    return float(om.grid[row, col])


def _frame(points, sensor=None):
    return SensorFrame(
        lidars={"top": replace(sensor or Lidar(), points=points)},
        primary_lidar_name="top",
    )


def _step(mapper, ego, lidar=None, lidar_sensor=None):
    pm = PerceptionModel(ego_vehicle=ego)
    cloud = np.empty((0, 4), dtype=np.float32) if lidar is None else lidar
    mapper.update(pm, _frame(cloud, lidar_sensor))
    return pm


def test_hit_occupied_and_ray_free():
    pm = _step(
        OccupancyMapper(setting=_FINE),
        EgoState(),
        np.array([[5.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    om = pm.occupancy_map
    assert _prob_at(om, 5.0, 0.0) > 0.5
    assert _prob_at(om, 2.0, 0.0) < 0.5
    assert _prob_at(om, 0.0, 5.0) == pytest.approx(0.5, abs=1e-5)


def test_empty_lidar_stays_unknown():
    pm = _step(OccupancyMapper(setting=_FINE), EgoState())
    assert pm.occupancy_map.grid.shape == (20, 20)
    np.testing.assert_allclose(pm.occupancy_map.grid, 0.5, atol=1e-5)


def test_expand_keeps_world_occupied_cell():
    mapper = OccupancyMapper(setting=_FINE)
    first = _step(mapper, EgoState(), np.array([[3.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    initial_shape = first.occupancy_map.grid.shape
    pm = _step(mapper, EgoState(x=30.0, y=0.0))
    assert _prob_at(pm.occupancy_map, 3.0, 0.0) > 0.5
    assert pm.occupancy_map.grid.shape[1] > initial_shape[1]


def test_reset_clears_log_odds():
    mapper = OccupancyMapper(setting=_FINE)
    _step(mapper, EgoState(), np.array([[5.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    mapper.reset()
    pm = _step(mapper, EgoState())
    np.testing.assert_allclose(pm.occupancy_map.grid, 0.5, atol=1e-5)
    assert pm.occupancy_map.grid.shape == (20, 20)


def test_update_uses_passed_ego_not_stale_pm_heading():
    """Lidar must be transformed with the pose it was taken at, not stale pm.ego."""
    mapper = OccupancyMapper(setting=_FINE)
    stale = EgoState(theta=np.pi / 2)
    matching = EgoState(theta=0.0)
    pm = PerceptionModel(ego_vehicle=stale)
    cloud = np.array([[5.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    mapper.update(pm, _frame(cloud), ego=matching)
    om = pm.occupancy_map
    assert _prob_at(om, 5.0, 0.0) > 0.5
    assert _prob_at(om, 0.0, 5.0) == pytest.approx(0.5, abs=1e-5)

    mapper.reset()
    pm_stale = PerceptionModel(ego_vehicle=stale)
    mapper.update(pm_stale, _frame(cloud))
    assert _prob_at(pm_stale.occupancy_map, 0.0, 5.0) > 0.5


def test_mapper_applies_lidar_mount():
    mount = np.eye(4)
    mount[0, 3] = 1.0
    # World hit (5, 0): body (5, 0), sensor frame (4, 0) with +1 m x mount.
    pm = _step(
        OccupancyMapper(setting=_FINE),
        EgoState(),
        np.array([[4.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        Lidar(base_to_sensor=mount),
    )
    assert _prob_at(pm.occupancy_map, 5.0, 0.0) > 0.5


def test_mapper_empty_lidar_is_unknown():
    pm = _step(
        OccupancyMapper(setting=_FINE),
        EgoState(),
        np.empty((0, 4), dtype=np.float32),
    )
    np.testing.assert_allclose(pm.occupancy_map.grid, 0.5, atol=1e-5)


def test_mapper_reset_clears_model_and_grid():
    mapper = OccupancyMapper(setting=_FINE)
    pm = PerceptionModel(ego_vehicle=EgoState())
    mapper.update(pm, _frame(np.array([[5.0, 0.0, 0.0, 0.0]], dtype=np.float32)))
    mapper.reset()
    pm.reset()
    assert pm.occupancy_map is None
    mapper.update(pm, _frame(np.empty((0, 4), dtype=np.float32)))
    np.testing.assert_allclose(pm.occupancy_map.grid, 0.5, atol=1e-5)


def test_mapper_capabilities_with_and_without_map():
    assert OccupancyMapper.stack_capabilities == frozenset({StackCapability.MAP_OCCUPANCY})
    assert OccupancyMapper().stack_capabilities == frozenset({StackCapability.MAP_OCCUPANCY})
    race = RaceMap(
        source_path="synthetic",
        left_bound=np.array([[0.0, 1.0], [10.0, 1.0]]),
        right_bound=np.array([[0.0, -1.0], [10.0, -1.0]]),
    )
    mapper = OccupancyMapper(map=race)
    assert mapper.stack_capabilities == frozenset({StackCapability.MAP_OCCUPANCY})


def test_occupancy_map_is_a_map():
    pm = _step(OccupancyMapper(setting=_FINE), EgoState())
    assert isinstance(pm.occupancy_map, Map)
    assert isinstance(pm.occupancy_map, OccupancyMap)


def test_occupancy_map_json_roundtrip(tmp_path):
    pm = _step(
        OccupancyMapper(setting=_FINE),
        EgoState(),
        np.array([[5.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    om = pm.occupancy_map
    om._reference_point = (24.5, 54.6)
    path = tmp_path / "occ.json"
    om.to_file(path)
    loaded = OccupancyMap.from_path(path)
    np.testing.assert_allclose(loaded.grid, om.grid, atol=1e-5)
    assert loaded.origin_x == pytest.approx(om.origin_x)
    assert loaded.origin_y == pytest.approx(om.origin_y)
    assert loaded.resolution == pytest.approx(om.resolution)
    assert loaded.reference_point == (24.5, 54.6)
    assert loaded.source_path == str(path)


def test_map_open_sniffs_occupancy_not_race(tmp_path):
    om = OccupancyMap(
        grid=np.full((3, 3), 0.7, dtype=np.float32),
        origin_x=-1.0,
        origin_y=-1.0,
        resolution=0.5,
    )
    occ_path = tmp_path / "occ.json"
    om.to_file(occ_path)
    race_path = tmp_path / "race.json"
    race_path.write_text(json.dumps({
        "LeftBound": [[0.0, 1.0], [10.0, 1.0]],
        "RightBound": [[0.0, -1.0], [10.0, -1.0]],
        "ReferencePoint": [1.0, 2.0],
    }), encoding="utf-8")

    assert OccupancyMap.is_loadable(occ_path)
    assert not OccupancyMap.is_loadable(race_path)
    assert RaceMap.is_loadable(race_path)
    assert not RaceMap.is_loadable(occ_path)
    opened = Map.open(occ_path)
    assert isinstance(opened, OccupancyMap)


def test_mapper_seeds_from_loaded_occupancy_map(tmp_path):
    built = _step(
        OccupancyMapper(setting=_FINE),
        EgoState(),
        np.array([[5.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    ).occupancy_map
    path = tmp_path / "occ.json"
    built.to_file(path)
    loaded = OccupancyMap.from_path(path)
    mapper = OccupancyMapper(map=loaded, setting=_FINE)
    pm = _step(mapper, EgoState())
    assert _prob_at(pm.occupancy_map, 5.0, 0.0) > 0.5
@pytest.mark.parametrize("theta", [0.0, np.pi / 2])
def test_rays_start_at_selected_lidar_not_vehicle(theta):
    mount = np.eye(4)
    mount[0, 3] = 2.0
    lidar = Lidar(base_to_sensor=mount, points=np.array([[3, 0, 0, 0]], dtype=np.float32))
    frame = SensorFrame(
        lidars={"unused": Lidar(), "front": lidar}, primary_lidar_name="front",
    )
    ego = EgoState(x=10.0, y=20.0, theta=theta)
    pm = PerceptionModel(ego_vehicle=ego)
    OccupancyMapper(setting=_FINE).update(pm, frame)
    # Use cell-interior coordinates to avoid floating-point grid boundaries.
    c, s = np.cos(theta), np.sin(theta)
    assert _prob_at(pm.occupancy_map, 10.1 + 5*c, 20.1 + 5*s) > 0.5
    assert _prob_at(pm.occupancy_map, 10.1 + 3*c, 20.1 + 3*s) < 0.5
    assert _prob_at(pm.occupancy_map, 10.1 + c, 20.1 + s) == pytest.approx(0.5)


@pytest.mark.parametrize("frame", [None, SensorFrame(), _frame(None)])
def test_mapper_handles_missing_reading(frame):
    pm = PerceptionModel(ego_vehicle=EgoState())
    OccupancyMapper(setting=_FINE).update(pm, frame)
    np.testing.assert_allclose(pm.occupancy_map.grid, 0.5, atol=1e-5)


def test_missing_reading_preserves_mount_for_map_bounds():
    mount = np.eye(4)
    mount[0, 3] = 30.0
    pm = PerceptionModel(ego_vehicle=EgoState())
    OccupancyMapper(setting=_FINE).update(pm, _frame(None, Lidar(base_to_sensor=mount)))
    assert pm.occupancy_map.grid.shape[1] > 20
    assert _prob_at(pm.occupancy_map, 30.0, 0.0) == pytest.approx(0.5)
