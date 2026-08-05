"""Regression tests for BasicSim LiDAR raycasting hits."""

from __future__ import annotations

import numpy as np
import pytest

from avlite.c10_perception.c11_perception_model import (
    AgentState,
    EgoState,
    PerceptionModel,
    RaceMap,
)
from avlite.c40_execution.c46_basic_sim import BasicSim
from avlite.c40_execution.c49_settings import ExecutionSettingsSchema


def _sim_with_settings(**lidar_kwargs) -> BasicSim:
    setting = ExecutionSettingsSchema()
    for key, value in lidar_kwargs.items():
        setattr(setting, key, value)
    ego = EgoState(x=0.0, y=0.0, theta=0.0, velocity=0.0)
    return BasicSim(ego_state=ego, pm=PerceptionModel(ego_vehicle=ego), setting=setting)


def test_lidar_empty_scene_returns_empty_cloud():
    sim = _sim_with_settings(
        c46_lidar_num_beams=36,
        c46_lidar_fov_deg=360.0,
        c46_lidar_range=50.0,
    )
    pts = sim._simulate_lidar_2d()
    cloud = sim.get_lidar_data()
    assert pts.shape == (0, 2)
    assert cloud.shape == (0, 4)


def test_lidar_hits_nearest_boundary_wall_ahead():
    """Forward beams should hit a wall at x=10, not a farther wall behind."""
    left = np.array([[10.0, -5.0], [10.0, 5.0]])
    right = np.array([[-1.0, -5.0], [-1.0, 5.0]])
    race = RaceMap(source_path="synthetic", left_bound=left, right_bound=right)
    setting = ExecutionSettingsSchema(
        c46_lidar_num_beams=5,
        c46_lidar_fov_deg=20.0,
        c46_lidar_range=50.0,
    )
    ego = EgoState(x=0.0, y=0.0, theta=0.0)
    sim = BasicSim(
        ego_state=ego,
        pm=PerceptionModel(ego_vehicle=ego),
        map=race,
        setting=setting,
    )

    pts = sim._simulate_lidar_2d()
    cloud = sim.get_lidar_data()

    assert len(pts) == 5
    np.testing.assert_allclose(pts[:, 0], 10.0, atol=1e-6)
    assert np.all(np.abs(pts[:, 1]) < 2.0)
    assert cloud.shape == (5, 4)
    np.testing.assert_allclose(cloud[:, :2], pts, atol=1e-5)
    np.testing.assert_allclose(cloud[:, 2:], 0.0)


def test_lidar_hits_agent_bounding_box_face():
    """Agent BB edges are raycast; the near face of an ahead agent is at x=13."""
    setting = ExecutionSettingsSchema(
        c46_lidar_num_beams=9,
        c46_lidar_fov_deg=40.0,
        c46_lidar_range=50.0,
    )
    ego = EgoState(x=0.0, y=0.0, theta=0.0)
    pm = PerceptionModel(ego_vehicle=ego)
    pm.agent_vehicles = [
        AgentState(x=15.0, y=0.0, theta=0.0, length=4.0, width=2.0, agent_id=1)
    ]
    sim = BasicSim(ego_state=ego, pm=pm, map=None, setting=setting)

    pts = sim._simulate_lidar_2d()
    assert len(pts) >= 1
    # Near face of axis-aligned 4 m box centred at x=15 is x=13.
    assert float(pts[:, 0].min()) == pytest.approx(13.0, abs=1e-6)


def test_lidar_misses_obstacles_beyond_range():
    left = np.array([[10.0, -5.0], [10.0, 5.0]])
    race = RaceMap(
        source_path="synthetic",
        left_bound=left,
        right_bound=np.array([[10.0, -5.0], [10.0, 5.0]]),
    )
    setting = ExecutionSettingsSchema(
        c46_lidar_num_beams=5,
        c46_lidar_fov_deg=20.0,
        c46_lidar_range=5.0,
    )
    ego = EgoState(x=0.0, y=0.0, theta=0.0)
    sim = BasicSim(
        ego_state=ego,
        pm=PerceptionModel(ego_vehicle=ego),
        map=race,
        setting=setting,
    )
    assert sim._simulate_lidar_2d().shape == (0, 2)


def test_lidar_prefers_nearest_of_overlapping_hits():
    """Wall at x=10 must win over an agent whose near face is at x=13."""
    left = np.array([[10.0, -5.0], [10.0, 5.0]])
    race = RaceMap(
        source_path="synthetic",
        left_bound=left,
        right_bound=np.empty((0, 2)),
    )
    setting = ExecutionSettingsSchema(
        c46_lidar_num_beams=5,
        c46_lidar_fov_deg=20.0,
        c46_lidar_range=50.0,
    )
    ego = EgoState(x=0.0, y=0.0, theta=0.0)
    pm = PerceptionModel(ego_vehicle=ego)
    pm.agent_vehicles = [
        AgentState(x=15.0, y=0.0, theta=0.0, length=4.0, width=2.0, agent_id=1)
    ]
    sim = BasicSim(ego_state=ego, pm=pm, map=race, setting=setting)

    pts = sim._simulate_lidar_2d()
    assert len(pts) == 5
    np.testing.assert_allclose(pts[:, 0], 10.0, atol=1e-6)
