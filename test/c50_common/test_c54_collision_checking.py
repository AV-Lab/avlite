"""Unit tests for collision checking (avlite.c60_common.c64_collision_checking).

Tests verify:
- Static obstacles block trajectories that intersect their footprint.
- Clear trajectories report no collision.
- precompute_obstacle_polygons returns one polygon per agent.
"""

import numpy as np

from avlite.c10_perception.c11_perception_model import AgentState, EgoState, PerceptionModel
from avlite.c60_common.c63_trajectory_tracker import TrajectoryTracker
from avlite.c60_common.c64_collision_checking import check_collision, precompute_obstacle_polygons


def _straight_trajectory(x_start: float, x_end: float, n: int = 20) -> TrajectoryTracker:
    xs = [x_start + (x_end - x_start) * i / (n - 1) for i in range(n)]
    path = [(x, 0.0) for x in xs]
    return TrajectoryTracker(path=path, velocity=[5.0] * n)


class TestCheckCollision:
    def test_clear_path_reports_no_collision(self):
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[AgentState(x=50.0, y=20.0, theta=0.0, velocity=0.0, agent_id=1)],
        )
        trajectory = _straight_trajectory(0.0, 100.0)
        hit, idx, _vel = check_collision(pm, trajectory)
        assert hit is False
        assert idx == -1

    def test_intersecting_static_agent_is_detected(self):
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[AgentState(x=50.0, y=0.0, theta=0.0, velocity=0.0, agent_id=1)],
        )
        trajectory = _straight_trajectory(0.0, 100.0)
        hit, idx, _vel = check_collision(pm, trajectory)
        assert hit is True
        assert idx >= 0

    def test_precomputed_polygons_match_slow_path(self):
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[AgentState(x=50.0, y=0.0, theta=0.0, velocity=0.0, agent_id=1)],
        )
        trajectory = _straight_trajectory(0.0, 100.0)
        polygons = precompute_obstacle_polygons(pm, total_time=2.0)
        assert len(polygons) == 1
        hit_fast, idx_fast, _ = check_collision(pm, trajectory, obstacle_polygons=polygons)
        hit_slow, idx_slow, _ = check_collision(pm, trajectory)
        assert hit_fast == hit_slow
        assert idx_fast == idx_slow
