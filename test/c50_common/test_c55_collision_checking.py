"""Unit tests for collision checking (avlite.c50_common.c55_collision_checking).

Tests verify:
- Static obstacles block trajectories that intersect their footprint.
- Clear trajectories report no collision.
- precompute_obstacle_polygons returns one polygon per agent.
- Ego + obstacle margins combine to ~1 m body-to-body clearance.
- Ego length extension catches front-corner side overlaps.
"""

import numpy as np

from avlite.c10_perception.c11_perception_model import AgentState, EgoState, PerceptionModel
from avlite.c50_common.c54_trajectory_tracker import TrajectoryTracker
from avlite.c50_common.c55_collision_checking import check_collision, precompute_obstacle_polygons


def _straight_trajectory(x_start: float, x_end: float, n: int = 20, y: float = 0.0) -> TrajectoryTracker:
    xs = [x_start + (x_end - x_start) * i / (n - 1) for i in range(n)]
    path = [(x, y) for x in xs]
    return TrajectoryTracker(path=path, velocity=[5.0] * n)


class TestCheckCollision:
    def test_clear_path_reports_no_collision(self):
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[AgentState(x=50.0, y=20.0, theta=0.0, velocity=0.0, agent_id=1)],
        )
        trajectory = _straight_trajectory(0.0, 100.0)
        hit, idx, _vel, clearance = check_collision(pm, trajectory)
        assert hit is False
        assert idx == -1
        assert clearance > 0

    def test_intersecting_static_agent_is_detected(self):
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[AgentState(x=50.0, y=0.0, theta=0.0, velocity=0.0, agent_id=1)],
        )
        trajectory = _straight_trajectory(0.0, 100.0)
        hit, idx, _vel, clearance = check_collision(pm, trajectory)
        assert hit is True
        assert idx >= 0
        assert clearance == 0.0

    def test_precomputed_polygons_match_slow_path(self):
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[AgentState(x=50.0, y=0.0, theta=0.0, velocity=0.0, agent_id=1)],
        )
        trajectory = _straight_trajectory(0.0, 100.0)
        polygons = precompute_obstacle_polygons(pm, total_time=2.0)
        assert len(polygons) == 1
        hit_fast, idx_fast, _, _ = check_collision(pm, trajectory, obstacle_polygons=polygons)
        hit_slow, idx_slow, _, _ = check_collision(pm, trajectory)
        assert hit_fast == hit_slow
        assert idx_fast == idx_slow


class TestMarginClearance:
    """Body gap ≈ collision_safety_margin + obstacle_inflation_margin (both 0.5 → 1.0 m)."""

    _EGO_W = 2.0
    _AGENT_W = 2.0
    _MARGIN = 0.5

    def _pm_and_polys(self, agent_y: float):
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0, width=self._EGO_W),
            agent_vehicles=[
                AgentState(x=50.0, y=agent_y, theta=0.0, velocity=0.0, agent_id=1, width=self._AGENT_W),
            ],
        )
        polys = precompute_obstacle_polygons(
            pm, total_time=1.0, obstacle_inflation_margin=self._MARGIN,
        )
        return pm, polys

    def test_body_gap_under_1m_collides(self):
        # center-to-center needed for touch: ego_w/2 + agent_w/2 + 1.0 = 3.0
        body_gap = 0.8
        agent_y = self._EGO_W / 2 + self._AGENT_W / 2 + body_gap
        pm, polys = self._pm_and_polys(agent_y)
        hit, _, _, clearance = check_collision(
            pm, _straight_trajectory(0.0, 100.0),
            obstacle_polygons=polys,
            collision_safety_margin=self._MARGIN,
        )
        assert hit is True
        assert clearance == 0.0

    def test_body_gap_over_1m_is_clear(self):
        body_gap = 1.2
        agent_y = self._EGO_W / 2 + self._AGENT_W / 2 + body_gap
        pm, polys = self._pm_and_polys(agent_y)
        hit, _, _, clearance = check_collision(
            pm, _straight_trajectory(0.0, 100.0),
            obstacle_polygons=polys,
            collision_safety_margin=self._MARGIN,
        )
        assert hit is False
        assert clearance > 0


class TestEgoLengthExtension:
    def test_front_corner_side_overlap_detected(self):
        # Agent sits just past the last centerline point, beside the ego front bumper.
        # Flat-cap tube ending at the last waypoint would miss this; length extension catches it.
        ego_len, ego_w = 4.5, 2.0
        agent_w = 2.0
        margin = 0.5
        # Lateral: inside the hard 1 m body floor so corridor+inflation must intersect.
        agent_y = ego_w / 2 + agent_w / 2 + 0.5
        # Longitudinal: agent center just ahead of path end, within ego half-length.
        path_end = 50.0
        agent_x = path_end + ego_len / 2 - 0.5

        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0, width=ego_w, length=ego_len),
            agent_vehicles=[
                AgentState(x=agent_x, y=agent_y, theta=0.0, velocity=0.0, agent_id=1, width=agent_w),
            ],
        )
        polys = precompute_obstacle_polygons(pm, total_time=1.0, obstacle_inflation_margin=margin)
        hit, _, _, _ = check_collision(
            pm, _straight_trajectory(0.0, path_end),
            obstacle_polygons=polys,
            collision_safety_margin=margin,
        )
        assert hit is True


class TestSlowPathConstantVelocitySweep:
    """Slow path (no obstacle_polygons) still fabricates a CV sweep for movers.

    This intentionally diverges from precompute_obstacle_polygons, which requires
    a SingleTrajectory prediction before sweeping.
    """

    def test_slow_path_sweeps_mover_across_corridor_without_prediction(self):
        # Agent starts clear of the corridor but drives toward it; CV sweep must hit.
        # Precompute without prediction keeps a static box → clear.
        agent_x, agent_y = 40.0, 8.0
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[
                AgentState(
                    x=agent_x, y=agent_y, theta=-np.pi / 2, velocity=5.0, agent_id=1,
                ),
            ],
        )
        trajectory = _straight_trajectory(0.0, 100.0)
        # path length 100 m @ 5 m/s → total_time ≈ 20 s → predicted y ≈ 8 - 100 = -92
        hit_slow, idx_slow, vel_slow, _ = check_collision(pm, trajectory)
        assert hit_slow is True
        assert idx_slow >= 0
        assert vel_slow == 5.0

        polys = precompute_obstacle_polygons(pm, total_time=20.0)
        hit_fast, _, _, clearance = check_collision(
            pm, trajectory, obstacle_polygons=polys,
        )
        assert hit_fast is False
        assert clearance > 0

    def test_slow_path_static_agent_stays_unswept(self):
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[
                AgentState(x=40.0, y=8.0, theta=-np.pi / 2, velocity=0.0, agent_id=1),
            ],
        )
        hit, idx, _, clearance = check_collision(pm, _straight_trajectory(0.0, 100.0))
        assert hit is False
        assert idx == -1
        assert clearance > 0


class TestDegenerateTrajectoryPoseCheck:
    def test_none_trajectory_uses_current_pose_bbs(self):
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=0.0),
            agent_vehicles=[
                AgentState(x=0.5, y=0.0, theta=0.0, velocity=3.0, agent_id=1),
            ],
        )
        hit, idx, vel, clearance = check_collision(pm, None)
        assert hit is True
        assert idx == 0
        assert vel == 3.0
        assert clearance == 0.0

    def test_one_point_trajectory_skips_corridor_and_cv_sweep(self):
        # Short path is common after end-of-path tracker fixes; movers must not be CV-swept.
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[
                AgentState(x=40.0, y=8.0, theta=-np.pi / 2, velocity=5.0, agent_id=1),
            ],
        )
        one_point = TrajectoryTracker(path=[(0.0, 0.0)], velocity=[5.0])
        hit, idx, _, clearance = check_collision(pm, one_point)
        assert hit is False
        assert idx == -1
        assert clearance > 1e5
