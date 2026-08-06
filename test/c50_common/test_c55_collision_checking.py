"""Unit tests for collision checking (avlite.c50_common.c55_collision_checking).

Tests verify:
- Static obstacles block trajectories that intersect their footprint.
- Clear trajectories report no collision.
- precompute_obstacle_polygons returns one polygon per agent.
- Ego + obstacle margins combine to ~1 m body-to-body clearance.
- Ego length extension catches front-corner side overlaps.
- Forward sweeps require prediction; movers without trajectories stay static.
- Beside/behind gating uses beside_rear_window / beside_sweep_time.
"""

import numpy as np

from avlite.c10_perception.c11_perception_model import (
    AgentState,
    EgoState,
    PerceptionModel,
    SingleTrajectory,
)
from avlite.c50_common.c54_trajectory_tracker import TrajectoryTracker
from avlite.c50_common.c55_collision_checking import check_collision, precompute_obstacle_polygons


def _forward_prediction(agent: AgentState, *, dt: float = 0.1, n_steps: int = 40) -> SingleTrajectory:
    steps = np.empty((n_steps, 2))
    for t in range(n_steps):
        time = (t + 1) * dt
        steps[t, 0] = agent.x + agent.velocity * np.cos(agent.theta) * time
        steps[t, 1] = agent.y + agent.velocity * np.sin(agent.theta) * time
    return SingleTrajectory(predict_delta_t=dt, trajectories={agent.agent_id: steps})


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


class TestPredictorGatedSweep:
    """precompute_obstacle_polygons must not fabricate constant-velocity sweeps."""

    def test_moving_agent_without_prediction_stays_static_box(self):
        agent = AgentState(x=50.0, y=0.0, theta=0.0, velocity=5.0, agent_id=1)
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[agent],
        )
        polys = precompute_obstacle_polygons(pm, total_time=2.0)
        assert abs(polys[0][0].centroid.x - 50.0) < 2.0

    def test_moving_agent_with_prediction_sweeps_forward(self):
        agent = AgentState(x=50.0, y=0.0, theta=0.0, velocity=5.0, agent_id=1)
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[agent],
            prediction=_forward_prediction(agent),
        )
        polys = precompute_obstacle_polygons(pm, total_time=2.0)
        # Convex hull of current + predicted poses reaches ~x=60.
        minx, _, maxx, _ = polys[0][0].bounds
        assert minx < 52.0
        assert maxx > 58.0

    def test_far_behind_agent_not_beside_swept(self):
        agent = AgentState(x=-30.0, y=0.0, theta=0.0, velocity=5.0, agent_id=1)
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[agent],
            prediction=_forward_prediction(agent),
        )
        polys = precompute_obstacle_polygons(
            pm,
            total_time=2.0,
            beside_sweep_time=1.0,
            beside_rear_window=10.0,
        )
        assert abs(polys[0][0].centroid.x - (-30.0)) < 2.0

    def test_just_behind_agent_uses_beside_sweep(self):
        agent = AgentState(x=-5.0, y=0.0, theta=0.0, velocity=5.0, agent_id=1)
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[agent],
            prediction=_forward_prediction(agent),
        )
        polys = precompute_obstacle_polygons(
            pm,
            total_time=2.0,
            beside_sweep_time=1.0,
            beside_rear_window=10.0,
        )
        minx, _, maxx, _ = polys[0][0].bounds
        assert minx < -3.0
        # 1 s of forward motion from x=-5 at 5 m/s → ~0, not the 2 s total_time tip.
        assert maxx > -2.0
        assert maxx < 4.0
