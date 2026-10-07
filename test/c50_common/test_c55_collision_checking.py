"""Unit tests for collision checking (avlite.c50_common.c55_collision_checking).

Tests verify:
- Static obstacles block trajectories that intersect their footprint.
- Clear trajectories report no collision.
- precompute_obstacle_polygons_2d returns one polygon per agent.
- Ego + obstacle margins combine to ~1 m body-to-body clearance.
- Ego length extension catches front-corner side overlaps.
"""

from unittest.mock import patch

import numpy as np

from avlite.c10_perception.c11_perception_model import (
    AgentState,
    AggregatedOccupancyFlow,
    EgoState,
    GMM,
    GP,
    MultiTrajectory,
    OccupancyFlow,
    PerceptionModel,
    SingleTrajectory,
)
from avlite.c50_common import c55_collision_checking as c55
from avlite.c50_common.c54_trajectory_tracker import TrajectoryTracker
from avlite.c10_perception.c19_settings import PerceptionSettings
from avlite.c50_common.c55_collision_checking import (
    check_collision_2d,
    collision_probability_2d,
    precompute_obstacle_polygons_2d,
)


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
        hit, idx, _vel, clearance = check_collision_2d(pm, trajectory)
        assert hit is False
        assert idx == -1
        assert clearance > 0

    def test_intersecting_static_agent_is_detected(self):
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[AgentState(x=50.0, y=0.0, theta=0.0, velocity=0.0, agent_id=1)],
        )
        trajectory = _straight_trajectory(0.0, 100.0)
        hit, idx, _vel, clearance = check_collision_2d(pm, trajectory)
        assert hit is True
        assert idx >= 0
        assert clearance == 0.0

    def test_precomputed_polygons_match_slow_path(self):
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[AgentState(x=50.0, y=0.0, theta=0.0, velocity=0.0, agent_id=1)],
        )
        trajectory = _straight_trajectory(0.0, 100.0)
        polygons = precompute_obstacle_polygons_2d(pm, total_time=2.0)
        assert len(polygons) == 1
        hit_fast, idx_fast, _, _ = check_collision_2d(pm, trajectory, obstacle_polygons=polygons)
        hit_slow, idx_slow, _, _ = check_collision_2d(pm, trajectory)
        assert hit_fast == hit_slow
        assert idx_fast == idx_slow


class TestMarginClearance:
    """Body gap ≈ ego_inflation_margin + obstacle_inflation_margin (both 0.5 → 1.0 m)."""

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
        polys = precompute_obstacle_polygons_2d(
            pm, total_time=1.0, obstacle_inflation_margin=self._MARGIN,
        )
        return pm, polys

    def test_body_gap_under_1m_collides(self):
        # center-to-center needed for touch: ego_w/2 + agent_w/2 + 1.0 = 3.0
        body_gap = 0.8
        agent_y = self._EGO_W / 2 + self._AGENT_W / 2 + body_gap
        pm, polys = self._pm_and_polys(agent_y)
        hit, _, _, clearance = check_collision_2d(
            pm, _straight_trajectory(0.0, 100.0),
            obstacle_polygons=polys,
            ego_inflation_margin=self._MARGIN,
        )
        assert hit is True
        assert clearance == 0.0

    def test_body_gap_over_1m_is_clear(self):
        body_gap = 1.2
        agent_y = self._EGO_W / 2 + self._AGENT_W / 2 + body_gap
        pm, polys = self._pm_and_polys(agent_y)
        hit, _, _, clearance = check_collision_2d(
            pm, _straight_trajectory(0.0, 100.0),
            obstacle_polygons=polys,
            ego_inflation_margin=self._MARGIN,
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
        polys = precompute_obstacle_polygons_2d(pm, total_time=1.0, obstacle_inflation_margin=margin)
        hit, _, _, _ = check_collision_2d(
            pm, _straight_trajectory(0.0, path_end),
            obstacle_polygons=polys,
            ego_inflation_margin=margin,
        )
        assert hit is True


def _timed_trajectory() -> TrajectoryTracker:
    xs = [0.0, 10.0, 20.0, 30.0, 40.0]
    return TrajectoryTracker(path=[(x, 0.0) for x in xs], velocity=[10.0] * len(xs))


class TestCollisionProbability:
    def test_crossing_mode_weight(self):
        modes = np.array([[[10.0, 0.0]], [[10.0, 20.0]]])
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=10.0),
            agent_vehicles=[AgentState(x=10.0, y=6.0, theta=0.0, velocity=0.0, agent_id=1)],
            prediction=MultiTrajectory(
                predict_delta_t=1.0,
                trajectories={1: modes},
                weights={1: np.array([0.3, 0.7])},
            ),
        )
        p = collision_probability_2d(pm, _timed_trajectory(), ego_inflation_margin=0.0)
        assert abs(p - 0.3) < 1e-9

    def test_both_modes_miss(self):
        modes = np.array([[[10.0, 20.0]], [[10.0, 25.0]]])
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=10.0),
            agent_vehicles=[AgentState(x=10.0, y=6.0, theta=0.0, velocity=0.0, agent_id=1)],
            prediction=MultiTrajectory(
                predict_delta_t=1.0,
                trajectories={1: modes},
                weights={1: np.array([0.3, 0.7])},
            ),
        )
        assert collision_probability_2d(pm, _timed_trajectory(), ego_inflation_margin=0.0) == 0.0

    def test_static_agent_on_path(self):
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=10.0),
            agent_vehicles=[AgentState(x=20.0, y=0.0, theta=0.0, velocity=0.0, agent_id=1)],
        )
        assert collision_probability_2d(pm, _timed_trajectory(), ego_inflation_margin=0.0) == 1.0

    def test_two_agents_are_independent(self):
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=10.0),
            agent_vehicles=[
                AgentState(x=10.0, y=6.0, theta=0.0, velocity=0.0, agent_id=1),
                AgentState(x=20.0, y=6.0, theta=0.0, velocity=0.0, agent_id=2),
            ],
            prediction=MultiTrajectory(
                predict_delta_t=1.0,
                trajectories={
                    1: np.array([[[10.0, 0.0]]]),
                    2: np.array([[[20.0, 6.0], [20.0, 0.0]]]),
                },
                weights={1: np.array([0.5]), 2: np.array([0.5])},
            ),
        )
        p = collision_probability_2d(pm, _timed_trajectory(), ego_inflation_margin=0.0)
        assert p == 0.75

    def test_single_trajectory_crossing_is_certain(self):
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=10.0),
            agent_vehicles=[AgentState(x=10.0, y=6.0, theta=0.0, velocity=0.0, agent_id=1)],
            prediction=SingleTrajectory(
                predict_delta_t=1.0,
                trajectories={1: np.array([[10.0, 0.0]])},
            ),
        )
        assert collision_probability_2d(pm, _timed_trajectory(), ego_inflation_margin=0.0) == 1.0

    def test_multi_trajectory_warns_each_call_and_stays_static(self):
        agent = AgentState(x=10.0, y=0.0, theta=0.0, velocity=5.0, agent_id=1)
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[agent],
            prediction=MultiTrajectory(
                trajectories={1: np.array([[[100.0, 0.0]]])},
                weights={1: np.array([1.0])},
            ),
        )
        with patch.object(c55.log, "warning") as warn:
            first = precompute_obstacle_polygons_2d(pm, total_time=2.0)
            precompute_obstacle_polygons_2d(pm, total_time=2.0)
        assert first[0][0].equals(agent.get_bb_polygon())
        assert warn.call_count == 2
        assert warn.call_args.args[1] == "MultiTrajectory"

    def test_gmm_crossing_mode_weight(self):
        modes = np.array([[[10.0, 0.0]], [[10.0, 20.0]]])
        # Nonzero covariance on the hitting mode. The sweep uses the mean, not this ellipse.
        covariances = np.zeros((2, 1, 2, 2))
        covariances[0, 0] = np.eye(2) * 25.0
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=10.0),
            agent_vehicles=[AgentState(x=10.0, y=6.0, theta=0.0, velocity=0.0, agent_id=1)],
            prediction=GMM(
                predict_delta_t=1.0,
                trajectories={1: modes},
                weights={1: np.array([0.3, 0.7])},
                covariances={1: covariances},
            ),
        )
        p = collision_probability_2d(pm, _timed_trajectory(), ego_inflation_margin=0.0)
        assert abs(p - 0.3) < 1e-9

    def test_gp_mean_is_not_swept(self):
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=10.0),
            agent_vehicles=[AgentState(x=10.0, y=6.0, theta=0.0, velocity=0.0, agent_id=1)],
            prediction=GP(
                predict_delta_t=1.0,
                means={1: np.array([[10.0, 0.0]])},
                covariance={1: np.eye(2)},
            ),
        )
        assert collision_probability_2d(pm, _timed_trajectory(), ego_inflation_margin=0.0) == 0.0

    def test_occupancy_flow_uses_cells_on_the_ego_box(self):
        # Agent sits on the path. A current-box scan would return 1.
        grid = np.zeros((21, 11))
        grid[0, 10] = 0.25
        grid[20, 0] = 0.9
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=10.0),
            agent_vehicles=[AgentState(x=20.0, y=0.0, theta=0.0, velocity=4.0, agent_id=1)],
            prediction=OccupancyFlow(
                predict_delta_t=1.0,
                occupancy_flow={1: [grid]},
                origin_x=0.0,
                origin_y=0.0,
                resolution=1.0,
            ),
        )
        found: list[tuple[int, float]] = []
        p = collision_probability_2d(
            pm, _timed_trajectory(), ego_inflation_margin=0.0, index_out=found,
        )
        assert p == 0.25
        assert found == [(1, 4.0)]

    def test_occupancy_flow_keeps_the_earlier_step_index(self):
        first = np.zeros((1, 21))
        first[0, 10] = 0.25
        second = np.zeros((1, 21))
        second[0, 20] = 0.5
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=10.0),
            agent_vehicles=[AgentState(x=20.0, y=6.0, theta=0.0, velocity=4.0, agent_id=1)],
            prediction=OccupancyFlow(
                predict_delta_t=1.0,
                occupancy_flow={1: [first, second]},
                origin_x=0.0,
                origin_y=0.0,
                resolution=1.0,
            ),
        )
        found: list[tuple[int, float]] = []
        p = collision_probability_2d(
            pm, _timed_trajectory(), ego_inflation_margin=0.0, index_out=found,
        )
        assert p == 0.5
        assert found == [(1, 4.0)]

    def test_occupancy_flow_agents_combine(self):
        low = np.zeros((1, 11))
        low[0, 10] = 0.25
        high = np.zeros((1, 11))
        high[0, 10] = 0.5
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=10.0),
            agent_vehicles=[
                AgentState(x=10.0, y=6.0, theta=0.0, velocity=0.0, agent_id=1),
                AgentState(x=20.0, y=6.0, theta=0.0, velocity=0.0, agent_id=2),
            ],
            prediction=OccupancyFlow(
                predict_delta_t=1.0,
                occupancy_flow={1: [low], 2: [high]},
                origin_x=0.0,
                origin_y=0.0,
                resolution=1.0,
            ),
        )
        p = collision_probability_2d(pm, _timed_trajectory(), ego_inflation_margin=0.0)
        assert p == 0.625

    def test_occupancy_flow_without_a_grid_uses_the_current_box(self):
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=10.0),
            agent_vehicles=[AgentState(x=20.0, y=0.0, theta=0.0, velocity=0.0, agent_id=1)],
            prediction=OccupancyFlow(predict_delta_t=1.0, occupancy_flow={}),
        )
        assert collision_probability_2d(pm, _timed_trajectory(), ego_inflation_margin=0.0) == 1.0

    def test_aggregated_occupancy_flow_is_scored_once(self):
        grid = np.zeros((1, 11))
        grid[0, 10] = 0.25
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=10.0),
            agent_vehicles=[
                AgentState(x=20.0, y=0.0, theta=0.0, velocity=4.0, agent_id=1),
                AgentState(x=30.0, y=0.0, theta=0.0, velocity=6.0, agent_id=2),
            ],
            prediction=AggregatedOccupancyFlow(
                predict_delta_t=1.0,
                occupancy_flow=[grid],
                origin_x=0.0,
                origin_y=0.0,
                resolution=1.0,
            ),
        )
        found: list[tuple[int, float]] = []
        p = collision_probability_2d(
            pm, _timed_trajectory(), ego_inflation_margin=0.0, index_out=found,
        )
        assert p == 0.25
        assert found == [(1, 0.0)]

    def test_gmm_highest_weight_mean_is_swept(self):
        agent = AgentState(x=10.0, y=0.0, theta=0.0, velocity=5.0, agent_id=1)
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[agent],
            prediction=GMM(
                predict_delta_t=1.0,
                trajectories={1: np.array([[[100.0, 0.0]]])},
                weights={1: np.array([1.0])},
                covariances={1: np.zeros((1, 1, 2, 2))},
            ),
        )
        with patch.object(c55.log, "warning") as warn:
            first = precompute_obstacle_polygons_2d(pm, total_time=2.0)
            precompute_obstacle_polygons_2d(pm, total_time=2.0)
        assert warn.call_count == 0
        assert not first[0][0].equals(agent.get_bb_polygon())
        assert first[0][0].bounds[2] > 90

    def test_gmm_sweeps_higher_weight_not_the_other_mean(self):
        agent = AgentState(x=10.0, y=0.0, theta=0.0, velocity=5.0, agent_id=1)
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[agent],
            prediction=GMM(
                predict_delta_t=1.0,
                trajectories={1: np.array([[[100.0, 0.0]], [[0.0, 50.0]]])},
                weights={1: np.array([0.3, 0.7])},
                covariances={1: np.zeros((2, 1, 2, 2))},
            ),
        )
        poly = precompute_obstacle_polygons_2d(pm, total_time=2.0)[0][0]
        _minx, _miny, maxx, maxy = poly.bounds
        assert maxy > 40
        assert maxx < 20

    def test_gp_mean_is_swept(self):
        agent = AgentState(x=10.0, y=0.0, theta=0.0, velocity=5.0, agent_id=1)
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[agent],
            prediction=GP(
                predict_delta_t=1.0,
                means={1: np.array([[100.0, 0.0]])},
                covariance={1: np.eye(2)},
            ),
        )
        with patch.object(c55.log, "warning") as warn:
            first = precompute_obstacle_polygons_2d(pm, total_time=2.0)
            precompute_obstacle_polygons_2d(pm, total_time=2.0)
        assert warn.call_count == 0
        assert not first[0][0].equals(agent.get_bb_polygon())
        assert first[0][0].bounds[2] > 90

    def test_occupancy_flow_warns_each_call_and_stays_static(self):
        agent = AgentState(x=10.0, y=0.0, theta=0.0, velocity=5.0, agent_id=1)
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[agent],
            prediction=OccupancyFlow(occupancy_flow={1: [np.zeros((2, 2))]}),
        )
        with patch.object(c55.log, "warning") as warn:
            first = precompute_obstacle_polygons_2d(pm, total_time=2.0)
            precompute_obstacle_polygons_2d(pm, total_time=2.0)
        assert first[0][0].equals(agent.get_bb_polygon())
        assert warn.call_count == 2
        assert warn.call_args.args[1] == "OccupancyFlow"

    def test_aggregated_occupancy_flow_warns_each_call_and_stays_static(self):
        agent = AgentState(x=10.0, y=0.0, theta=0.0, velocity=5.0, agent_id=1)
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[agent],
            prediction=AggregatedOccupancyFlow(occupancy_flow=[np.zeros((2, 2))]),
        )
        with patch.object(c55.log, "warning") as warn:
            first = precompute_obstacle_polygons_2d(pm, total_time=2.0)
            precompute_obstacle_polygons_2d(pm, total_time=2.0)
        assert first[0][0].equals(agent.get_bb_polygon())
        assert warn.call_count == 2
        assert warn.call_args.args[1] == "AggregatedOccupancyFlow"


def _crossing_multi(weight: float, velocity: float = 3.5) -> PerceptionModel:
    modes = np.array([[[10.0, 0.0]], [[10.0, 20.0]]])
    return PerceptionModel(
        ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=10.0),
        agent_vehicles=[AgentState(x=10.0, y=6.0, theta=0.0, velocity=velocity, agent_id=1)],
        prediction=MultiTrajectory(
            predict_delta_t=1.0,
            trajectories={1: modes},
            weights={1: np.array([weight, 1.0 - weight])},
        ),
    )


class TestLocalCollision:
    def _enable(self):
        PerceptionSettings.c15_probabilistic_collision_checking = True

    def test_flag_off_keeps_geometric_check(self):
        pm = _crossing_multi(0.3)
        trajectory = _timed_trajectory()
        local = check_collision_2d(pm, trajectory, ego_inflation_margin=0.0)
        geometric = check_collision_2d(pm, trajectory, ego_inflation_margin=0.0)
        assert PerceptionSettings.c15_probabilistic_collision_checking is False
        assert local[0] is False
        assert local[0] == geometric[0]
        assert local[1] == geometric[1]

    def test_single_trajectory_matches_geometric_check(self):
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[AgentState(x=50.0, y=0.0, theta=0.0, velocity=0.0, agent_id=1)],
            prediction=SingleTrajectory(
                predict_delta_t=0.1,
                trajectories={1: np.array([[50.0, 0.0], [60.0, 0.0]])},
            ),
        )
        trajectory = _straight_trajectory(0.0, 100.0)
        geometric = check_collision_2d(pm, trajectory)
        local = check_collision_2d(pm, trajectory)
        assert local == geometric
        assert local[0] is True
        assert local[3] == 0.0

    def test_default_max_flags_weight_above_five_percent(self):
        self._enable()
        hit, idx, vel, clearance = check_collision_2d(
            _crossing_multi(0.3), _timed_trajectory(), ego_inflation_margin=0.0,
        )
        assert hit is True
        assert idx >= 0
        assert vel == 3.5
        assert clearance == 0.0

    def test_default_max_clears_weight_at_or_below_five_percent(self):
        self._enable()
        hit, idx, _vel, clearance = check_collision_2d(
            _crossing_multi(0.03), _timed_trajectory(), ego_inflation_margin=0.0,
        )
        assert hit is False
        assert idx == -1
        assert clearance > 0

        PerceptionSettings.c15_max_local_collision_probability = 0.25
        hit, idx, _vel, clearance = check_collision_2d(
            _crossing_multi(0.25), _timed_trajectory(), ego_inflation_margin=0.0,
        )
        assert hit is False
        assert idx == -1
        assert clearance > 0

    def test_gp_uses_current_box_not_mean_sweep(self):
        agent = AgentState(x=10.0, y=6.0, theta=0.0, velocity=5.0, agent_id=1)
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=10.0),
            agent_vehicles=[agent],
            prediction=GP(
                predict_delta_t=1.0,
                means={1: np.array([[10.0, 0.0]])},
                covariance={1: np.eye(2)},
            ),
        )
        trajectory = _timed_trajectory()
        swept, *_ = check_collision_2d(
            pm, trajectory,
            obstacle_polygons=precompute_obstacle_polygons_2d(pm, total_time=2.0),
            ego_inflation_margin=0.0,
        )
        self._enable()
        hit, idx, _vel, clearance = check_collision_2d(
            pm, trajectory, ego_inflation_margin=0.0,
        )
        assert swept is True
        assert hit is False
        assert idx == -1
        assert clearance > 0
