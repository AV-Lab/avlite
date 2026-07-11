"""Unit tests for the frenet lattice planners and primitives (avlite.c20_planning.c28_local_lattice_planners).

Covers both the lattice data structures (Node/Edge/Lattice) and the
GreedyLatticePlanner plan-switching and edge-chain concatenation logic.
"""

import pytest

from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c20_planning.c28_local_lattice_planners import Edge, GreedyLatticePlanner, Lattice, Node
from avlite.c50_common.c54_trajectory_tracker import TrajectoryTracker

_FIXED_PLANNER_TIME = 1000.0


@pytest.fixture
def fixed_planner_time(monkeypatch):
    monkeypatch.setattr(
        "avlite.c20_planning.c28_local_lattice_planners.time.time",
        lambda: _FIXED_PLANNER_TIME,
    )


def _straight_global_plan(x_end: float = 100.0, n: int = 20, velocity: float = 10.0) -> GlobalPlan:
    xs = [x_end * i / (n - 1) for i in range(n)]
    path = [(x, 0.0) for x in xs]
    vel = [velocity] * n
    left = [3.0] * n
    right = [-3.0] * n
    trajectory = TrajectoryTracker(path=path, velocity=vel)
    trajectory.ref_left_boundary_d = left
    trajectory.ref_right_boundary_d = right
    return GlobalPlan(
        start_point=path[0],
        goal_point=path[-1],
        path=path,
        velocity=vel,
        trajectory=trajectory,
        left_boundary_d=left,
        right_boundary_d=right,
    )


def _edge_with_velocity(global_tj: TrajectoryTracker, velocity: float, collision: bool = False) -> Edge:
    start = Node(s=0.0, d=0.0, x=0.0, y=0.0)
    end = Node(s=20.0, d=0.0, x=20.0, y=0.0)
    edge = Edge(start=start, end=end, global_tj=global_tj, num_of_points=10)
    edge.local_trajectory.velocity = [velocity] * len(edge.local_trajectory.velocity)
    edge.collision = collision
    return edge


def _edge_at(global_tj: TrajectoryTracker, s0: float, s1: float, collision: bool = False) -> Edge:
    start = Node(s=s0, d=0.0, x=s0, y=0.0)
    end = Node(s=s1, d=0.0, x=s1, y=0.0)
    edge = Edge(start=start, end=end, global_tj=global_tj, num_of_points=10)
    edge.collision = collision
    return edge


def _edge_at_sd(
    global_tj: TrajectoryTracker,
    s0: float,
    d0: float,
    s1: float,
    d1: float,
    collision: bool = False,
    boundary_violation: bool = False,
) -> Edge:
    x0, y0 = global_tj.convert_sd_to_xy(s0, d0)
    x1, y1 = global_tj.convert_sd_to_xy(s1, d1)
    start = Node(s=s0, d=d0, x=x0, y=y0)
    end = Node(s=s1, d=d1, x=x1, y=y1)
    edge = Edge(start=start, end=end, global_tj=global_tj, num_of_points=10)
    edge.collision = collision
    edge.boundary_violation = boundary_violation
    return edge


def _link_edges(edges: list[Edge]) -> Edge:
    for i in range(len(edges) - 1):
        edges[i].selected_next_local_plan = edges[i + 1]
    return edges[0]


def _chain_has_collision(head: Edge) -> bool:
    edge = head
    while edge is not None:
        if edge.collision:
            return True
        edge = edge.selected_next_local_plan
    return False


def _chain_with_uniform_velocity(global_tj: TrajectoryTracker, velocity: float, n_edges: int = 3) -> Edge:
    edges = []
    for i in range(n_edges):
        s0 = i * 20.0
        s1 = (i + 1) * 20.0
        edge = _edge_at(global_tj, s0, s1)
        edge.local_trajectory.velocity = [velocity] * len(edge.local_trajectory.velocity)
        edges.append(edge)
    return _link_edges(edges)


def _plan_length_acceptable(
    planner: GreedyLatticePlanner, new_plan: Edge, new_clean: bool = True
) -> bool:
    """Mirror replan commit gate in c28_local_lattice_planners."""
    new_len = planner.local_plan_len(new_plan)
    prev_len = planner.local_plan_len() if planner.selected_local_plan else None
    old_colliding = (
        _chain_has_collision(planner.selected_local_plan)
        if planner.selected_local_plan is not None
        else False
    )
    return (prev_len is None and new_len >= 1) or (
        prev_len is not None and (
            new_len >= prev_len
            or abs(new_len - prev_len) <= 1
            or (old_colliding and new_clean)
        )
    )


class TestLatticeNode:
    def test_equal_nodes_share_hash(self):
        a = Node(s=1.0, d=0.0, x=1.0, y=0.0)
        b = Node(s=1.0, d=0.0, x=1.0, y=0.0)
        assert hash(a) == hash(b)


class TestLatticeEdge:
    def test_edge_builds_local_trajectory(self):
        global_tj = _straight_global_plan().trajectory
        start = Node(s=0.0, d=0.0, x=0.0, y=0.0)
        end = Node(s=20.0, d=0.0, x=20.0, y=0.0)
        edge = Edge(start=start, end=end, global_tj=global_tj, num_of_points=10)
        assert edge.local_trajectory is not None
        assert len(edge.local_trajectory.path_x) == 10


class TestSampleNodes:
    def _lattice(self, planning_horizon: int = 2) -> Lattice:
        global_tj = _straight_global_plan().trajectory
        n = len(global_tj.path)
        return Lattice(
            global_trajectory=global_tj,
            ref_left_boundary_d=[3.0] * n,   # upper limit (larger d)
            ref_right_boundary_d=[-3.0] * n,  # lower limit (smaller d)
            planning_horizon=planning_horizon,
        )

    def test_each_level_has_sample_size_distinct_offsets(self):
        # Regression: with the boundary convention left(+) > right(-), the sampling
        # band must not collapse — each level should carry `sample_size` distinct d's
        # (reference node + sample_size-1 samples), not just 2 points.
        lattice = self._lattice(planning_horizon=2)
        sample_size = 5
        lattice.sample_nodes(
            s=0.0, d=0.0, sample_size=sample_size, maneuver_distance=20.0,
            boundary_clearance=0.5, lateral_reach=float('inf'), sample_distribution=0,
        )
        for level in range(1, 3):
            offsets = {round(node.d, 6) for node in lattice.lattice_nodes_by_level[level]}
            assert len(offsets) == sample_size, (level, offsets)

    def test_samples_stay_within_boundary_band(self):
        lattice = self._lattice(planning_horizon=1)
        lattice.sample_nodes(
            s=0.0, d=0.0, sample_size=6, maneuver_distance=20.0,
            boundary_clearance=0.5, lateral_reach=float('inf'), sample_distribution=1,
        )
        for node in lattice.lattice_nodes_by_level[1]:
            assert -2.5 <= node.d <= 2.5  # [right + clr, left - clr]


class TestLatticeReset:
    def test_reset_clears_accumulated_state(self):
        global_tj = _straight_global_plan().trajectory
        lattice = Lattice(
            global_trajectory=global_tj,
            ref_left_boundary_d=[3.0] * len(global_tj.path),
            ref_right_boundary_d=[-3.0] * len(global_tj.path),
            planning_horizon=1,
        )
        lattice.nodes.append(Node(s=5.0, d=0.0, x=5.0, y=0.0))
        lattice.lattice_nodes_by_level[0].append(Node(s=0.0, d=0.0, x=0.0, y=0.0))
        lattice.reset()
        assert lattice.nodes == []
        assert lattice.edges == []
        assert len(lattice.lattice_nodes_by_level) == 0


class TestShouldSwitchPlan:
    def test_switches_to_faster_plan_when_current_not_marked_collision(self, fixed_planner_time):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=3.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        slow_chain = _chain_with_uniform_velocity(global_plan.trajectory, velocity=3.0)
        fast_chain = _chain_with_uniform_velocity(global_plan.trajectory, velocity=10.0)

        planner.selected_local_plan = slow_chain
        planner._last_plan_change_time = _FIXED_PLANNER_TIME
        assert planner.should_switch_plan(fast_chain) is False

        planner._last_plan_change_time = 0.0
        assert planner.should_switch_plan(fast_chain) is True

    def test_keeps_clean_plan_when_faster_alternative_without_wait(self, fixed_planner_time):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=8.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        current = _chain_with_uniform_velocity(global_plan.trajectory, velocity=8.0)
        faster = _chain_with_uniform_velocity(global_plan.trajectory, velocity=10.0)

        planner.selected_local_plan = current
        planner._last_plan_change_time = _FIXED_PLANNER_TIME

        assert planner.should_switch_plan(faster) is False

    def test_refreshes_single_edge_without_wait(self, fixed_planner_time):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=8.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        current = _edge_with_velocity(global_plan.trajectory, velocity=8.0, collision=False)
        faster = _edge_with_velocity(global_plan.trajectory, velocity=10.0, collision=False)

        planner.set_selected_plan(current)
        planner._last_plan_change_time = _FIXED_PLANNER_TIME

        assert planner.should_switch_plan(faster) is True

    def test_holds_single_edge_when_unchanged(self, fixed_planner_time):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=8.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        current = _edge_with_velocity(global_plan.trajectory, velocity=8.0, collision=False)
        near_identical = _edge_with_velocity(global_plan.trajectory, velocity=8.2, collision=False)

        planner.set_selected_plan(current)
        planner._last_plan_change_time = _FIXED_PLANNER_TIME

        assert planner.should_switch_plan(near_identical) is False

    def test_switches_when_agent_cleared_on_colliding_head(self, fixed_planner_time):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=3.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        slow_colliding = _edge_with_velocity(global_plan.trajectory, velocity=3.0, collision=True)
        slow_colliding.collision_idx = 8
        slow_colliding.local_trajectory.current_wp = 0
        fast_clean = _edge_with_velocity(global_plan.trajectory, velocity=10.0, collision=False)

        planner.selected_local_plan = slow_colliding
        planner._last_plan_change_time = _FIXED_PLANNER_TIME

        assert planner.should_switch_plan(fast_clean) is True

    def test_keeps_plan_when_alternative_is_not_faster(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=8.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        current = _chain_with_uniform_velocity(global_plan.trajectory, velocity=8.0)
        similar = _chain_with_uniform_velocity(global_plan.trajectory, velocity=8.2)

        planner.selected_local_plan = current
        planner._last_plan_change_time = 0.0

        assert planner.should_switch_plan(similar) is False

    def test_switches_to_longer_clean_chain(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        one = _edge_at(global_plan.trajectory, 0.0, 20.0)
        three = _link_edges([
            _edge_at(global_plan.trajectory, 0.0, 20.0),
            _edge_at(global_plan.trajectory, 20.0, 40.0),
            _edge_at(global_plan.trajectory, 40.0, 60.0),
        ])
        planner.set_selected_plan(one)
        planner._last_plan_change_time = _FIXED_PLANNER_TIME

        assert planner.should_switch_plan(three) is True

    def test_keeps_longer_by_one_without_wait_or_gain(self, fixed_planner_time):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        one = _edge_at(global_plan.trajectory, 0.0, 20.0)
        two = _link_edges([
            _edge_at(global_plan.trajectory, 0.0, 20.0),
            _edge_at(global_plan.trajectory, 20.0, 40.0),
        ])
        planner.set_selected_plan(one)
        planner._last_plan_change_time = _FIXED_PLANNER_TIME

        assert planner.should_switch_plan(two) is False

    def test_switches_longer_by_one_after_wait(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        one = _edge_at(global_plan.trajectory, 0.0, 20.0)
        two = _link_edges([
            _edge_at(global_plan.trajectory, 0.0, 20.0),
            _edge_at(global_plan.trajectory, 20.0, 40.0),
        ])
        planner.set_selected_plan(one)
        planner._last_plan_change_time = 0.0

        assert planner.should_switch_plan(two) is True

    def test_switches_to_shorter_clean_when_old_chain_collides(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        colliding_three = _link_edges([
            _edge_at(global_plan.trajectory, 0.0, 20.0, collision=True),
            _edge_at(global_plan.trajectory, 20.0, 40.0),
            _edge_at(global_plan.trajectory, 40.0, 60.0),
        ])
        one_clean = _edge_at(global_plan.trajectory, 0.0, 20.0)
        planner.set_selected_plan(colliding_three)
        planner._last_plan_change_time = 0.0

        assert planner.should_switch_plan(one_clean) is True

    def test_keeps_collision_escape_without_wait_or_gain(self, fixed_planner_time):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        colliding_three = _link_edges([
            _edge_at(global_plan.trajectory, 0.0, 20.0, collision=True),
            _edge_at(global_plan.trajectory, 20.0, 40.0),
            _edge_at(global_plan.trajectory, 40.0, 60.0),
        ])
        one_clean = _edge_at(global_plan.trajectory, 0.0, 20.0)
        planner.set_selected_plan(colliding_three)
        planner._last_plan_change_time = _FIXED_PLANNER_TIME

        assert planner.should_switch_plan(one_clean) is False

    def test_keeps_same_length_clean_plan_without_jitter(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=8.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        current = _link_edges([
            _edge_at(global_plan.trajectory, 0.0, 20.0),
            _edge_at(global_plan.trajectory, 20.0, 40.0),
            _edge_at(global_plan.trajectory, 40.0, 60.0),
        ])
        alternative = _link_edges([
            _edge_at(global_plan.trajectory, 0.0, 20.0),
            _edge_at(global_plan.trajectory, 20.0, 40.0),
            _edge_at(global_plan.trajectory, 40.0, 60.0),
        ])
        planner.set_selected_plan(current)
        planner._last_plan_change_time = 0.0

        assert planner.should_switch_plan(alternative) is False


class TestGetLocalPlanConcatChain:
    def test_two_edge_chain_spans_tail_endpoint(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        e0 = _edge_at(global_plan.trajectory, 0.0, 20.0)
        e1 = _edge_at(global_plan.trajectory, 20.0, 40.0)
        head = _link_edges([e0, e1])

        planner.set_selected_plan(head)
        local = planner.get_local_plan()
        single_len = len(e0.local_trajectory.path)

        assert len(local.path) > single_len
        assert local.path[-1] == pytest.approx(e1.local_trajectory.path[-1])
        assert local.path[0] == pytest.approx(e0.local_trajectory.path[0])


class TestPlanLengthAcceptance:
    def test_accepts_growth_and_collision_escape_via_planner_state(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        one = _edge_at(global_plan.trajectory, 0.0, 20.0)
        planner.set_selected_plan(one)
        assert planner.local_plan_len() == 1

        three = _link_edges([
            _edge_at(global_plan.trajectory, 0.0, 20.0),
            _edge_at(global_plan.trajectory, 20.0, 40.0),
            _edge_at(global_plan.trajectory, 40.0, 60.0),
        ])
        assert _plan_length_acceptable(planner, three) is True

        colliding_three = _link_edges([
            _edge_at(global_plan.trajectory, 0.0, 20.0, collision=True),
            _edge_at(global_plan.trajectory, 20.0, 40.0),
            _edge_at(global_plan.trajectory, 40.0, 60.0),
        ])
        planner.set_selected_plan(colliding_three)
        assert _plan_length_acceptable(planner, one) is True

        clean_three = _link_edges([
            _edge_at(global_plan.trajectory, 0.0, 20.0),
            _edge_at(global_plan.trajectory, 20.0, 40.0),
            _edge_at(global_plan.trajectory, 40.0, 60.0),
        ])
        planner.set_selected_plan(clean_three)
        assert _plan_length_acceptable(planner, one) is False


class TestOvertakeChainBuilding:
    def test_feasible_candidates_allow_boundary_when_agent_ahead(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        center_collide = _edge_at(global_plan.trajectory, 0.0, 20.0, collision=True)
        boundary_pass = _edge_at_sd(global_plan.trajectory, 0.0, 2.0, 20.0, 2.0, boundary_violation=True)
        edges = [center_collide, boundary_pass]

        assert planner._feasible_candidates(edges, agent_blocks_ahead=False) == []
        assert planner._feasible_candidates(edges, agent_blocks_ahead=True) == [boundary_pass]

    def test_centerline_edge_feasible_within_reach(self, monkeypatch):
        global_plan = _straight_global_plan()
        # High ego speed shrinks the kinematic reach so the reach gate is meaningful.
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=20.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        # Simulate the discretized curvature measurement spuriously rejecting everything.
        monkeypatch.setattr(planner, "_is_curvature_feasible", lambda e: False)

        reach = planner._lateral_reach()
        assert reach < 3.0  # ensure the "beyond reach" case is actually beyond

        ref_within = _edge_at(global_plan.trajectory, 0.0, 20.0)  # start.d=end.d=0
        ref_beyond = _edge_at_sd(global_plan.trajectory, 0.0, 3.0, 20.0, 0.0)  # shift > reach
        non_ref = _edge_at_sd(global_plan.trajectory, 0.0, 0.0, 20.0, 2.0)  # end.d beyond d0

        candidates = planner._feasible_candidates(
            [ref_within, ref_beyond, non_ref], agent_blocks_ahead=False
        )
        assert candidates == [ref_within]

    def test_build_chain_prefers_lateral_successor_when_agent_ahead(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        e0 = _edge_at_sd(global_plan.trajectory, 0.0, 2.0, 20.0, 2.0)
        e1_center = _edge_at_sd(global_plan.trajectory, 20.0, 2.0, 40.0, 0.0)
        e1_lateral = _edge_at_sd(global_plan.trajectory, 20.0, 2.0, 40.0, 2.0)
        e0.next_edges = [e1_center, e1_lateral]

        chain = planner._build_selected_chain([e0], agent_blocks_ahead=True)
        assert chain.selected_next_local_plan is e1_lateral
        assert planner.local_plan_len(chain) == 2

    def test_agent_blocks_ahead_detects_agent_in_front(self):
        from avlite.c10_perception.c11_perception_model import AgentState

        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)
        planner.location_sd = (0.0, 0.0)

        assert planner._agent_blocks_ahead() is False

        pm.agent_vehicles = [
            AgentState(x=15.0, y=0.0, theta=0.0, velocity=3.0, agent_id=1),
        ]
        assert planner._agent_blocks_ahead() is True

    def test_emergency_passing_uses_boundary_relaxed_chain(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        center_collide = _edge_at(global_plan.trajectory, 0.0, 20.0, collision=True)
        lateral_pass = _edge_at_sd(global_plan.trajectory, 0.0, 2.0, 20.0, 2.0, boundary_violation=True)
        tail = _edge_at_sd(global_plan.trajectory, 20.0, 2.0, 40.0, 2.0)
        lateral_pass.next_edges = [tail]

        passing = planner._feasible_candidates([center_collide, lateral_pass], agent_blocks_ahead=True)
        assert passing == [lateral_pass]
        chain = planner._build_selected_chain(passing, agent_blocks_ahead=True)
        assert not chain.collision
        assert planner.local_plan_len(chain) == 2


class TestSetGlobalPlanClearsStaleChain:
    def test_set_global_plan_drops_committed_chain(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        head = _link_edges([
            _edge_at(global_plan.trajectory, 0.0, 20.0),
            _edge_at(global_plan.trajectory, 20.0, 40.0),
        ])
        planner.set_selected_plan(head)
        assert planner.selected_local_plan is not None
        assert planner._committed_trajectory is not None

        # A rebuilt global plan carries a fresh TrajectoryTracker; the old edge
        # chain references the previous one, so it must be dropped on set.
        new_global_plan = _straight_global_plan()
        planner.set_global_plan(new_global_plan)

        assert planner.selected_local_plan is None
        assert planner._committed_trajectory is None
        # get_local_plan falls back to the new global trajectory until next replan.
        assert planner.get_local_plan() is not None


class TestDebounceRelease:
    def test_single_edge_release_is_debounced(self, monkeypatch):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        # Commit a single-edge plan.
        planner.set_selected_plan(_edge_at(global_plan.trajectory, 0.0, 20.0))
        assert planner.local_plan_len() == 1

        # Force the miss path deterministically: level-0 edges exist but none feasible,
        # and should_switch_plan refuses the fallback colliding edge.
        dummy = _edge_at(global_plan.trajectory, 0.0, 20.0, collision=True)
        monkeypatch.setattr(planner.lattice, "reset", lambda *a, **k: None)
        monkeypatch.setattr(planner.lattice, "sample_nodes", lambda *a, **k: None)
        monkeypatch.setattr(planner.lattice, "generate_lattice_from_nodes", lambda *a, **k: None)
        planner.lattice.level0_edges = [dummy]
        monkeypatch.setattr(planner, "_feasible_candidates", lambda *a, **k: [])
        monkeypatch.setattr(planner, "should_switch_plan", lambda *a, **k: False)

        release_ticks = planner._no_plan_release_ticks
        for _ in range(release_ticks - 1):
            planner.replan()
            assert planner.selected_local_plan is not None  # held, no blink

        planner.replan()
        assert planner.selected_local_plan is None
        assert planner._committed_trajectory is None
