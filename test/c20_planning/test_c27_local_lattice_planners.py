"""Unit tests for GreedyLatticePlanner plan switching and edge-chain concat (c27)."""

import time

import pytest

from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c20_planning.c27_local_lattice_planners import GreedyLatticePlanner
from avlite.c20_planning.c28_lattice import Edge, Node
from avlite.c60_common.c63_trajectory_tracker import TrajectoryTracker


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


def _plan_length_acceptable(
    planner: GreedyLatticePlanner, new_plan: Edge, new_clean: bool = True
) -> bool:
    """Mirror replan commit gate in c27_local_lattice_planners."""
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


class TestShouldSwitchPlan:
    def test_switches_to_faster_plan_when_current_not_marked_collision(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=3.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        slow_edge = _edge_with_velocity(global_plan.trajectory, velocity=3.0, collision=False)
        fast_edge = _edge_with_velocity(global_plan.trajectory, velocity=10.0, collision=False)

        planner.selected_local_plan = slow_edge
        planner._last_plan_change_time = time.time()
        assert planner.should_switch_plan(fast_edge) is False

        planner._last_plan_change_time = 0.0
        assert planner.should_switch_plan(fast_edge) is True

    def test_keeps_clean_plan_when_faster_alternative_without_wait(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=8.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        current = _edge_with_velocity(global_plan.trajectory, velocity=8.0, collision=False)
        faster = _edge_with_velocity(global_plan.trajectory, velocity=10.0, collision=False)

        planner.selected_local_plan = current
        planner._last_plan_change_time = time.time()

        assert planner.should_switch_plan(faster) is False

    def test_switches_when_agent_cleared_on_colliding_head(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=3.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        slow_colliding = _edge_with_velocity(global_plan.trajectory, velocity=3.0, collision=True)
        slow_colliding.collision_idx = 8
        slow_colliding.local_trajectory.current_wp = 0
        fast_clean = _edge_with_velocity(global_plan.trajectory, velocity=10.0, collision=False)

        planner.selected_local_plan = slow_colliding
        planner._last_plan_change_time = time.time()

        assert planner.should_switch_plan(fast_clean) is True

    def test_keeps_plan_when_alternative_is_not_faster(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=8.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        current = _edge_with_velocity(global_plan.trajectory, velocity=8.0, collision=False)
        similar = _edge_with_velocity(global_plan.trajectory, velocity=8.2, collision=False)

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
        planner._last_plan_change_time = time.time()

        assert planner.should_switch_plan(three) is True

    def test_keeps_longer_by_one_without_wait_or_gain(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        one = _edge_at(global_plan.trajectory, 0.0, 20.0)
        two = _link_edges([
            _edge_at(global_plan.trajectory, 0.0, 20.0),
            _edge_at(global_plan.trajectory, 20.0, 40.0),
        ])
        planner.set_selected_plan(one)
        planner._last_plan_change_time = time.time()

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

    def test_keeps_collision_escape_without_wait_or_gain(self):
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
        planner._last_plan_change_time = time.time()

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
    @pytest.mark.parametrize(
        "prev_len,new_len,old_colliding,new_clean,expected",
        [
            (None, 1, False, True, True),
            (None, 0, False, True, False),
            (1, 3, False, True, True),
            (3, 2, False, True, True),
            (3, 3, False, True, True),
            (3, 4, False, True, True),
            (3, 1, False, True, False),
            (3, 1, True, True, True),
            (3, 5, False, True, True),
        ],
    )
    def test_acceptance_rule(self, prev_len, new_len, old_colliding, new_clean, expected):
        acceptable = (prev_len is None and new_len >= 1) or (
            prev_len is not None and (
                new_len >= prev_len
                or abs(new_len - prev_len) <= 1
                or (old_colliding and new_clean)
            )
        )
        assert acceptable == expected

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
