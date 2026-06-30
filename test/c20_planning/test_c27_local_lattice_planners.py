"""Unit tests for GreedyLatticePlanner plan switching (c27)."""

import numpy as np

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


class TestShouldSwitchPlan:
    def test_switches_to_faster_plan_when_current_not_marked_collision(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=3.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        slow_edge = _edge_with_velocity(global_plan.trajectory, velocity=3.0, collision=False)
        fast_edge = _edge_with_velocity(global_plan.trajectory, velocity=10.0, collision=False)

        planner.selected_local_plan = slow_edge
        planner._last_plan_change_time = 0.0

        assert planner.should_switch_plan(fast_edge) is True

    def test_keeps_plan_when_alternative_is_not_faster(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=8.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)

        current = _edge_with_velocity(global_plan.trajectory, velocity=8.0, collision=False)
        similar = _edge_with_velocity(global_plan.trajectory, velocity=8.2, collision=False)

        planner.selected_local_plan = current
        planner._last_plan_change_time = 0.0

        assert planner.should_switch_plan(similar) is False
