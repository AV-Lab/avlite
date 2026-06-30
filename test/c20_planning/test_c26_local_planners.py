"""Unit tests for velocity local planner (avlite.c20_planning.c26_local_planners)."""

import numpy as np

from avlite.c10_perception.c11_perception_model import AgentState, EgoState, PerceptionModel
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c20_planning.c26_local_planners import VelocityLocalPlanner
from avlite.c60_common.c63_trajectory_tracker import TrajectoryTracker


def _straight_global_plan(x_end: float = 100.0, n: int = 20, velocity: float = 5.0) -> GlobalPlan:
    xs = [x_end * i / (n - 1) for i in range(n)]
    path = [(x, 0.0) for x in xs]
    vel = [velocity] * n
    trajectory = TrajectoryTracker(path=path, velocity=vel)
    return GlobalPlan(start_point=path[0], goal_point=path[-1], path=path, velocity=vel, trajectory=trajectory)


class TestVelocityLocalPlanner:
    def test_no_obstacle_keeps_global_velocity(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[AgentState(x=50.0, y=20.0, theta=0.0, velocity=0.0, agent_id=1)],
        )
        planner = VelocityLocalPlanner(global_plan=global_plan, env=pm)
        planner.replan()

        local_plan = planner.get_local_plan()
        assert np.allclose(local_plan.velocity, global_plan.velocity)

    def test_obstacle_ahead_reduces_velocity_before_collision(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[AgentState(x=50.0, y=0.0, theta=0.0, velocity=0.0, agent_id=1)],
        )
        planner = VelocityLocalPlanner(global_plan=global_plan, env=pm)
        planner.replan()

        velocity = np.asarray(planner.get_local_plan().velocity)
        assert velocity[-1] < 1.0
        assert np.mean(velocity) < np.mean(global_plan.velocity)

    def test_registers_in_local_planner_registry(self):
        from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy

        assert "VelocityLocalPlanner" in LocalPlanningStrategy.registry
