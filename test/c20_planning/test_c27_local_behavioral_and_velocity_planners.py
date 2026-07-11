"""Unit tests for the velocity local planner (avlite.c20_planning.c27_local_behavioral_and_velocity_planners)."""

import numpy as np

from avlite.c10_perception.c11_perception_model import AgentState, EgoState, PerceptionModel, SingleTrajectory
from avlite.c20_planning.c21_planning_model import GlobalPlan
from avlite.c20_planning.c27_local_behavioral_and_velocity_planners import VelocityLocalPlanner
from avlite.c20_planning.c29_settings import PlanningSettingsSchema
from avlite.c50_common.c54_trajectory_tracker import TrajectoryTracker
from avlite.c50_common.c55_collision_checking import check_collision


def _straight_global_plan(x_end: float = 100.0, n: int = 20, velocity: float = 5.0) -> GlobalPlan:
    xs = [x_end * i / (n - 1) for i in range(n)]
    path = [(x, 0.0) for x in xs]
    vel = [velocity] * n
    trajectory = TrajectoryTracker(path=path, velocity=vel)
    return GlobalPlan(start_point=path[0], goal_point=path[-1], path=path, velocity=vel, trajectory=trajectory)


def _planner_at_ego_x(global_plan: GlobalPlan, pm: PerceptionModel, ego_x: float) -> VelocityLocalPlanner:
    planner = VelocityLocalPlanner(global_plan=global_plan, env=pm)
    planner.global_trajectory.update_waypoint_by_xy(ego_x, 0.0)
    return planner


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

    def test_static_obstacle_reduces_velocity_before_collision(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[AgentState(x=50.0, y=0.0, theta=0.0, velocity=0.0, agent_id=1)],
        )
        planner = VelocityLocalPlanner(global_plan=global_plan, env=pm)
        planner.replan()

        velocity = np.asarray(planner.get_local_plan().velocity)
        hit, collision_idx, _ = check_collision(pm, global_plan.trajectory)
        assert hit is True
        assert np.mean(velocity) < np.mean(global_plan.velocity)
        assert velocity[collision_idx:].max() < 0.5

    def test_moving_obstacle_matches_agent_speed(self):
        global_plan = _straight_global_plan()
        agent_speed = 3.0
        agent = AgentState(x=50.0, y=0.0, theta=0.0, velocity=agent_speed, agent_id=1)
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=20.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[agent],
        )
        # Speed-match for movers needs prediction sweeps (no constant-velocity fallback).
        dt = 0.1
        n_steps = 50
        steps = np.empty((n_steps, 2))
        for t in range(n_steps):
            time = (t + 1) * dt
            steps[t, 0] = agent.x + agent.velocity * time
            steps[t, 1] = agent.y
        pm.prediction = SingleTrajectory(predict_delta_t=dt, trajectories={1: steps})
        planner = _planner_at_ego_x(global_plan, pm, ego_x=20.0)
        planner.replan()

        velocity = np.asarray(planner.get_local_plan().velocity)
        assert velocity[-1] > 1.5
        assert velocity[-1] <= agent_speed + 0.5

    def test_close_follow_static_lead_brakes_immediately(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=40.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[AgentState(x=50.0, y=0.0, theta=0.0, velocity=0.0, agent_id=1)],
        )
        planner = _planner_at_ego_x(global_plan, pm, ego_x=40.0)
        planner.replan()

        tj = planner.get_local_plan().as_trajectory()
        assert tj is not None
        assert tj.velocity[tj.current_wp] < 2.0

    def test_close_follow_moving_lead_decelerates_toward_agent(self):
        global_plan = _straight_global_plan()
        agent_speed = 3.0
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=40.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[AgentState(x=50.0, y=0.0, theta=0.0, velocity=agent_speed, agent_id=1)],
        )
        planner = _planner_at_ego_x(global_plan, pm, ego_x=40.0)
        planner.replan()

        tj = planner.get_local_plan().as_trajectory()
        assert tj is not None
        assert tj.velocity[tj.current_wp] <= 3.5

    def test_far_follow_moving_lead_still_brakes(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[AgentState(x=50.0, y=0.0, theta=0.0, velocity=0.0, agent_id=1)],
        )
        planner = VelocityLocalPlanner(global_plan=global_plan, env=pm)
        planner.replan()

        velocity = np.asarray(planner.get_local_plan().velocity)
        assert np.mean(velocity) < np.mean(global_plan.velocity)

    def test_comfortable_gap_keeps_global_speed(self):
        global_plan = _straight_global_plan(x_end=500.0, n=100, velocity=13.0)
        agent_speed = 4.6
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=50.0, y=0.0, theta=0.0, velocity=13.0),
            agent_vehicles=[AgentState(x=400.0, y=0.0, theta=0.0, velocity=agent_speed, agent_id=1)],
        )
        planner = _planner_at_ego_x(global_plan, pm, ego_x=50.0)
        planner.replan()

        tj = planner.get_local_plan().as_trajectory()
        assert tj is not None
        assert tj.velocity[tj.current_wp] >= 12.5

    def test_slightly_slower_than_lead_matches_without_emergency_decel(self):
        global_plan = _straight_global_plan(x_end=200.0, n=50, velocity=5.0)
        agent_speed = 4.9
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=48.0, y=0.0, theta=0.0, velocity=4.6),
            agent_vehicles=[AgentState(x=55.0, y=0.0, theta=0.0, velocity=agent_speed, agent_id=1)],
        )
        planner = _planner_at_ego_x(global_plan, pm, ego_x=48.0)
        planner.replan()

        tj = planner.get_local_plan().as_trajectory()
        assert tj is not None
        assert tj.velocity[tj.current_wp] >= 4.5
        assert tj.velocity[tj.current_wp] <= agent_speed + 0.2

    def test_registers_in_local_planner_registry(self):
        from avlite.c20_planning.c23_local_planning_strategy import LocalPlanningStrategy

        assert "VelocityLocalPlanner" in LocalPlanningStrategy.registry

    def test_replan_limits_local_plan_to_horizon(self):
        n = 200
        path = [(float(i), 0.0) for i in range(n)]
        velocity = [5.0] * n
        trajectory = TrajectoryTracker(path=path, velocity=velocity)
        global_plan = GlobalPlan(
            start_point=path[0], goal_point=path[-1], path=path, velocity=velocity, trajectory=trajectory
        )
        pm = PerceptionModel(ego_vehicle=EgoState(x=30.0, y=0.0, theta=0.0, velocity=5.0))
        setting = PlanningSettingsSchema(c27_planning_horizon_points=50)
        planner = VelocityLocalPlanner(global_plan=global_plan, env=pm, setting=setting)
        planner.global_trajectory.update_waypoint_by_xy(30.0, 0.0)
        planner.replan()

        local = planner.get_local_plan()
        assert len(local.path) == 50
        assert abs(local.path[0][0] - 30.0) < 1.0

    def test_get_local_plan_before_replan_returns_horizon(self):
        n = 200
        path = [(float(i), 0.0) for i in range(n)]
        velocity = [5.0] * n
        trajectory = TrajectoryTracker(path=path, velocity=velocity)
        global_plan = GlobalPlan(
            start_point=path[0], goal_point=path[-1], path=path, velocity=velocity, trajectory=trajectory
        )
        pm = PerceptionModel(ego_vehicle=EgoState(x=30.0, y=0.0, theta=0.0, velocity=5.0))
        setting = PlanningSettingsSchema(c27_planning_horizon_points=50)
        planner = VelocityLocalPlanner(global_plan=global_plan, env=pm, setting=setting)
        planner.global_trajectory.update_waypoint_by_xy(30.0, 0.0)

        local = planner.get_local_plan()
        assert len(local.path) == 50
        assert abs(local.path[0][0] - 30.0) < 1.0

    def test_apply_speed_match_static_obstacle_reduces_velocity(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=40.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[AgentState(x=50.0, y=0.0, theta=0.0, velocity=0.0, agent_id=1)],
        )
        planner = _planner_at_ego_x(global_plan, pm, ego_x=40.0)
        tj = TrajectoryTracker(path=list(global_plan.path), velocity=list(global_plan.velocity))
        tj.update_waypoint_by_xy(40.0, 0.0)
        collision_idx = int(np.argmin(np.abs(np.asarray(tj.path_x) - 50.0)))

        planner.apply_speed_match(tj, collision_idx, 0.0)

        velocity = np.asarray(tj.velocity)
        assert velocity[tj.current_wp] < 2.0
        assert velocity[collision_idx:].max() < 0.5

    def test_apply_speed_match_without_ref_velocity(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(
            ego_vehicle=EgoState(x=40.0, y=0.0, theta=0.0, velocity=5.0),
            agent_vehicles=[AgentState(x=50.0, y=0.0, theta=0.0, velocity=3.0, agent_id=1)],
        )
        planner = _planner_at_ego_x(global_plan, pm, ego_x=40.0)
        tj = TrajectoryTracker(path=list(global_plan.path), velocity=list(global_plan.velocity))
        tj.update_waypoint_by_xy(40.0, 0.0)
        collision_idx = int(np.argmin(np.abs(np.asarray(tj.path_x) - 50.0)))

        planner.apply_speed_match(tj, collision_idx, 3.0, ref_velocity=None)

        assert tj.velocity[tj.current_wp] <= 3.5


class TestCruiseBehavioralPlanner:
    def test_sets_cruise_behavior(self):
        from avlite.c20_planning.c21_planning_model import LocalBehavior, LocalPlan
        from avlite.c20_planning.c27_local_behavioral_and_velocity_planners import CruiseBehavioralPlanner

        plan = CruiseBehavioralPlanner().plan_behavior(LocalPlan())
        assert plan.behavior == LocalBehavior.CRUISE

    def test_registers_in_behavioral_registry(self):
        from avlite.c20_planning.c23_local_planning_strategy import LocalBehavioralPlanningStrategy

        assert "CruiseBehavioralPlanner" in LocalBehavioralPlanningStrategy.registry
