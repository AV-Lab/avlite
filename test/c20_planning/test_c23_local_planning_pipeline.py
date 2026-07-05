"""Unit tests for the local planning pipeline and dual-role planners (c23)."""

import numpy as np

from avlite.c10_perception.c11_perception_model import EgoState, PerceptionModel
from avlite.c20_planning.c21_planning_model import GlobalPlan, LocalBehavior, LocalPlan
from avlite.c20_planning.c23_local_planning_strategy import (
    CruiseBehavioralPlanner,
    LocalBehavioralPlanningStrategy,
    LocalPathPlanningStrategy,
    LocalPlanningPipeline,
    LocalPlanningStrategy,
    LocalVelocityPlanningStrategy,
)
from avlite.c20_planning.c26_local_planners import VelocityLocalPlanner
from avlite.c20_planning.c27_local_lattice_planners import (
    GreedyLatticePlanner,
    LatticePlanningStrategy,
)
from avlite.c20_planning.c29_settings import PlanningSettingsSchema
from avlite.c50_common.c53_trajectory_tracker import TrajectoryTracker


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


class TestRegistrations:
    def test_velocity_planner_is_dual_role(self):
        assert "VelocityLocalPlanner" in LocalPlanningStrategy.registry
        assert "VelocityLocalPlanner" in LocalVelocityPlanningStrategy.registry
        assert issubclass(VelocityLocalPlanner, LocalPlanningStrategy)
        assert issubclass(VelocityLocalPlanner, LocalVelocityPlanningStrategy)

    def test_greedy_lattice_is_path_strategy(self):
        assert "GreedyLatticePlanner" in LocalPlanningStrategy.registry
        assert "GreedyLatticePlanner" in LocalPathPlanningStrategy.registry
        assert issubclass(GreedyLatticePlanner, LocalPathPlanningStrategy)

    def test_pipeline_and_behavioral_registered(self):
        assert "LocalPlanningPipeline" in LocalPlanningStrategy.registry
        assert "CruiseBehavioralPlanner" in LocalBehavioralPlanningStrategy.registry

    def test_abstract_intermediate_not_registered(self):
        assert "LatticePlanningStrategy" not in LocalPlanningStrategy.registry
        assert "LatticePlanningStrategy" not in LocalPathPlanningStrategy.registry


class TestDualRoleMethods:
    def test_behavioral_sets_intent(self):
        plan = LocalPlan()
        out = CruiseBehavioralPlanner().plan_behavior(plan)
        assert out.behavior is LocalBehavior.CRUISE

    def test_velocity_stage_profiles_in_place(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0))
        planner = VelocityLocalPlanner(global_plan=global_plan, env=pm)
        plan = LocalPlan.from_trajectory(global_plan.trajectory)
        out = planner.plan_velocity(plan)
        assert out.trajectory is not None
        assert len(out.velocity) == len(out.trajectory.velocity)

    def test_path_stage_fills_geometry(self):
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0))
        planner = GreedyLatticePlanner(global_plan=global_plan, env=pm)
        plan = LocalPlan()
        out = planner.plan_path(plan)
        assert out.trajectory is not None
        assert len(out.path) > 0


class TestLocalPlanningPipeline:
    def _pipeline(self, path="GreedyLatticePlanner", behavioral="", velocity=""):
        setting = PlanningSettingsSchema()
        setting.c23_behavioral_strategy = behavioral
        setting.c23_path_strategy = path
        setting.c23_velocity_strategy = velocity
        global_plan = _straight_global_plan()
        pm = PerceptionModel(ego_vehicle=EgoState(x=0.0, y=0.0, theta=0.0, velocity=5.0))
        return LocalPlanningPipeline(global_plan=global_plan, env=pm, setting=setting), pm

    def test_pipeline_runs_path_stage(self):
        pipeline, _ = self._pipeline(path="GreedyLatticePlanner")
        pipeline.replan()
        plan = pipeline.get_local_plan()
        assert isinstance(plan, LocalPlan)
        assert plan.as_trajectory() is not None

    def test_pipeline_full_stack(self):
        pipeline, _ = self._pipeline(
            path="GreedyLatticePlanner",
            behavioral="CruiseBehavioralPlanner",
            velocity="VelocityLocalPlanner",
        )
        pipeline.replan()
        plan = pipeline.get_local_plan()
        assert plan.behavior is LocalBehavior.CRUISE
        assert len(plan.velocity) > 0

    def test_pipeline_no_path_falls_back_to_global(self):
        pipeline, _ = self._pipeline(path="")
        pipeline.replan()
        plan = pipeline.get_local_plan()
        assert plan.as_trajectory() is not None

    def test_pipeline_capabilities(self):
        from avlite.c50_common.c51_capabilities import StackCapability

        pipeline, _ = self._pipeline()
        assert StackCapability.LOCAL_PLAN in pipeline.stack_capabilities

    def test_pipeline_step_advances_child(self):
        pipeline, _ = self._pipeline(path="GreedyLatticePlanner")
        pipeline.replan()
        state = EgoState(x=5.0, y=0.0, theta=0.0, velocity=5.0)
        pipeline.step(state)
        assert pipeline.location_xy == (5.0, 0.0)
